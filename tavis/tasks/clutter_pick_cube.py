"""
ClutterPickCube — TAVIS-HEAD Task 3.

Four YCB distractor objects (randomly sampled from the 5 TAVIS-HEAD set,
same objects as ClutterPickLift) plus a red cube are randomly placed on the
table.  The target is always the red cube — no language conditioning on
object identity is needed.

The robot must: visually search the cluttered scene, locate the red cube,
grasp it and lift it.  Tests active gaze for *visual search* with a
visually distinct target.
"""

import random

import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from isaaclab_arena.tasks.task_base import TaskBase

from ._common import (
    RIGID_PROPS,
    TABLE_HEIGHT,
    TAVIS_HEAD_OBJECT_NAMES,
    SimpleTableSceneCfg,
    hide_object,
    make_object_cfg,
)


# ---------------------------------------------------------------------------
# Prompt — fixed (target is always the red cube)
# ---------------------------------------------------------------------------

PROMPT = "Find the red cube and pick it up."

# All scene object names (5 TAVIS-HEAD distractors + the cube)
_ALL_NAMES = TAVIS_HEAD_OBJECT_NAMES + ["red_cube"]


# ---------------------------------------------------------------------------
# Task variants  (same structure as ClutterPickLift)
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "id": {
        "pose_range": {
            "x": (0.35, 0.45),
            "y": (-0.25, 0.25),
            "z": (TABLE_HEIGHT + 0.1, TABLE_HEIGHT + 0.1),
            "yaw": (-3.14159, 3.14159),
            "roll": (-1.5708, -1.5708),
        },
        "min_separation": 0.1,
    },
    "ood_spatial": {
        "pose_range": {
            "x": (0.3, 0.5),
            "y": (-0.35, 0.35),
            "z": (TABLE_HEIGHT + 0.1, TABLE_HEIGHT + 0.1),
            "yaw": (-3.14159, 3.14159),
            "roll": (-1.5708, -1.5708),
        },
        "min_separation": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Event / termination functions
# ---------------------------------------------------------------------------

def _sample_trial(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    task: "ClutterPickCubeTask",
) -> None:
    """Reset event: pick 4 of 5 TAVIS-HEAD objects as distractors, hide the 5th."""
    selected = random.sample(TAVIS_HEAD_OBJECT_NAMES, 4)
    for name in TAVIS_HEAD_OBJECT_NAMES:
        if name not in selected:
            hide_object(env, name, env_ids)
    task._current_prompt = PROMPT
    env.unwrapped.task = task


def _success(
    env: ManagerBasedRLEnv,
    min_height: float = 1.2,
    max_velocity: float = 1.0,
) -> torch.Tensor:
    """True iff the red cube is above *min_height* with low velocity."""
    obj = env.scene["red_cube"]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    obj_vel = torch.norm(obj.data.root_lin_vel_w, dim=1)
    return torch.logical_and(obj_z > min_height, obj_vel < max_velocity)


# ---------------------------------------------------------------------------
# Scene / termination / event configs
# ---------------------------------------------------------------------------

_DROP_Z = TABLE_HEIGHT + 0.1


@configclass
class ClutterPickCubeSceneCfg(SimpleTableSceneCfg):
    """Table + 5 TAVIS-HEAD YCB objects (4 used as distractors, 1 hidden) + red cube."""

    soup_can = make_object_cfg("soup_can")
    meat_can = make_object_cfg("meat_can")
    tuna_fish_can = make_object_cfg("tuna_fish_can")
    gelatin_box = make_object_cfg("gelatin_box")
    pudding_box = make_object_cfg("pudding_box")

    red_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/red_cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.45, 0.0, _DROP_Z],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.055, 0.055, 0.055),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            rigid_props=RIGID_PROPS,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class ClutterPickCubeTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@configclass
class ClutterPickCubeEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_object_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {},
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg(name) for name in _ALL_NAMES],
        },
    )
    sample_trial = EventTerm(func=_sample_trial, mode="reset", params={})


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class ClutterPickCubeTask(TaskBase):
    """TAVIS-HEAD Task 3: clutter pick cube.

    Four of five TAVIS-HEAD YCB objects (randomly sampled) and a red cube are
    placed on the table; the fifth object is hidden.  The target is always the
    red cube.  No language conditioning on object identity — tests active gaze
    for visual search with a visually distinct target.

    Parameters
    ----------
    task_variant : "id" | "ood_spatial"
        Selects a preset from ``VARIANTS``.
    variant_overrides : dict | None
        Patches individual keys of the selected preset.
    """

    VARIANTS: dict[str, dict] = VARIANTS

    def __init__(
        self,
        task_variant: str = "id",
        variant_overrides: dict | None = None,
        episode_length_s: float = 100.0,
    ):
        super().__init__(episode_length_s=episode_length_s)
        effective = dict(self.VARIANTS[task_variant])
        if variant_overrides:
            effective.update(variant_overrides)
        self._effective_variant = effective

        self.target_object: str = "red_cube"
        self._current_prompt: str = PROMPT

    def get_prompt(self) -> str:
        return self._current_prompt

    def get_scene_cfg(self):
        return ClutterPickCubeSceneCfg()

    def get_termination_cfg(self):
        return ClutterPickCubeTerminationsCfg()

    def get_events_cfg(self):
        cfg = ClutterPickCubeEventCfg()
        cfg.sample_trial.params["task"] = self
        cfg.randomize_object_positions.params["pose_range"] = self._effective_variant["pose_range"]
        cfg.randomize_object_positions.params["min_separation"] = self._effective_variant["min_separation"]
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str = None):
        return None

    def get_metrics(self) -> list:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(2.0, 0.0, 2.0), lookat=(0.0, 0.0, 1.0))
