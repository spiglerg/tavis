"""
OccludedReach — TAVIS-HANDS Task 2.

A static narrow vertical screen / monitor sits on the table close to the
robot.  A single YCB object (sampled from the TAVIS-HEAD set) is placed
behind the screen.  The robot must reach around the screen with the closer
hand and grasp + lift the object.  The screen is intentionally narrow (~16
cm wide) so the robot's hands at default pose don't collide with it; the
wrist cameras observe the target as the arms move forward, while the head
camera mostly sees past the narrow obstacle on either side.

Tests *reach-around localization* under a static workspace obstacle.

NOTE — screen geometry
----------------------
The screen is built parametrically (panel + post + base) as a vertical /
portrait monitor.  A real Sketchfab Monitor.usdz asset lives at
``tavis/assets/tasks/monitor/Monitor.usdz`` but it is landscape
(4:3 aspect) and would either look small (uniform scale) or distorted
(non-uniform scale) once forced to portrait — see the
``asset_monitor_usdz.md`` reference note for the analysis and the swap-in
snippet if you ever want to try a USD asset here.
"""

import math
import random

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp

from isaaclab_arena.tasks.task_base import TaskBase

from ._common import (
    TABLE_HEIGHT,
    TAVIS_HEAD_OBJECT_NAMES,
    SimpleTableSceneCfg,
    hide_object,
    make_object_cfg,
    place_object,
)


# ---------------------------------------------------------------------------
# Prompt — fixed for every episode
# ---------------------------------------------------------------------------

PROMPT = "Reach around the screen and pick up the object behind it."


# ---------------------------------------------------------------------------
# Screen geometry constants
#
# Parametric monitor in PORTRAIT orientation: narrow PANEL on a short POST
# atop a flat BASE.  All dimensions in metres.  The panel is intentionally
# narrow (~16 cm wide) — the robot's hands at default pose are in front of
# the chest and a wider screen would collide with them.
# Top of panel sits at ~1.52 m above ground (well below the head cam at
# default pose, so the head still sees the workspace through the gaps on
# either side; the screen is an arm-path obstacle, not a vision blocker).
# ---------------------------------------------------------------------------

SCREEN_X         = 0.27       # close to the robot, well within elbow reach
SCREEN_Y         = 0.0

PANEL_WIDTH      = 0.16       # y extent — narrow (collision-safe with default hand pose)
PANEL_HEIGHT     = 0.40       # z extent — taller than wide → portrait
PANEL_THICKNESS  = 0.015      # x extent — very thin

POST_X           = 0.030
POST_Y           = 0.030
POST_HEIGHT      = 0.10

BASE_X           = 0.15
BASE_Y           = 0.10
BASE_HEIGHT      = 0.020

# Z-centres of each piece (above table top)
_BASE_Z   = TABLE_HEIGHT + BASE_HEIGHT / 2
_POST_Z   = TABLE_HEIGHT + BASE_HEIGHT + POST_HEIGHT / 2
_PANEL_Z  = TABLE_HEIGHT + BASE_HEIGHT + POST_HEIGHT + PANEL_HEIGHT / 2

_PANEL_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.05, 0.05, 0.05),  # dark "screen"
    roughness=0.4,
    metallic=0.0,
)
_FRAME_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.18, 0.18, 0.18),
    roughness=0.6,
    metallic=0.1,
)


def _kinematic_cuboid(name: str, size, pos, material) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="/World/envs/env_.*/" + name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(pos), rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=tuple(size),
            visual_material=material,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


# ---------------------------------------------------------------------------
# Task variants
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "id": {
        # Single target placed in the workspace BEHIND the screen.
        "obj_x_range": (0.40, 0.50),
        "obj_y_range": (-0.20, 0.20),
        "drop_z": TABLE_HEIGHT + 0.1,
    },
    "ood_spatial": {
        # Full position randomization range — no need to keep target visible.
        "obj_x_range": (0.36, 0.53),
        "obj_y_range": (-0.3, 0.3),
        "drop_z": TABLE_HEIGHT + 0.1,
    },
}


# ---------------------------------------------------------------------------
# Event / termination functions
# ---------------------------------------------------------------------------

def _sample_trial(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    task: "OccludedReachTask",
) -> None:
    """Reset event: pick 1 target object at a random spot behind the screen."""
    v = task._effective_variant

    for eid in env_ids.tolist():
        eid_t = torch.tensor([eid], device=env_ids.device)

        target = random.choice(TAVIS_HEAD_OBJECT_NAMES)
        task._env_state[eid] = {"target_object": target, "prompt": PROMPT}

        x = random.uniform(*v["obj_x_range"])
        y = random.uniform(*v["obj_y_range"])
        yaw = random.uniform(-math.pi, math.pi)
        place_object(env, target, x, y, v["drop_z"], yaw, eid_t)

        for name in TAVIS_HEAD_OBJECT_NAMES:
            if name != target:
                hide_object(env, name, eid_t)

    last = env_ids[-1].item()
    task.target_object = task._env_state[last]["target_object"]
    task._current_prompt = PROMPT
    env.unwrapped.task = task


def _success(
    env: ManagerBasedRLEnv,
    min_height: float = 1.25,
    max_velocity: float = 1.0,
) -> torch.Tensor:
    """True iff the target object is lifted above *min_height* with low velocity."""
    task = env.unwrapped.task
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for eid in range(env.num_envs):
        state = task._env_state.get(eid)
        if state is None:
            continue
        obj = env.scene[state["target_object"]]
        z = obj.data.root_pos_w[eid, 2] - env.scene.env_origins[eid, 2]
        vel = torch.norm(obj.data.root_lin_vel_w[eid])
        result[eid] = (z > min_height) and (vel < max_velocity)
    return result


# ---------------------------------------------------------------------------
# Scene / termination / event configs
# ---------------------------------------------------------------------------

@configclass
class OccludedReachSceneCfg(SimpleTableSceneCfg):
    """Table + 5 TAVIS-HEAD YCB objects (1 used as target, 4 hidden) +
    parametric monitor (panel on post on base) blocking the head view.
    """

    soup_can = make_object_cfg("soup_can")
    meat_can = make_object_cfg("meat_can")
    tuna_fish_can = make_object_cfg("tuna_fish_can")
    gelatin_box = make_object_cfg("gelatin_box")
    pudding_box = make_object_cfg("pudding_box")

    screen_base = _kinematic_cuboid(
        "screen_base",
        size=(BASE_X, BASE_Y, BASE_HEIGHT),
        pos=(SCREEN_X, SCREEN_Y, _BASE_Z),
        material=_FRAME_MATERIAL,
    )
    screen_post = _kinematic_cuboid(
        "screen_post",
        size=(POST_X, POST_Y, POST_HEIGHT),
        pos=(SCREEN_X, SCREEN_Y, _POST_Z),
        material=_FRAME_MATERIAL,
    )
    screen_panel = _kinematic_cuboid(
        "screen_panel",
        size=(PANEL_THICKNESS, PANEL_WIDTH, PANEL_HEIGHT),
        pos=(SCREEN_X, SCREEN_Y, _PANEL_Z),
        material=_PANEL_MATERIAL,
    )


@configclass
class OccludedReachTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@configclass
class OccludedReachEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    sample_trial = EventTerm(func=_sample_trial, mode="reset", params={})


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class OccludedReachTask(TaskBase):
    """TAVIS-HANDS Task 2: occluded reach.

    A static screen sits on the table between the robot's head and the
    workspace; the robot must reach around with the closer hand to grasp a
    single target object placed behind the screen.  The head camera cannot
    see the target — only the wrist cameras observe it as the arms move
    forward.

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

        self.target_object: str = TAVIS_HEAD_OBJECT_NAMES[0]
        self._current_prompt: str = PROMPT
        self._env_state: dict[int, dict] = {}

    def get_prompt(self, env_id: int | None = None) -> str:
        if env_id is not None and env_id in self._env_state:
            return self._env_state[env_id]["prompt"]
        return self._current_prompt

    def get_scene_cfg(self):
        return OccludedReachSceneCfg()

    def get_termination_cfg(self):
        return OccludedReachTerminationsCfg()

    def get_events_cfg(self):
        cfg = OccludedReachEventCfg()
        cfg.sample_trial.params["task"] = self
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str = None):
        return None

    def get_metrics(self) -> list:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(2.0, 0.0, 2.0), lookat=(0.0, 0.0, 1.0))
