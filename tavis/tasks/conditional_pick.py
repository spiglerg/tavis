"""
ConditionalPick — TAVIS-HEAD Task 1.

A flat card (thin cuboid, ~8x5 cm) is placed in the centre of the workspace.
It is either **red** (pick the left object) or **green** (pick the right
object).  Two YCB objects from the TAVIS-HEAD set are placed left and right
of the card.

The robot must: (1) look at the card to read the colour, (2) look at the
indicated object, (3) grasp and lift it.  Tests gaze for *information
gathering*.
"""

import math
import random

import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm

from tavis.mdp import reset_scene_to_default_safe
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp

from isaaclab_arena.tasks.task_base import TaskBase

from ._common import (
    TABLE_HEIGHT,
    TAVIS_HEAD_OBJECT_NAMES,
    SimpleTableSceneCfg,
    change_prim_color,
    hide_object,
    make_object_cfg,
    place_object,
)


# ---------------------------------------------------------------------------
# Prompt — fixed for every episode (the challenge is visual, not linguistic)
# ---------------------------------------------------------------------------

PROMPT = "Look at the card. If it is red, pick the object on the left. If it is green, pick the object on the right."


# ---------------------------------------------------------------------------
# Task variants
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "id": {
        "left_zone": {"x": (0.35, 0.45), "y": (0.15, 0.25)},
        "right_zone": {"x": (0.35, 0.45), "y": (-0.25, -0.15)},
        "center_zone": {"x": (0.35, 0.45), "y": (-0.05, 0.05)},
        "drop_z": TABLE_HEIGHT + 0.1,
    },
    "ood_spatial": {
        "left_zone": {"x": (0.30, 0.50), "y": (0.10, 0.35)},
        "right_zone": {"x": (0.30, 0.50), "y": (-0.35, -0.10)},
        "center_zone": {"x": (0.30, 0.50), "y": (-0.08, 0.08)},
        "drop_z": TABLE_HEIGHT + 0.1,
    },
}


# ---------------------------------------------------------------------------
# Event / termination functions
# ---------------------------------------------------------------------------

def _sample_trial(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    task: "ConditionalPickTask",
) -> None:
    """Reset event: pick 2 objects, assign left/right, set card colour.

    Per-env: each env_id gets an independent random trial so that
    auto-resets in parallel evaluation don't corrupt other envs' state.
    """
    v = task._effective_variant
    drop_z = v["drop_z"]

    for eid in env_ids.tolist():
        eid_t = torch.tensor([eid], device=env_ids.device)

        selected = random.sample(TAVIS_HEAD_OBJECT_NAMES, 2)
        left_obj, right_obj = selected
        card_is_red = random.choice([True, False])
        target = left_obj if card_is_red else right_obj

        task._env_state[eid] = {"target_object": target, "prompt": PROMPT}

        # Place the two selected objects in their respective zones
        for name, zone_key in [(left_obj, "left_zone"),
                               (right_obj, "right_zone")]:
            zone = v[zone_key]
            x = random.uniform(*zone["x"])
            y = random.uniform(*zone["y"])
            yaw = random.uniform(-math.pi, math.pi)
            place_object(env, name, x, y, drop_z, yaw, eid_t)

        # Hide the remaining 3 objects
        for name in TAVIS_HEAD_OBJECT_NAMES:
            if name not in selected:
                hide_object(env, name, eid_t)

        # Place the card flat on the table in the centre zone
        zone = v["center_zone"]
        cx = random.uniform(*zone["x"])
        cy = random.uniform(*zone["y"])
        card = env.scene["cue_card"]
        pose = card.data.default_root_state[eid_t, :7].clone()
        pose[:, 0] = cx + env.scene.env_origins[eid_t, 0]
        pose[:, 1] = cy + env.scene.env_origins[eid_t, 1]
        pose[:, 2] = TABLE_HEIGHT + 0.002 + env.scene.env_origins[eid_t, 2]
        pose[:, 3] = 1.0
        pose[:, 4:7] = 0.0
        card.write_root_pose_to_sim(pose, eid_t)

        # Set card colour (red or green)
        color = (1.0, 0.0, 0.0) if card_is_red else (0.0, 1.0, 0.0)
        change_prim_color(env, f"/World/envs/env_{eid}/cue_card", color)

    # Backward compat: keep scalar attributes (reflects last-reset env)
    last = env_ids[-1].item()
    task.target_object = task._env_state[last]["target_object"]
    task._current_prompt = PROMPT
    env.unwrapped.task = task


def _success(
    env: ManagerBasedRLEnv,
    min_height: float = 1.2,
    max_velocity: float = 1.0,
) -> torch.Tensor:
    """True iff the target object is above *min_height* with low velocity."""
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

_CARD_Z = TABLE_HEIGHT + 0.002


@configclass
class ConditionalPickSceneCfg(SimpleTableSceneCfg):
    """Table + 5 YCB objects + cue card (thin cuboid)."""

    soup_can = make_object_cfg("soup_can")
    meat_can = make_object_cfg("meat_can")
    tuna_fish_can = make_object_cfg("tuna_fish_can")
    gelatin_box = make_object_cfg("gelatin_box")
    pudding_box = make_object_cfg("pudding_box")

    cue_card = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cue_card",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.40, 0.0, _CARD_Z],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.05, 0.003),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )


@configclass
class ConditionalPickTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@configclass
class ConditionalPickEventCfg:
    reset_all = EventTerm(func=reset_scene_to_default_safe, mode="reset")
    sample_trial = EventTerm(func=_sample_trial, mode="reset", params={})


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class ConditionalPickTask(TaskBase):
    """TAVIS-HEAD Task 1: conditional pick.

    A coloured card indicates which of two objects to grasp.  Red = left,
    green = right.  Tests gaze for *information gathering*.

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

        self.left_object: str = TAVIS_HEAD_OBJECT_NAMES[0]
        self.right_object: str = TAVIS_HEAD_OBJECT_NAMES[1]
        self.card_is_red: bool = True
        self.target_object: str = self.left_object
        self._current_prompt: str = PROMPT
        self._env_state: dict[int, dict] = {}

    def get_prompt(self, env_id: int | None = None) -> str:
        if env_id is not None and env_id in self._env_state:
            return self._env_state[env_id]["prompt"]
        return self._current_prompt

    def get_scene_cfg(self):
        return ConditionalPickSceneCfg()

    def get_termination_cfg(self):
        return ConditionalPickTerminationsCfg()

    def get_events_cfg(self):
        cfg = ConditionalPickEventCfg()
        cfg.sample_trial.params["task"] = self
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str = None):
        return None

    def get_metrics(self) -> list:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(2.0, 0.0, 2.0), lookat=(0.0, 0.0, 1.0))
