"""
WaitThenAct — TAVIS-HEAD Task 2.

A sphere "lightbulb" (~3 cm radius) is placed on the table.  It starts
**red** and turns **green** after a random delay (2–6 s by default).
One YCB object from the TAVIS-HEAD set is placed in the centre area.

The robot must: (1) look at the light and monitor it, (2) when it turns
green, look at the object, (3) grasp and lift it.  Tests gaze for
*temporal monitoring*.
"""

import math
import random

import torch

from isaaclab.assets import RigidObjectCfg
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
    change_prim_color,
    hide_object,
    make_object_cfg,
    place_object,
)


# ---------------------------------------------------------------------------
# Prompt — fixed for every episode (only one object visible; challenge is
# temporal, not linguistic)
# ---------------------------------------------------------------------------

PROMPT = "Watch the red light. When it turns green, pick up the object."


# ---------------------------------------------------------------------------
# Task variants
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "id": {
        # Object in front, light farther back — non-overlapping x ranges.
        "object_zone": {"x": (0.35, 0.45), "y": (-0.12, 0.12)},
        "light_zone":  {"x": (0.6, 0.7), "y": (-0.10, 0.10)},
        "delay_range_s": (2.0, 5.0),
        "drop_z": TABLE_HEIGHT + 0.1,
    },
    "ood_spatial": {
        "object_zone": {"x": (0.30, 0.50), "y": (-0.2, 0.2)},
        "light_zone":  {"x": (0.55, 0.75), "y": (-0.15, 0.15)},
        "delay_range_s": (2.0, 8.0),    ## in ood-spatial?  maybe should be 'ood task'?
        "drop_z": TABLE_HEIGHT + 0.1,
    },
}


# ---------------------------------------------------------------------------
# Event / termination functions
# ---------------------------------------------------------------------------

def _sample_trial(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    task: "WaitThenActTask",
) -> None:
    """Reset event: pick 1 object, hide others, place light, sample delay.

    Per-env: each env_id gets an independent random trial so that
    auto-resets in parallel evaluation don't corrupt other envs' state.
    """
    v = task._effective_variant

    for eid in env_ids.tolist():
        eid_t = torch.tensor([eid], device=env_ids.device)

        target = random.choice(TAVIS_HEAD_OBJECT_NAMES)
        delay_s = random.uniform(*v["delay_range_s"])

        task._env_state[eid] = {
            "target_object": target,
            "prompt": PROMPT,
            "light_is_green": False,
            "green_step": int(delay_s / env.step_dt),
        }

        # Place the selected object
        oz = v["object_zone"]
        x = random.uniform(*oz["x"])
        y = random.uniform(*oz["y"])
        yaw = random.uniform(-math.pi, math.pi)
        place_object(env, target, x, y, v["drop_z"], yaw, eid_t)

        # Hide the other 4 objects
        for name in TAVIS_HEAD_OBJECT_NAMES:
            if name != target:
                hide_object(env, name, eid_t)

        # Place the light sphere on the table
        lz = v["light_zone"]
        lx = random.uniform(*lz["x"])
        ly = random.uniform(*lz["y"])
        light = env.scene["signal_light"]
        pose = light.data.default_root_state[eid_t, :7].clone()
        pose[:, 0] = lx + env.scene.env_origins[eid_t, 0]
        pose[:, 1] = ly + env.scene.env_origins[eid_t, 1]
        pose[:, 2] = TABLE_HEIGHT + 0.03 + env.scene.env_origins[eid_t, 2]
        pose[:, 3] = 1.0
        pose[:, 4:7] = 0.0
        light.write_root_pose_to_sim(pose, eid_t)

        # Reset light to red
        change_prim_color(env, f"/World/envs/env_{eid}/signal_light", (1.0, 0.0, 0.0))

    # Backward compat: keep scalar attributes (reflects last-reset env)
    last = env_ids[-1].item()
    task.target_object = task._env_state[last]["target_object"]
    task._current_prompt = PROMPT
    task.light_is_green = False
    task.green_step = task._env_state[last]["green_step"]
    env.unwrapped.task = task


def _check_and_update(
    env: ManagerBasedRLEnv,
    min_height: float = 1.2,
    max_velocity: float = 1.0,
) -> torch.Tensor:
    """Termination: flip light colour at green_step, then check lift success.

    The light transitions from red to green once the sampled delay elapses.
    Success requires the light to be green AND the object to be lifted.
    Per-env state ensures parallel envs don't interfere with each other.
    """
    task = env.unwrapped.task
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for eid in range(env.num_envs):
        state = task._env_state.get(eid)
        if state is None:
            continue

        # Flip light from red to green once the delay has elapsed
        if not state["light_is_green"] and env.episode_length_buf[eid] >= state["green_step"]:
            state["light_is_green"] = True
            change_prim_color(env, f"/World/envs/env_{eid}/signal_light", (0.0, 1.0, 0.0))

        if not state["light_is_green"]:
            continue

        obj = env.scene[state["target_object"]]
        z = obj.data.root_pos_w[eid, 2] - env.scene.env_origins[eid, 2]
        vel = torch.norm(obj.data.root_lin_vel_w[eid])
        result[eid] = (z > min_height) and (vel < max_velocity)

    return result


# ---------------------------------------------------------------------------
# Scene / termination / event configs
# ---------------------------------------------------------------------------

_LIGHT_Z = TABLE_HEIGHT + 0.03  # sphere centre: bottom touches table


@configclass
class WaitThenActSceneCfg(SimpleTableSceneCfg):
    """Table + 5 YCB objects + signal light (sphere)."""

    soup_can = make_object_cfg("soup_can")
    meat_can = make_object_cfg("meat_can")
    tuna_fish_can = make_object_cfg("tuna_fish_can")
    gelatin_box = make_object_cfg("gelatin_box")
    pudding_box = make_object_cfg("pudding_box")

    signal_light = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/signal_light",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.52, 0.0, _LIGHT_Z],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )


@configclass
class WaitThenActTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_check_and_update, params={})


@configclass
class WaitThenActEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    sample_trial = EventTerm(func=_sample_trial, mode="reset", params={})


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class WaitThenActTask(TaskBase):
    """TAVIS-HEAD Task 2: wait then act.

    A signal light starts red and turns green after a random delay.
    The robot must monitor the light and grasp the object only after
    the light turns green.  Tests gaze for *temporal monitoring*.

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
        self.light_is_green: bool = False
        self.green_step: int = 0
        self._env_state: dict[int, dict] = {}

    def get_prompt(self, env_id: int | None = None) -> str:
        if env_id is not None and env_id in self._env_state:
            return self._env_state[env_id]["prompt"]
        return self._current_prompt

    def get_scene_cfg(self):
        return WaitThenActSceneCfg()

    def get_termination_cfg(self):
        return WaitThenActTerminationsCfg()

    def get_events_cfg(self):
        cfg = WaitThenActEventCfg()
        cfg.sample_trial.params["task"] = self
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str = None):
        return None

    def get_metrics(self) -> list:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(2.0, 0.0, 2.0), lookat=(0.0, 0.0, 1.0))
