"""
ClutterPickLift — TAVIS-HEAD Task 0.

Five YCB objects randomly placed on the table (rejection-sampled, minimum
separation).  In the 'id' variant most objects are within the head camera
FOV at the default head pose; in 'ood_spatial' the wider spread means some
objects require scanning / active gaze to find.
The language prompt names the target; the robot must lift it above 1.3 m.
"""

import random

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm

from tavis.mdp import reset_scene_to_default_safe
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from isaaclab_arena.tasks.task_base import TaskBase

from ._common import TABLE_HEIGHT, TAVIS_HEAD_OBJECT_NAMES, SimpleTableSceneCfg, make_object_cfg


# ---------------------------------------------------------------------------
# Task-specific language prompts per object (for clutter pick-lift)
# ---------------------------------------------------------------------------

PROMPTS: dict[str, list[str]] = {
    "soup_can": [
        "Pick up the tomato soup can and lift it.",
        "Grasp the soup can and hold it up.",
        "Lift the red soup can off the table.",
    ],
    "meat_can": [
        "Pick up the potted meat can and lift it.",
        "Grasp the can of spam and hold it up.",
        "Lift the meat can off the table.",
    ],
    "tuna_fish_can": [
        "Pick up the tuna fish can and lift it.",
        "Grasp the tuna can and hold it up.",
        "Lift the tuna fish can off the table.",
    ],
    "gelatin_box": [
        "Pick up the gelatin box and lift it.",
        "Grasp the gelatin box and hold it up.",
        "Lift the gelatin box off the table.",
    ],
    "pudding_box": [
        "Pick up the pudding box and lift it.",
        "Grasp the pudding box and hold it up.",
        "Lift the pudding box off the table.",
    ],
}


# ---------------------------------------------------------------------------
# Task variants
#
# task_variant selects a preset; variant_overrides patches individual keys.
# Keys:
#   pose_range     – passed to randomize_object_pose (x/y/z/yaw tuples).
#                    z is a drop height; objects settle onto the table under
#                    gravity before recording starts.  Tune if needed.
#   min_separation – minimum 2-D centre-to-centre distance between objects (m).
#
# ood_spatial is INCLUSIVE (contains the id region) to avoid an artificial
# annular layout at evaluation time.
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "id": {
        "pose_range": {
            "x": (0.35, 0.45),
            "y": (-0.25, 0.25),
            "z": (TABLE_HEIGHT + 0.1, TABLE_HEIGHT + 0.1),
            "yaw": (-3.14159, 3.14159),
            "roll": (-1.5708, -1.5708),  # keep upright
        },
        "min_separation": 0.1,
    },
    "ood_spatial": {
        "pose_range": {
            "x": (0.3, 0.5),
            "y": (-0.35, 0.35),
            "z": (TABLE_HEIGHT + 0.1, TABLE_HEIGHT + 0.1),
            "yaw": (-3.14159, 3.14159),
            "roll": (-1.5708, -1.5708),  # keep upright
        },
        "min_separation": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Event / termination functions
# ---------------------------------------------------------------------------

def _sample_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    task: "ClutterPickLiftTask",
) -> None:
    """Reset event: sample a random target object and one of its prompts.

    Per-env state is stored in task._env_state so that parallel envs
    (num_envs > 1) each track their own target independently.
    """
    for eid in env_ids.tolist():
        target = random.choice(TAVIS_HEAD_OBJECT_NAMES)
        task._env_state[eid] = {
            "target_object": target,
            "prompt": random.choice(PROMPTS[target]),
        }
    # Backward compat: keep scalar attributes (reflects last-reset env)
    last = env_ids[-1].item()
    task.target_object = task._env_state[last]["target_object"]
    task._current_prompt = task._env_state[last]["prompt"]
    env.unwrapped.task = task


def _success(
    env: ManagerBasedRLEnv,
    min_height: float = 1.2,
    max_velocity: float = 1.0,
) -> torch.Tensor:
    """True iff the TARGET object is above min_height with low velocity."""
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
class ClutterPickLiftSceneCfg(SimpleTableSceneCfg):
    """SimpleTableSceneCfg + all 5 TAVIS-HEAD YCB objects.

    init_state positions are overridden by the randomize event each reset;
    the values here are just non-catastrophic defaults.
    """
    soup_can = make_object_cfg("soup_can")
    meat_can = make_object_cfg("meat_can")
    tuna_fish_can = make_object_cfg("tuna_fish_can")
    gelatin_box = make_object_cfg("gelatin_box")
    pudding_box = make_object_cfg("pudding_box")


@configclass
class ClutterPickLiftTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})  # task injected in get_termination_cfg()


@configclass
class ClutterPickLiftEventCfg:
    reset_all = EventTerm(func=reset_scene_to_default_safe, mode="reset")
    sample_target = EventTerm(func=_sample_target, mode="reset", params={})  # task injected
    randomize_object_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {},    # filled in get_events_cfg()
            "min_separation": 0.0,  # filled in get_events_cfg()
            "asset_cfgs": [SceneEntityCfg(name) for name in TAVIS_HEAD_OBJECT_NAMES],
        },
    )


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class ClutterPickLiftTask(TaskBase):
    """TAVIS-HEAD Task 0: clutter pick-lift.

    Five YCB objects are randomly placed on the table (rejection sampling,
    minimum separation).  The language prompt names the target; lift it
    above 1.3 m.  In the 'ood_spatial' variant the wider spread may require
    active gaze to locate the target.

    Also serves as the base scene for TAVIS-SOCIAL (GALT metric overlay,
    no scene changes needed).

    Parameters
    ----------
    task_variant : "id" | "ood_spatial"
        Selects a preset from ``VARIANTS``.
    variant_overrides : dict | None
        Patches individual keys of the selected preset, e.g.
        ``{"min_separation": 0.12}``.  Only specified keys are overridden.
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

        # Updated each episode by the sample_target event.
        self.target_object: str = TAVIS_HEAD_OBJECT_NAMES[0]
        self._current_prompt: str = PROMPTS[self.target_object][0]
        self._env_state: dict[int, dict] = {}

    def get_prompt(self, env_id: int | None = None) -> str:
        if env_id is not None and env_id in self._env_state:
            return self._env_state[env_id]["prompt"]
        return self._current_prompt

    def get_scene_cfg(self):
        return ClutterPickLiftSceneCfg()

    def get_termination_cfg(self):
        return ClutterPickLiftTerminationsCfg()

    def get_events_cfg(self):
        cfg = ClutterPickLiftEventCfg()
        cfg.sample_target.params["task"] = self
        cfg.randomize_object_positions.params["pose_range"] = self._effective_variant["pose_range"]
        cfg.randomize_object_positions.params["min_separation"] = self._effective_variant["min_separation"]
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str = None):
        return None

    def get_metrics(self) -> list:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(2.0, 0.0, 2.0), lookat=(0.0, 0.0, 1.0))
