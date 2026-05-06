"""
MultiShelfScan — TAVIS-HEAD Task (d).

Five YCB objects from the TAVIS-HEAD set are placed across a 3-shelf unit
(3 shelves, 3 slots each = 9 possible positions).  The shelves are widely
spaced vertically so the head camera can only see one shelf at a time —
the robot must pitch its head up/down to scan.

A language prompt names the target; the robot must locate it on the shelf,
retrieve it, and bring it towards its body (success: target x < threshold).

The shelf unit is built parametrically from CuboidCfg primitives (3 boards,
2 side panels, 1 back panel), all kinematic.
"""

import math
import random

import torch

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp

from isaaclab_arena.tasks.task_base import TaskBase

from ._common import (
    TAVIS_HEAD_OBJECT_NAMES,
    BaseSceneCfg,
    hide_object,
    make_object_cfg,
    place_object,
)


# ---------------------------------------------------------------------------
# Shelf geometry constants
# ---------------------------------------------------------------------------

# Shelf unit position (front face towards robot at origin)
_SHELF_CENTER_X = 0.52   # centre of shelf depth (x-axis = forward)
_SHELF_CENTER_Y = 0.00   # centred in front of robot
_SHELF_DEPTH    = 0.11   # x extent (shallow — easier gripper access)
_SHELF_WIDTH    = 0.70   # y extent

_BOARD_THICKNESS = 0.02
_SIDE_THICKNESS  = 0.02
_BACK_THICKNESS  = 0.01

# Shelf surface heights (z of each board centre)
_SHELF_HEIGHTS = (0.97, 1.10, 1.27)
# Top board = "ceiling" of the cabinet, above the top shelf
_TOP_BOARD_Z   = _SHELF_HEIGHTS[-1] + 0.25
_SHELF_BOTTOM  = _SHELF_HEIGHTS[0] - _BOARD_THICKNESS / 2
_SHELF_TOP     = _TOP_BOARD_Z + _BOARD_THICKNESS / 2
_SHELF_MID_Z   = (_SHELF_BOTTOM + _SHELF_TOP) / 2
_SHELF_SPAN_Z  = _SHELF_TOP - _SHELF_BOTTOM

# Object drop height above shelf board surface (keep low so objects don't
# hit the shelf above — inter-shelf gap is ~0.18m after board thickness)
_DROP_OFFSET = 0.04

# Slot positions on each shelf (y offsets: left / right only)
# Centre slot removed — not enough clearance for gripper approach
_SLOT_Y_OFFSETS = (0.15, -0.15)

# All 9 slots as (shelf_index, slot_y)
_ALL_SLOTS: list[tuple[int, float]] = [
    (si, sy) for si in range(3) for sy in _SLOT_Y_OFFSETS
]

# Shelf visual material
_SHELF_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.55, 0.35, 0.20),  # warm wood tone
    roughness=0.7,
    metallic=0.0,
)


def _shelf_board(name: str, z: float) -> AssetBaseCfg:
    """Horizontal shelf board at height *z* (centre of board)."""
    return AssetBaseCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_SHELF_CENTER_X, _SHELF_CENTER_Y, z],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_SHELF_DEPTH, _SHELF_WIDTH, _BOARD_THICKNESS),
            visual_material=_SHELF_MATERIAL,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


def _side_panel(name: str, y: float) -> AssetBaseCfg:
    """Vertical side panel at y-offset *y*."""
    return AssetBaseCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_SHELF_CENTER_X, y, _SHELF_MID_Z],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_SHELF_DEPTH, _SIDE_THICKNESS, _SHELF_SPAN_Z),
            visual_material=_SHELF_MATERIAL,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


def _back_panel() -> AssetBaseCfg:
    """Back panel (farthest from robot)."""
    back_x = _SHELF_CENTER_X + _SHELF_DEPTH / 2 + _BACK_THICKNESS / 2
    return AssetBaseCfg(
        prim_path="/World/envs/env_.*/shelf_back",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[back_x, _SHELF_CENTER_Y, _SHELF_MID_Z],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_BACK_THICKNESS, _SHELF_WIDTH, _SHELF_SPAN_Z),
            visual_material=_SHELF_MATERIAL,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


# ---------------------------------------------------------------------------
# Language prompts per object
# ---------------------------------------------------------------------------

PROMPTS: dict[str, list[str]] = {
    "soup_can": [
        "Find the tomato soup can on the shelf and bring it to me.",
        "Retrieve the soup can from the shelves.",
        "Look through the shelves, find the red soup can, and take it.",
    ],
    "meat_can": [
        "Find the potted meat can on the shelf and bring it to me.",
        "Retrieve the spam can from the shelves.",
        "Look through the shelves, find the meat can, and take it.",
    ],
    "tuna_fish_can": [
        "Find the tuna fish can on the shelf and bring it to me.",
        "Retrieve the tuna can from the shelves.",
        "Look through the shelves, find the tuna can, and take it.",
    ],
    "gelatin_box": [
        "Find the gelatin box on the shelf and bring it to me.",
        "Retrieve the gelatin box from the shelves.",
        "Look through the shelves, find the gelatin box, and take it.",
    ],
    "pudding_box": [
        "Find the pudding box on the shelf and bring it to me.",
        "Retrieve the pudding box from the shelves.",
        "Look through the shelves, find the pudding box, and take it.",
    ],
}


# ---------------------------------------------------------------------------
# Task variants
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "id": {
        # Objects always placed near slot centres
        "slot_jitter_y": 0.03,
        "slot_jitter_x": 0.03,
        # Success = object pulled past the front edge of the shelf
        "success_x_threshold": _SHELF_CENTER_X - _SHELF_DEPTH / 2 - 0.03,
    },
    "ood_spatial": {
        # More jitter — harder to predict exact position
        "slot_jitter_y": 0.10,
        "slot_jitter_x": 0.03,
        "success_x_threshold": _SHELF_CENTER_X - _SHELF_DEPTH / 2 - 0.03,
    },
}


# ---------------------------------------------------------------------------
# Event / termination functions
# ---------------------------------------------------------------------------

def _sample_trial(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    task: "MultiShelfScanTask",
) -> None:
    """Reset event: assign 5 objects to 5 of the 9 shelf slots.

    Per-env: each env_id gets an independent random trial so that
    auto-resets in parallel evaluation don't corrupt other envs' state.
    """
    v = task._effective_variant
    jitter_y = v["slot_jitter_y"]
    jitter_x = v["slot_jitter_x"]

    # Safe x/y bounds (keep objects inside the shelf with margin)
    x_min = _SHELF_CENTER_X - _SHELF_DEPTH / 2 + 0.02
    x_max = _SHELF_CENTER_X + _SHELF_DEPTH / 2 - 0.02
    y_inner = _SHELF_WIDTH / 2 - 0.03  # margin from side panels

    for eid in env_ids.tolist():
        eid_t = torch.tensor([eid], device=env_ids.device)

        # Pick 5 random slots (out of 9)
        chosen_slots = random.sample(_ALL_SLOTS, 5)
        assignment = list(zip(TAVIS_HEAD_OBJECT_NAMES, chosen_slots))

        # Choose target
        target = random.choice(TAVIS_HEAD_OBJECT_NAMES)
        task._env_state[eid] = {
            "target_object": target,
            "prompt": random.choice(PROMPTS[target]),
        }

        for name, (shelf_idx, slot_y) in assignment:
            shelf_z = _SHELF_HEIGHTS[shelf_idx] + _BOARD_THICKNESS / 2 + _DROP_OFFSET
            x = _SHELF_CENTER_X + random.uniform(-jitter_x, jitter_x)
            x = max(x_min, min(x_max, x))
            y = slot_y + random.uniform(-jitter_y, jitter_y)
            y = max(-y_inner, min(y_inner, y))
            yaw = random.uniform(-math.pi, math.pi)
            place_object(env, name, x, y, shelf_z, yaw, eid_t)

    # Backward compat: keep scalar attributes (reflects last-reset env)
    last = env_ids[-1].item()
    task.target_object = task._env_state[last]["target_object"]
    task._current_prompt = task._env_state[last]["prompt"]
    env.unwrapped.task = task


def _success(
    env: ManagerBasedRLEnv,
    max_velocity: float = 1.0,
) -> torch.Tensor:
    """True iff the target object is brought towards the robot (x < threshold)."""
    task = env.unwrapped.task
    threshold = task._effective_variant["success_x_threshold"]
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for eid in range(env.num_envs):
        state = task._env_state.get(eid)
        if state is None:
            continue
        obj = env.scene[state["target_object"]]
        x = obj.data.root_pos_w[eid, 0] - env.scene.env_origins[eid, 0]
        vel = torch.norm(obj.data.root_lin_vel_w[eid])
        result[eid] = (x < threshold) and (vel < max_velocity)
    return result


# ---------------------------------------------------------------------------
# Scene / termination / event configs
# ---------------------------------------------------------------------------

@configclass
class MultiShelfScanSceneCfg(BaseSceneCfg):
    """Shelf unit (no table) + all 5 TAVIS-HEAD YCB objects.

    Shelf boards, side panels, and back panel form the shelf unit.
    """

    # -- Override shadow light: tilt forward so it shines into the open
    #    front of the cabinet (from above-behind the robot) --
    shadow_light = AssetBaseCfg(
        prim_path="/World/shadow_light",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-1, 1, 3],
            rot=[0.928, -0.147, -0.338, 0.054],  # ~40° forward + ~18° side tilt from left
        ),
        spawn=sim_utils.DistantLightCfg(
            prim_type="DistantLight",
            intensity=1000.0,
            color=(1., 1., 1.),
            angle=0.5,
        ),
    )

    # -- Shelf structure (parametric cuboids) --
    shelf_board_0 = _shelf_board("shelf_board_0", _SHELF_HEIGHTS[0])
    shelf_board_1 = _shelf_board("shelf_board_1", _SHELF_HEIGHTS[1])
    shelf_board_2 = _shelf_board("shelf_board_2", _SHELF_HEIGHTS[2])
    shelf_board_top = _shelf_board("shelf_board_top", _TOP_BOARD_Z)

    shelf_side_left  = _side_panel("shelf_side_left",
                                   _SHELF_CENTER_Y + _SHELF_WIDTH / 2 + _SIDE_THICKNESS / 2)
    shelf_side_right = _side_panel("shelf_side_right",
                                   _SHELF_CENTER_Y - _SHELF_WIDTH / 2 - _SIDE_THICKNESS / 2)

    shelf_back = _back_panel()

    # -- YCB objects --
    soup_can = make_object_cfg("soup_can")
    meat_can = make_object_cfg("meat_can")
    tuna_fish_can = make_object_cfg("tuna_fish_can")
    gelatin_box = make_object_cfg("gelatin_box")
    pudding_box = make_object_cfg("pudding_box")


@configclass
class MultiShelfScanTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@configclass
class MultiShelfScanEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    sample_trial = EventTerm(func=_sample_trial, mode="reset", params={})


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class MultiShelfScanTask(TaskBase):
    """TAVIS-HEAD Task (d): multi-shelf vertical scan.

    Five YCB objects are distributed across a 3-shelf unit (9 possible
    slots).  The shelves are spaced widely enough that the head camera
    can only see one shelf at a time, requiring vertical scanning.

    A language prompt names the target; the robot must scan the shelves,
    locate the target, retrieve it, and bring it towards its body.

    Success: target object x-position < threshold (brought towards robot).

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
        episode_length_s: float = 120.0,
    ):
        super().__init__(episode_length_s=episode_length_s)
        effective = dict(self.VARIANTS[task_variant])
        if variant_overrides:
            effective.update(variant_overrides)
        self._effective_variant = effective

        self.target_object: str = TAVIS_HEAD_OBJECT_NAMES[0]
        self._current_prompt: str = PROMPTS[self.target_object][0]
        self._env_state: dict[int, dict] = {}

    def get_prompt(self, env_id: int | None = None) -> str:
        if env_id is not None and env_id in self._env_state:
            return self._env_state[env_id]["prompt"]
        return self._current_prompt

    def get_scene_cfg(self):
        return MultiShelfScanSceneCfg()

    def get_termination_cfg(self):
        return MultiShelfScanTerminationsCfg()

    def get_events_cfg(self):
        cfg = MultiShelfScanEventCfg()
        cfg.sample_trial.params["task"] = self
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str = None):
        return None

    def get_metrics(self) -> list:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(2.0, 0.0, 2.0), lookat=(0.5, 0.0, 1.0))
