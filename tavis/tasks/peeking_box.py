"""
PeekingBox — TAVIS-HANDS Task 1.

A box sits on the table with one side open (left or right, randomized per
episode).  A target object is inside the box.  The robot's head camera sees
only the closed outside of the box; the wrist cameras must determine which
side is open and the appropriate hand reaches in to grasp + lift the object.

The box is built from 4 kinematic walls (back, front, top, and one side):
the bottom is the table surface itself.  Both side walls are spawned but
one is hidden (teleported below ground) each episode so that side is open.
A mild yaw randomization (±10° in id, larger in ood) avoids canonical-pose
overfitting.

Tests *bimanual perception* under static container occlusion.
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
    hide_object,
    make_object_cfg,
    place_object,
)


# ---------------------------------------------------------------------------
# Prompt — fixed for every episode (the challenge is perceptual, not linguistic)
# ---------------------------------------------------------------------------

PROMPT = "Retrieve the object from inside the box."


# ---------------------------------------------------------------------------
# Box geometry constants (box-local frame, before yaw)
#
# Box centre = (BOX_CENTER_X, BOX_CENTER_Y) on the table; bottom = table top.
# Half-extents along box-local x (depth, toward robot) and y (sideways).
# ---------------------------------------------------------------------------

BOX_CENTER_X    = 0.43      # workspace centre, comfortably reachable
BOX_CENTER_Y    = 0.0
BOX_HALF_DEPTH  = 0.2      # half-extent along box-local x (depth)
BOX_HALF_WIDTH  = 0.07      # half-extent along box-local y (lateral)
BOX_HEIGHT      = 0.20      # interior height (top wall sits at table_top + this)
WALL_THICKNESS  = 0.008

# Sink the box 5mm into the table top to avoid a visible floating gap
# (covers the PhysX collision contact offset / table-top z imprecision).
_BOX_BOTTOM_Z   = TABLE_HEIGHT - 0.005
_BOX_TOP_Z      = _BOX_BOTTOM_Z + BOX_HEIGHT

# Wall visual material (warm wood-ish, like the multi-shelf cabinet)
_BOX_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.55, 0.35, 0.20),
    roughness=0.7,
    metallic=0.0,
)


def _wall_cfg(name: str, size: tuple[float, float, float],
              default_pos: tuple[float, float, float]) -> RigidObjectCfg:
    """Kinematic wall (cuboid) — pose written each reset."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(default_pos), rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=_BOX_MATERIAL,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


_HIDE_POS = (-2.5, 0.0, -1.0)  # below ground, behind robot

_ALL_BOX_WALL_NAMES = (
    "box_wall_back", "box_wall_front", "box_wall_top",
    "box_wall_pos_y", "box_wall_neg_y",
)


# ---------------------------------------------------------------------------
# Task variants
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "id": {
        # Box centre randomization (small)
        "box_x_range":   (BOX_CENTER_X - 0.02, BOX_CENTER_X + 0.02),
        "box_y_range":   (BOX_CENTER_Y - 0.04, BOX_CENTER_Y + 0.04),
        "box_yaw_range": (-math.radians(5), math.radians(5)),
        # Object position INSIDE the box (box-local frame, depth-randomized)
        "obj_dx_range":  (-0.04, 0.04),
        "obj_dy_range":  (-0.02, 0.02),
    },
    "ood_spatial": {
        "box_x_range":   (BOX_CENTER_X - 0.04, BOX_CENTER_X + 0.04),
        "box_y_range":   (BOX_CENTER_Y - 0.04, BOX_CENTER_Y + 0.04),
        "box_yaw_range": (-math.radians(10), math.radians(10)),
        "obj_dx_range":  (-0.04, 0.04),
        "obj_dy_range":  (-0.04, 0.04),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for a pure z-axis rotation."""
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


def _write_wall_pose(env, name: str, eid_t: torch.Tensor,
                     wx: float, wy: float, wz: float, yaw: float) -> None:
    asset = env.scene[name]
    pose = asset.data.default_root_state[eid_t, :7].clone()
    qw, qx, qy, qz = _yaw_quat(yaw)
    pose[:, 0] = wx + env.scene.env_origins[eid_t, 0]
    pose[:, 1] = wy + env.scene.env_origins[eid_t, 1]
    pose[:, 2] = wz + env.scene.env_origins[eid_t, 2]
    pose[:, 3] = qw
    pose[:, 4] = qx
    pose[:, 5] = qy
    pose[:, 6] = qz
    asset.write_root_pose_to_sim(pose, eid_t)


def _hide_wall(env, name: str, eid_t: torch.Tensor) -> None:
    asset = env.scene[name]
    pose = asset.data.default_root_state[eid_t, :7].clone()
    pose[:, 0] = _HIDE_POS[0] + env.scene.env_origins[eid_t, 0]
    pose[:, 1] = _HIDE_POS[1] + env.scene.env_origins[eid_t, 1]
    pose[:, 2] = _HIDE_POS[2] + env.scene.env_origins[eid_t, 2]
    pose[:, 3] = 1.0
    pose[:, 4:7] = 0.0
    asset.write_root_pose_to_sim(pose, eid_t)


# ---------------------------------------------------------------------------
# Event / termination functions
# ---------------------------------------------------------------------------

def _sample_trial(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    task: "PeekingBoxTask",
) -> None:
    """Reset event: sample box pose, target object, open side; place walls + object."""
    v = task._effective_variant

    for eid in env_ids.tolist():
        eid_t = torch.tensor([eid], device=env_ids.device)

        # --- sample episode parameters ---
        cx = random.uniform(*v["box_x_range"])
        cy = random.uniform(*v["box_y_range"])
        yaw = random.uniform(*v["box_yaw_range"])
        open_side = random.choice(["+y", "-y"])  # which side is open
        target = random.choice(TAVIS_HEAD_OBJECT_NAMES)

        task._env_state[eid] = {
            "target_object": target,
            "prompt": PROMPT,
            "open_side": open_side,
            "box_yaw": yaw,
            "box_center": (cx, cy),
        }

        # --- wall placements (box-local offsets, then rotated by yaw) ---
        # Back wall (+x face)
        bx, by = _rotate_xy(BOX_HALF_DEPTH + WALL_THICKNESS / 2, 0.0, yaw)
        _write_wall_pose(env, "box_wall_back", eid_t,
                         cx + bx, cy + by, _BOX_BOTTOM_Z + BOX_HEIGHT / 2, yaw)
        # Front wall (-x face, toward robot)
        fx, fy = _rotate_xy(-(BOX_HALF_DEPTH + WALL_THICKNESS / 2), 0.0, yaw)
        _write_wall_pose(env, "box_wall_front", eid_t,
                         cx + fx, cy + fy, _BOX_BOTTOM_Z + BOX_HEIGHT / 2, yaw)
        # Top wall
        _write_wall_pose(env, "box_wall_top", eid_t,
                         cx, cy, _BOX_TOP_Z + WALL_THICKNESS / 2, yaw)
        # Side walls (+y / -y in box-local frame)
        for wall_name, sign in [("box_wall_pos_y", +1), ("box_wall_neg_y", -1)]:
            sx, sy = _rotate_xy(0.0, sign * (BOX_HALF_WIDTH + WALL_THICKNESS / 2), yaw)
            _write_wall_pose(env, wall_name, eid_t,
                             cx + sx, cy + sy, _BOX_BOTTOM_Z + BOX_HEIGHT / 2, yaw)

        # Hide the wall on the open side
        hidden_wall = "box_wall_pos_y" if open_side == "+y" else "box_wall_neg_y"
        _hide_wall(env, hidden_wall, eid_t)

        # --- target object placement (inside box, box-local frame) ---
        dx = random.uniform(*v["obj_dx_range"])
        dy = random.uniform(*v["obj_dy_range"])
        ox, oy = _rotate_xy(dx, dy, yaw)
        wx = cx + ox
        wy = cy + oy
        # drop slightly above table so gravity settles it on the table inside the box
        oz = TABLE_HEIGHT + 0.05
        obj_yaw = random.uniform(-math.pi, math.pi)
        place_object(env, target, wx, wy, oz, obj_yaw, eid_t)

        # Hide the other 4 objects
        for name in TAVIS_HEAD_OBJECT_NAMES:
            if name != target:
                hide_object(env, name, eid_t)

    # Backward compat: scalar attributes (reflects last-reset env)
    last = env_ids[-1].item()
    task.target_object = task._env_state[last]["target_object"]
    task._current_prompt = PROMPT
    env.unwrapped.task = task


def _success(
    env: ManagerBasedRLEnv,
    min_height: float = 1.25,
    max_velocity: float = 1.0,
) -> torch.Tensor:
    """True iff target object is lifted above the box (out of the container)."""
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

# Wall sizes (in box-local frame, before yaw rotation)
# Front/back panels extend the full width of the top panel so they fully
# cover the side walls' outer edges from a head-on view (otherwise you can
# see directly which side wall is missing without using the wrist cams).
_BACK_FRONT_SIZE = (WALL_THICKNESS,
                    2 * BOX_HALF_WIDTH + 2 * WALL_THICKNESS,
                    BOX_HEIGHT)
_TOP_SIZE        = (2 * BOX_HALF_DEPTH + 2 * WALL_THICKNESS,
                    2 * BOX_HALF_WIDTH + 2 * WALL_THICKNESS,
                    WALL_THICKNESS)
_SIDE_SIZE       = (2 * BOX_HALF_DEPTH, WALL_THICKNESS, BOX_HEIGHT)


@configclass
class PeekingBoxSceneCfg(SimpleTableSceneCfg):
    """Table + 5 TAVIS-HEAD YCB objects (1 used as target, 4 hidden) +
    5 kinematic walls forming a box on the table (one side hidden each episode).
    """

    soup_can = make_object_cfg("soup_can")
    meat_can = make_object_cfg("meat_can")
    tuna_fish_can = make_object_cfg("tuna_fish_can")
    gelatin_box = make_object_cfg("gelatin_box")
    pudding_box = make_object_cfg("pudding_box")

    box_wall_back = _wall_cfg(
        "box_wall_back", _BACK_FRONT_SIZE,
        (BOX_CENTER_X + BOX_HALF_DEPTH, BOX_CENTER_Y, _BOX_BOTTOM_Z + BOX_HEIGHT / 2),
    )
    box_wall_front = _wall_cfg(
        "box_wall_front", _BACK_FRONT_SIZE,
        (BOX_CENTER_X - BOX_HALF_DEPTH, BOX_CENTER_Y, _BOX_BOTTOM_Z + BOX_HEIGHT / 2),
    )
    box_wall_top = _wall_cfg(
        "box_wall_top", _TOP_SIZE,
        (BOX_CENTER_X, BOX_CENTER_Y, _BOX_TOP_Z + WALL_THICKNESS / 2),
    )
    box_wall_pos_y = _wall_cfg(
        "box_wall_pos_y", _SIDE_SIZE,
        (BOX_CENTER_X, BOX_CENTER_Y + BOX_HALF_WIDTH, _BOX_BOTTOM_Z + BOX_HEIGHT / 2),
    )
    box_wall_neg_y = _wall_cfg(
        "box_wall_neg_y", _SIDE_SIZE,
        (BOX_CENTER_X, BOX_CENTER_Y - BOX_HALF_WIDTH, _BOX_BOTTOM_Z + BOX_HEIGHT / 2),
    )


@configclass
class PeekingBoxTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@configclass
class PeekingBoxEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    sample_trial = EventTerm(func=_sample_trial, mode="reset", params={})


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class PeekingBoxTask(TaskBase):
    """TAVIS-HANDS Task 1: peeking box.

    A box with one side opening (left or right, randomized) sits on the table.
    A target YCB object is inside.  Head sees only the closed outside; wrist
    cameras must determine which side is open and the appropriate hand reaches
    in to grasp and lift the object out.

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
        return PeekingBoxSceneCfg()

    def get_termination_cfg(self):
        return PeekingBoxTerminationsCfg()

    def get_events_cfg(self):
        cfg = PeekingBoxEventCfg()
        cfg.sample_trial.params["task"] = self
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str = None):
        return None

    def get_metrics(self) -> list:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(2.0, 0.0, 2.0), lookat=(0.0, 0.0, 1.0))
