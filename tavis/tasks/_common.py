"""Shared scene primitives for TAVIS tasks.

Defines TABLE_HEIGHT, RIGID_PROPS, SimpleTableSceneCfg (packing table,
ground, lights, fixed camera), and the standard TAVIS-HEAD object set
(5 YCB objects with language prompts and scene configs).

Task scene configs subclass SimpleTableSceneCfg or TavisHeadSceneCfg
and may override any field (e.g. fixed_camera position/FOV).
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


TABLE_HEIGHT = 1.0  # packing_table surface above ground (m)

RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

_TASK_ASSETS = Path(__file__).parent.parent / "assets" / "tasks"     # local physics-baked USDs
_YCB_ASSETS = _TASK_ASSETS / "ycb"

_OBJECT_SCALE = (0.75, 0.75, 0.75)

# ---------------------------------------------------------------------------
# YCB object registry — physics USDs (local edits or Nucleus originals).
# Used by make_object_cfg() to build RigidObjectCfg for any task.
# Prompts are task-specific and defined per-task, not here.
# ---------------------------------------------------------------------------

ALL_OBJECTS: dict[str, str] = {
    "soup_can":        str(_YCB_ASSETS / "005_tomato_soup_can_physics.usd"),
    "meat_can":        str(_YCB_ASSETS / "010_potted_meat_can_physics.usd"),
    "tuna_fish_can":   str(_YCB_ASSETS / "007_tuna_fish_can_physics.usd"),
    "gelatin_box":     str(_YCB_ASSETS / "009_gelatin_box_physics.usd"),
    "pudding_box":     str(_YCB_ASSETS / "008_pudding_box_physics.usd"),
    "sugar_box":       str(_YCB_ASSETS / "004_sugar_box_physics.usd"),
    "cracker_box":     str(_YCB_ASSETS / "003_cracker_box_physics.usd"),
    "bleach_cleanser": str(_YCB_ASSETS / "021_bleach_cleanser_physics.usd"),
    "banana":          str(_YCB_ASSETS / "011_banana_physics.usd"),
    "mustard":         f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
}

# ---------------------------------------------------------------------------
# TAVIS-HEAD subset — used by TAVIS-HEAD tasks (clutter pick-lift, etc.)
# ---------------------------------------------------------------------------

TAVIS_HEAD_OBJECT_NAMES: list[str] = [
    "soup_can", "meat_can", "tuna_fish_can", "gelatin_box", "pudding_box",
]

# Distractors for ClutterPickCube — no overlap with TAVIS_HEAD_OBJECT_NAMES
CLUTTER_CUBE_DISTRACTOR_NAMES: list[str] = [
    "sugar_box", "cracker_box", "mustard", "banana",
]


@configclass
class BaseSceneCfg:
    """Ground, lights, and fixed workspace camera — no furniture.

    Subclass this (or SimpleTableSceneCfg) and add RigidObjectCfg fields
    for task-specific objects.  Override fixed_camera if a different
    viewpoint is needed.
    """

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(1., 1., 1.),
                                     intensity=800.0),
    )

    shadow_light = AssetBaseCfg(
        prim_path="/World/shadow_light",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[1, 1, 3],
            rot=[0.96985, -0.17101, 0.17101, -0.03015],
        ),
        spawn=sim_utils.DistantLightCfg(
            prim_type="DistantLight",
            intensity=2000.0,
            color=(1., 1., 1.),
            angle=0.5,
        ),
    )

    # Fixed third-person workspace camera.  Tasks may override position/FOV.
    fixed_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/fixed_camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.05,# (120deg) #7.33 (110deg), #8.8 (100deg), #12.0 (80deg),  # deg = 2*arctan(sensor_size/(2*focal_length))
            horizontal_aperture=20.955,
            clipping_range=(0.1, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # TO KEEP: camera above head, fixed, pointing down at an angle;  there is self-occlusion with arms, but the robot can see what's ahead. This to be used in
            # environments where frontal manipulation is important (e.g., kitchen environments).
            # The Pose below, instead, is for a camera placed in front of the robot, looking toward the robot!
            pos=(0.2, 0.0, 1.43),
            rot=(0.9238795, 0.0, 0.3826834, 0.0), # (0, -0.258819, 0, 0.9659258),   ## when head looks down, a small part of it is still visible in the fixed cam :/
            # The Pose below, instead, is for a camera placed in front of the robot, looking toward the robot!
            #pos=(1, 0.0, 1.43),
            #rot=(0, -0.258819, 0, 0.9659258),   ## when head looks down, a small part of it is still visible in the fixed cam :/
            convention="world",
        ),
        data_types=["rgb"],
        height=480,
        width=640,
    )


@configclass
class SimpleTableSceneCfg(BaseSceneCfg):
    """BaseSceneCfg + packing table.

    Subclass this and add RigidObjectCfg fields for task-specific objects.
    """

    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.60, -0.1, 0.0],
            rot=[0.7071, 0.0, 0.0, -0.7071],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path = str(_TASK_ASSETS / "packing_table.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


_DROP_Z = TABLE_HEIGHT + 0.1  # drop height for YCB objects (gravity settles them)


def make_object_cfg(
    name: str,
    scale: tuple[float, float, float] | None = None,
) -> RigidObjectCfg:
    """Build a RigidObjectCfg from the ALL_OBJECTS registry."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 0.0, _DROP_Z], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=ALL_OBJECTS[name],
            scale=scale if scale is not None else _OBJECT_SCALE,
            rigid_props=RIGID_PROPS,
            semantic_tags=[("class", name)],
        ),
    )


# ---------------------------------------------------------------------------
# Shared helpers for multi-task use (colour changes, object placement)
# ---------------------------------------------------------------------------

_HIDE_POS = (-2.0, 0.0, 0.0)  # behind the robot, out of camera view


def _quat_from_roll_yaw(roll: float, yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) from roll and yaw angles (pitch = 0)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cy, sr * cy, sr * sy, cr * sy)


def change_prim_color(env, prim_path: str, color: tuple[float, float, float]):
    """Change the diffuseColor of a UsdPreviewSurface shader at runtime.

    *prim_path* is the full USD path to the rigid-body prim, e.g.
    ``"/World/envs/env_0/cue_card"``.  For shape spawners (CuboidCfg,
    SphereCfg, …) the shader lives at
    ``{prim_path}/geometry/material/Shader``.
    """
    from pxr import Gf
    stage = env.sim.stage
    shader = stage.GetPrimAtPath(f"{prim_path}/geometry/material/Shader")
    shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*color))


def place_object(env, name: str, x: float, y: float, z: float, yaw: float, env_ids):
    """Write a new root state for a YCB object (roll = -pi/2 to stay upright)."""
    asset = env.scene[name]
    state = asset.data.default_root_state[env_ids].clone()
    q = _quat_from_roll_yaw(-math.pi / 2, yaw)
    state[:, 0] = x + env.scene.env_origins[env_ids, 0]
    state[:, 1] = y + env.scene.env_origins[env_ids, 1]
    state[:, 2] = z + env.scene.env_origins[env_ids, 2]
    state[:, 3], state[:, 4], state[:, 5], state[:, 6] = q
    state[:, 7:] = 0.0
    asset.write_root_state_to_sim(state, env_ids)


def hide_object(env, name: str, env_ids):
    """Teleport an object below the ground plane (out of view and reach)."""
    asset = env.scene[name]
    state = asset.data.default_root_state[env_ids].clone()
    state[:, 0] = _HIDE_POS[0] + env.scene.env_origins[env_ids, 0]
    state[:, 1] = _HIDE_POS[1] + env.scene.env_origins[env_ids, 1]
    state[:, 2] = _HIDE_POS[2] + env.scene.env_origins[env_ids, 2]
    state[:, 7:] = 0.0
    asset.write_root_state_to_sim(state, env_ids)