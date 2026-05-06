"""
Pollen Robotics Reachy2 Embodiment for IsaacLab-Arena.

Custom parallel grippers (1D open/close per hand),
3-DOF neck (roll+pitch+yaw), 7-DOF arms.
Mobile base and tripod locked via high stiffness (not in action space).

Unified 19D action space:
  [left_arm_ik(7), right_arm_ik(7), head_roll(1), head_pitch_yaw(2),
   left_gripper(1), right_gripper(1)]

NOTE: The reachy2 USD has many dummy/fixed intermediate links in the
shoulder, elbow, and wrist ball-joint mechanisms.  Only the 7 actuated
revolute joints per arm are used for IK.

NOTE: If the USD still contains orphaned bar-mimic joints whose target
links have been removed, PhysX may fail to load.  Remove the joints
left_bar_joint_mimic, right_bar_joint_mimic, left_bar_prism_joint_mimic,
right_bar_prism_joint_mimic (and left_bar_base / right_bar_base prims)
from the USD if you get articulation load errors.
"""

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg, JointPositionToLimitsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place import mdp

from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.utils.isaaclab_utils.resets import reset_all_articulation_joints

from tavis.controllers import NullSpaceIKControllerCfg, NullSpaceIKActionCfg
from tavis.mdp.observations import image_or_zeros, get_eef_pos_canonical
from tavis.actions import GripperMimicActionCfg


# Canonical frame offset for Reachy2.
# Reachy2's root link is at the wheel base (floor level).  The canonical
# frame is centered at approximately hip/waist height to match GR1T2.
#   x = -0.01: arms sit ~1 cm behind the base center
#   z =  0.93: matches GR1T2's root height above ground
_CANONICAL_FRAME_OFFSET = (-0.01, 0.0, 0.93)


# -- Reachy2 arm joint names (explicit, in kinematic-chain order) ----------
_L_ARM_JOINTS = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_elbow_yaw",
    "l_elbow_pitch",
    "l_wrist_roll",
    "l_wrist_pitch",
    "l_wrist_yaw",
]
_R_ARM_JOINTS = [
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_elbow_yaw",
    "r_elbow_pitch",
    "r_wrist_roll",
    "r_wrist_pitch",
    "r_wrist_yaw",
]


# ---------------------------------------------------------------------
# Action config  (19D unified)
# ---------------------------------------------------------------------
@configclass
class Reachy2ActionsCfg:
    """19D unified action space for Reachy2."""

    # Left arm IK (7D: pos xyz + quat wxyz)
    left_arm_ik = NullSpaceIKActionCfg(
        asset_name="robot",
        joint_names=_L_ARM_JOINTS,
        body_name="l_palm_ik_target",
        controller=NullSpaceIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            use_nullspace_control=True,
            nullspace_stiffness=0.1, #1.0, # TUNE
            nullspace_joint_targets={
                "l_shoulder_pitch": -0.3,
                "l_shoulder_roll": 0.0,
                "l_elbow_yaw": 0.0,
                "l_elbow_pitch": -1.7,
                "l_wrist_roll": 0.0,
                "l_wrist_pitch": 0.0,
                "l_wrist_yaw": 0.0,
            },
        ),
    )

    # Right arm IK (7D: pos xyz + quat wxyz)
    right_arm_ik = NullSpaceIKActionCfg(
        asset_name="robot",
        joint_names=_R_ARM_JOINTS,
        body_name="r_palm_ik_target",
        controller=NullSpaceIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            use_nullspace_control=True,
            nullspace_stiffness=0.1, #1.0, # TUNE
            nullspace_joint_targets={
                "r_shoulder_pitch": -0.3,
                "r_shoulder_roll": 0.0,
                "r_elbow_yaw": 0.0,
                "r_elbow_pitch": -1.7,
                "r_wrist_roll": 0.0,
                "r_wrist_pitch": 0.0,
                "r_wrist_yaw": 0.0,
            },
        ),
    )

    # Head roll (1D)
    head_roll = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["neck_roll"],
        scale=1.0,
        use_default_offset=True,
    )

    # Head pitch + yaw (2D)
    head_pitch_yaw = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "neck_pitch",
            "neck_yaw",
        ],
        scale=1.0,
        use_default_offset=True,
    )

    # Left gripper (1D, normalized [-1, 1]: +1=close, -1=open)
    # Maps a single action to driver + all mimic finger joints via gearing/offset.
    # Driver axis is inverted (negative = close). scale/action_offset produce
    # the driver target; mimic joints follow via their own gearing/offset.
    left_gripper = GripperMimicActionCfg(
        asset_name="robot",
        driver_joint="l_hand_finger",
        mimic_joints={
            "l_hand_finger_proximal": (-0.4689, 0.554),
            "l_hand_finger_distal": (0.4689, -0.554),
            "l_hand_finger_proximal_mimic": (-0.4689, 0.554),
            "l_hand_finger_distal_mimic": (0.4689, -0.554),
        },
        scale=-1.876,
        action_offset=0.916,
    )

    # Right gripper (1D, normalized [-1, 1]: +1=close, -1=open)
    right_gripper = GripperMimicActionCfg(
        asset_name="robot",
        driver_joint="r_hand_finger",
        mimic_joints={
            "r_hand_finger_proximal": (-0.4689, 0.554),
            "r_hand_finger_distal": (0.4689, -0.554),
            "r_hand_finger_proximal_mimic": (-0.4689, 0.554),
            "r_hand_finger_distal_mimic": (0.4689, -0.554),
        },
        scale=-1.876,
        action_offset=0.916,
    )


# ---------------------------------------------------------------------
# Scene config (robot only -- task adds table/objects)
# ---------------------------------------------------------------------

# TUNE: tripod_joint controls torso height; initial pos=(0,0,z)
# determines base placement.  Adjust both to match table height.
_INIT_Z = 0.0               # base_link height above ground


@configclass
class Reachy2SceneCfg:
    """Reachy2 robot scene configuration."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{Path(__file__).parent.absolute()}/../assets/robots/reachy2/reachy2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                fix_root_link=True,  # Anchor base to world — prevents mobile base drift in multi-env
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, _INIT_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # Left arm (rest pose)
                "l_shoulder_pitch": -0.3,
                "l_shoulder_roll": 0.0,
                "l_elbow_yaw": 0.0,
                "l_elbow_pitch": -1.7,
                "l_wrist_roll": 0.0,
                "l_wrist_pitch": 0.0,
                "l_wrist_yaw": 0.0,
                # Right arm (rest pose)
                "r_shoulder_pitch": -0.3,
                "r_shoulder_roll": 0.0,
                "r_elbow_yaw": 0.0,
                "r_elbow_pitch": -1.7,
                "r_wrist_roll": 0.0,
                "r_wrist_pitch": 0.0,
                "r_wrist_yaw": 0.0,
                # Head
                "neck_roll": 0.0,
                "neck_pitch": 0.0,
                "neck_yaw": 0.0,
                # Grippers open – initial positions must match the PhysX
                # mimic joint formula (target = offset + gearing * driver)
                # at driver=0 so the mimic solver starts in equilibrium.
                "l_hand_finger": 2.4, # 0
                "r_hand_finger": 2.4, # 0
                ".*hand_finger_proximal": -0.554, # 0.554    # = -0.4689 * 2.79 + 0.554
                ".*hand_finger_distal": 0.554,  #- 0.554      # =  0.4689 * 2.79 - 0.554
                ".*hand_finger_proximal_mimic": -0.554, # 0.554
                ".*hand_finger_distal_mimic": 0.554, # - 0.554
                # Antennas
                "antenna_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Antennas (locked)
            "antenna_lock": ImplicitActuatorCfg(
                joint_names_expr=["antenna_.*"],
                effort_limit_sim=100.0,
                velocity_limit_sim=1.0,
                stiffness=10_000_000.0,
                damping=200.0,
            ),
            # Head / neck
            "neck": ImplicitActuatorCfg(
                joint_names_expr=["neck_.*"],
                stiffness=50.0,
                damping=5.0,
            ),
            # Arms (both left and right)
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    "l_shoulder_.*",
                    "l_elbow_.*",
                    "l_wrist_.*",
                    "r_shoulder_.*",
                    "r_elbow_.*",
                    "r_wrist_.*",
                ],
                effort_limit_sim=30.0,
                velocity_limit_sim=5.0,
                stiffness=60.0,
                damping=10.0,
            ),
            # All gripper finger joints (driver + mimic) — controlled via
            # GripperMimicAction which computes per-joint targets from a
            # single 1D action using gearing/offset.
            "gripper_fingers": ImplicitActuatorCfg(
                joint_names_expr=[
                    "l_hand_finger",
                    "r_hand_finger",
                    "l_hand_finger_proximal",
                    "l_hand_finger_distal",
                    "l_hand_finger_proximal_mimic",
                    "l_hand_finger_distal_mimic",
                    "r_hand_finger_proximal",
                    "r_hand_finger_distal",
                    "r_hand_finger_proximal_mimic",
                    "r_hand_finger_distal_mimic",
                ],
                stiffness=70.0,
                damping=12.0,
                effort_limit_sim=20.0,
                velocity_limit_sim=5.0,
            ),
        },
    )


# ---------------------------------------------------------------------
# Camera config
# ---------------------------------------------------------------------
@configclass
class Reachy2CameraCfg:
    """Camera configuration for Reachy2."""

    # Head camera -- mounted on `head` link (near tof_camera position)
    head_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head/head_camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0,  # ~70 deg FOV
            horizontal_aperture=20.955,
            clipping_range=(0.15, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.02, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        height=480,
        width=640,
    )

    # Left wrist camera -- mounted on palm link (at gripper base)
    left_wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/l_hand_palm_link/left_wrist_camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=21.0,  # ~53 deg FOV
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.01),
            rot=(0, 0.4617486, 0, 0.8870108),
            convention="world",
        ),
        data_types=["rgb"],
        height=480,
        width=640,
    )

    # Right wrist camera -- mounted on palm link (at gripper base)
    right_wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/r_hand_palm_link/right_wrist_camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=21.0,  # ~53 deg FOV
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.01),
            rot=(0, 0.4617486, 0, 0.8870108),
            convention="world",
        ),
        data_types=["rgb"],
        height=480,
        width=640,
    )


# ---------------------------------------------------------------------
# GPU PhysX Fabric writeback workaround (reachy2-specific).
#
# Problem: on GPU physics, PhysX does not write back the world transform of
# `head`, `l_hand_palm_link`, `r_hand_palm_link` (and their revolute-child
# parents `neck_link` / `l_wrist_link` / `r_wrist_link`) to Fabric. The 3-deep
# fixed-joint chain downstream of each revolute child triggers a writeback
# optimization in PhysX that leaves these bodies stale in Fabric. The
# rendering pipeline (which reads via Fabric) then sees stale camera parent
# transforms and renders from the USD-authored rest pose.
#
# Meanwhile `robot.data.body_pos_w` / `body_quat_w` are always live because
# they read from `_root_physx_view` (the solver tensor), not from Fabric.
#
# Workaround: for each step where a reachy2 camera is sampled, read
# body_pos_w/body_quat_w of the camera's original attachment body, compose
# with the original CameraCfg.OffsetCfg values, and call
# `cam.set_world_poses(...)` which syncs Fabric AND USD (because the Camera's
# internal XformPrimView has `sync_usd_on_fabric_write=True`). Then force a
# sim render pass so the renderer picks up the new pose, and invalidate the
# sensor buffers so the next .data access re-reads fresh annotator output.
# Result: zero-lag correct camera images on both CPU and GPU physics.
#
# This runs as a side-effect inside a custom ObsTerm function wrapping
# image_or_zeros. It is gated on GPU device -- CPU physics has correct
# Fabric writeback and doesn't need the override.
#
# Set _REACHY2_CAM_WORKAROUND_ENABLED = False to disable and fall back to
# stock image_or_zeros (cameras will break on GPU).
# ---------------------------------------------------------------------

_REACHY2_CAM_WORKAROUND_ENABLED = True

# Camera parent bodies and their OffsetCfg values (must match CameraCfg above
# and the teleop training datasets).
_REACHY2_CAM_BINDINGS = [
    ("head_camera",        "head",              (0.02, 0.0, 0.05),  (1.0, 0.0, 0.0, 0.0)),
    ("left_wrist_camera",  "l_hand_palm_link",  (0.10, 0.0, 0.01),  (0.0, 0.4617486, 0.0, 0.8870108)),
    ("right_wrist_camera", "r_hand_palm_link",  (0.10, 0.0, 0.01),  (0.0, 0.4617486, 0.0, 0.8870108)),
]

_REACHY2_CAM_DRIVER_CACHE: dict = {}  # id(env) -> cache dict


def _reachy2_cam_driver_init(env):
    """Build per-env cache: body name → body index for each unique camera parent.

    Uses pure pxr USD API for the actual sync (no XformPrimView, no warp, no
    Fabric dependency). This avoids the ``Invalid device identifier`` crash that
    XformPrimView triggers during early env init.
    """
    scene = env.scene
    robot = scene["robot"]
    name_to_idx = {n: i for i, n in enumerate(robot.data.body_names)}

    bodies = {}  # body_name -> body_idx (unique)
    for cam_name, body_name, _, _ in _REACHY2_CAM_BINDINGS:
        if cam_name not in scene.sensors:
            continue
        if body_name not in name_to_idx:
            print(f"[reachy2 cam-driver] body {body_name!r} not in articulation; skipping {cam_name}")
            continue
        bodies[body_name] = name_to_idx[body_name]

    if bodies:
        print(f"[reachy2 cam-driver] init OK: syncing {list(bodies.keys())} via pxr USD API")

    return {"robot": robot, "bodies": bodies, "num_envs": env.num_envs, "last_step": -1}


def _reachy2_cam_driver_update(env, cache):
    """Write each camera-parent body's world pose directly to USD xformOps
    using the pxr API. No XformPrimView, no warp, no Fabric — just direct
    USD attribute writes.

    For each body, we:
      1) Read body_pos_w / body_quat_w from the solver (always live)
      2) Get the body prim's USD parent (``Robot``) world transform via
         UsdGeom.XformCache (anchored by fix_root_link, so not stale)
      3) Compute ``local = world * inverse(parent)``
      4) Write xformOp:translate and xformOp:orient on the body prim

    The camera prims (children of these bodies in the USD hierarchy) keep their
    original OffsetCfg-derived local xformOps. USD scene-graph composition
    gives the correct camera world transform automatically.

    Then force a sim render and mark camera sensors outdated so the next
    ``.data`` access re-reads fresh annotator output.
    """
    from pxr import Gf, UsdGeom, Usd

    robot = cache["robot"]
    stage = env.sim.stage
    num_envs = cache["num_envs"]
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    for body_name, body_idx in cache["bodies"].items():
        body_pos_all = robot.data.body_pos_w[:, body_idx].detach().cpu().numpy()
        body_quat_all = robot.data.body_quat_w[:, body_idx].detach().cpu().numpy()

        for env_i in range(num_envs):
            prim_path = f"/World/envs/env_{env_i}/Robot/{body_name}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue

            pos = body_pos_all[env_i]
            qw, qx, qy, qz = body_quat_all[env_i]

            world_tf = Gf.Matrix4d()
            world_tf.SetRotateOnly(Gf.Quatd(float(qw), float(qx), float(qy), float(qz)))
            world_tf.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

            parent_world_tf = xform_cache.GetLocalToWorldTransform(prim.GetParent())
            local_tf = world_tf * parent_world_tf.GetInverse()

            translate_attr = prim.GetAttribute("xformOp:translate")
            orient_attr = prim.GetAttribute("xformOp:orient")
            if translate_attr.IsValid():
                translate_attr.Set(local_tf.ExtractTranslation())
            if orient_attr.IsValid():
                orient_attr.Set(local_tf.ExtractRotationQuat())

    env.sim.render()

    for cam_name, _, _, _ in _REACHY2_CAM_BINDINGS:
        if cam_name in env.scene.sensors:
            env.scene.sensors[cam_name]._is_outdated[:] = True


def _reachy2_image_with_pose_sync(env, sensor_name: str, data_type: str = "rgb", normalize: bool = False):
    """ObsTerm wrapper: applies the reachy2 GPU Fabric workaround (once per obs
    compute step) then returns the image via the standard image_or_zeros path.

    The first call in a given env.step() triggers the pose update + re-render;
    subsequent calls within the same step just return the already-fresh cached data.
    """
    if _REACHY2_CAM_WORKAROUND_ENABLED and "cuda" in str(getattr(env, "device", "")):
        env_id = id(env)
        cache = _REACHY2_CAM_DRIVER_CACHE.get(env_id)
        try:
            if cache is None:
                cache = _reachy2_cam_driver_init(env)
                _REACHY2_CAM_DRIVER_CACHE[env_id] = cache
            current_step = int(getattr(env, "common_step_counter", 0))
            if cache["last_step"] != current_step:
                _reachy2_cam_driver_update(env, cache)
                cache["last_step"] = current_step
        except Exception:
            # Init can fail during observation_manager._prepare_terms (shape
            # probe before sim is fully ready). Retry on next call.
            _REACHY2_CAM_DRIVER_CACHE.pop(env_id, None)
    return image_or_zeros(env, sensor_name, data_type, normalize)


# ---------------------------------------------------------------------
# Observation config
# ---------------------------------------------------------------------
@configclass
class Reachy2ObservationsCfg:
    """Observation specifications for Reachy2."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(
            func=get_eef_pos_canonical,
            params={"link_name": "l_palm_ik_target", "canonical_frame_offset": _CANONICAL_FRAME_OFFSET},
        )
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "l_palm_ik_target"})
        right_eef_pos = ObsTerm(
            func=get_eef_pos_canonical,
            params={"link_name": "r_palm_ik_target", "canonical_frame_offset": _CANONICAL_FRAME_OFFSET},
        )
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "r_palm_ik_target"})

        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["neck_pitch", "neck_roll", "neck_yaw"]},
        )

        # Camera observations. The robot-attached cameras (head, left_wrist,
        # right_wrist) use _reachy2_image_with_pose_sync which applies the GPU
        # Fabric writeback workaround -- see REACHY2_GPU_FABRIC_BUG.md.
        # The fixed_camera is scene-anchored and uses stock image_or_zeros.
        head_camera_rgb = ObsTerm(
            func=_reachy2_image_with_pose_sync,
            params={"sensor_name": "head_camera", "data_type": "rgb", "normalize": False},
        )
        left_wrist_camera_rgb = ObsTerm(
            func=_reachy2_image_with_pose_sync,
            params={"sensor_name": "left_wrist_camera", "data_type": "rgb", "normalize": False},
        )
        right_wrist_camera_rgb = ObsTerm(
            func=_reachy2_image_with_pose_sync,
            params={"sensor_name": "right_wrist_camera", "data_type": "rgb", "normalize": False},
        )
        fixed_camera_rgb = ObsTerm(
            func=image_or_zeros,
            params={"sensor_name": "fixed_camera", "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------
# Event config
# ---------------------------------------------------------------------
@configclass
class Reachy2EventCfg:
    """Event configuration for Reachy2."""

    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")


# ---------------------------------------------------------------------
# Embodiment class
# ---------------------------------------------------------------------
class Reachy2Embodiment(EmbodimentBase):
    """Embodiment for the Reachy2 robot with NullspaceIK control."""

    name = "reachy2"

    # Offset from robot root to the canonical frame origin.
    # See CanonicalFrameWrapper for details on the canonical coordinate system.
    canonical_frame_offset = _CANONICAL_FRAME_OFFSET

    # Indices of neck joints in the action vector (19D unified space: roll=14, pitch=15, yaw=16).
    # Used by train_policy.py to zero out neck actions in fixedcam mode.
    neck_action_indices = [14, 15, 16]

    # Indices of neck joints in observation.state (= robot_joint_pos, all joints in PhysX order).
    # Verified empirically: neck_roll=5, neck_pitch=8, neck_yaw=9.
    # Used by train_policy.py to zero out neck state in fixedcam mode.
    neck_state_indices = [5, 8, 9]

    # -- Teleop wrapper configuration (used by GenericTorsoExperimentWrapper) --
    teleop_config = {
        # Action indices in the 19D unified action space
        "neck_action_indices": {"pitch": 15, "yaw": 16},  # disabling roll, but it can be enabled by adding it here {"roll":14} — defaults to 0
        "left_arm_ik_action_index": 0,
        "right_arm_ik_action_index": 7,
        "left_gripper_action_index": 17,
        "right_gripper_action_index": 18,
        # EEF body names (for reading current EEF pose from sim)
        "left_arm_eef_body_name": "l_palm_ik_target",
        "right_arm_eef_body_name": "r_palm_ik_target",
        # Default EEF rest pose (position xyz in robot base frame)
        "left_arm_position_zero": [0.3, 0.2, 0.3],
        "right_arm_position_zero": [0.3, -0.2, 0.3],
        # Default EEF rest orientation (scipy xyzw quaternion)
        "left_arm_pose_zero_quat_xyzw": [-0.00045091, -0.7682481, -0.00857063, 0.6400948],
        "right_arm_pose_zero_quat_xyzw": [0.00045053, -0.76824796, 0.00857048, 0.6400948],
        # Height offset for Quest controller z -> sim z mapping
        "default_z_offset": 0.52,
        # First-person view camera observation key
        "fpv_camera_obs_key": "head_camera_rgb",
    }

    def __init__(self, enable_cameras: bool = True, initial_pose=None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = Reachy2SceneCfg()
        self.action_config = Reachy2ActionsCfg()
        self.observation_config = Reachy2ObservationsCfg()
        self.event_config = Reachy2EventCfg()
        self.camera_config = Reachy2CameraCfg() if enable_cameras else None
        self.mimic_env = None

    def get_observation_cfg(self):
        """Return observation config directly.

        We override the base class to skip make_camera_observation_cfg() --
        camera observations are already in PolicyCfg via image_or_zeros,
        which handles both enabled and disabled cameras gracefully.
        The camera_config is still used to add camera sensors to the scene
        (via get_scene_cfg -> combine_configclass_instances).
        """
        return self.observation_config

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        """Set simulation parameters matching our teleop pipeline."""
        env_cfg.sim.dt = 1.0 / 60.0
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.sim.render.enable_shadows = True
        return env_cfg
