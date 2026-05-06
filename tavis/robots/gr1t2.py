"""
Fourier GR1T2 Embodiment for IsaacLab-Arena.

Robotiq 2F-85 parallel grippers (1D open/close per hand),
3-DOF head (roll+pitch+yaw), 7-DOF arms.
Waist and legs locked via high stiffness (not in action space).

Unified 19D action space:
  [left_arm_ik(7), right_arm_ik(7), head_roll(1), head_pitch_yaw(2),
   left_gripper(1), right_gripper(1)]
"""

from pathlib import Path

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


"""
# Joints to include in the observation state vector (19D).
# Excludes locked leg/waist/ankle joints (near-zero variance in training data)
# and redundant gripper mimic joints (deterministic function of driver).
_OBSERVATION_JOINT_NAMES = [
    # Head (3)
    "head_roll_joint", "head_pitch_joint", "head_yaw_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    # Gripper drivers (2)
    "finger_joint", "finger_joint_0",
]
"""


# Canonical frame offset for GR1T2.
# GR1T2's root link is at the hips (~0.93 m above ground), which is
# already at the canonical hip-level origin.  No offset needed.
_CANONICAL_FRAME_OFFSET = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------
# Action config  (19D unified)
# ---------------------------------------------------------------------
@configclass
class GR1T2ActionsCfg:
    """19D unified action space for GR1T2 with Robotiq 2F-85 grippers."""

    # Left arm IK (7D: pos xyz + quat wxyz)
    left_arm_ik = NullSpaceIKActionCfg(
        asset_name="robot",
        joint_names="left_.*_joint",
        body_name="left_hand_pitch_link",
        controller=NullSpaceIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            use_nullspace_control=True,
            nullspace_stiffness=1.0,
            nullspace_joint_targets={
                "left_shoulder_pitch_joint": -0.335,
                "left_shoulder_roll_joint": -0.008,
                "left_shoulder_yaw_joint": 0.046,
                "left_elbow_pitch_joint": -1.778,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
            },
        ),
    )

    # Right arm IK (7D: pos xyz + quat wxyz)
    right_arm_ik = NullSpaceIKActionCfg(
        asset_name="robot",
        joint_names="right_.*_joint",
        body_name="right_hand_pitch_link",
        controller=NullSpaceIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            use_nullspace_control=True,
            nullspace_stiffness=1.0,
            nullspace_joint_targets={
                "right_shoulder_pitch_joint": -0.335,
                "right_shoulder_roll_joint": 0.008,
                "right_shoulder_yaw_joint": -0.047,
                "right_elbow_pitch_joint": -1.778,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
            },
        ),
    )

    # Head roll (1D)
    head_roll = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["head_roll_joint"],
        scale=1.0,
        use_default_offset=True,
    )

    # Head pitch + yaw (2D)
    head_pitch_yaw = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "head_pitch_joint",
            "head_yaw_joint",
        ],
        scale=1.0,
        use_default_offset=True,
    )

    # Left gripper (1D, normalized [-1, 1])
    left_gripper = JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
    )

    # Right gripper (1D, normalized [-1, 1])
    right_gripper = JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["finger_joint_0"],
    )


# ---------------------------------------------------------------------
# Scene config (robot only — task adds table/objects)
# ---------------------------------------------------------------------
@configclass
class GR1T2SceneCfg:
    """GR1T2 robot scene configuration."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{Path(__file__).parent.absolute()}/../assets/robots/GR1/GR1T2_robotiq85.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.93),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # Left arm (tuned rest pose)
                "left_shoulder_pitch_joint": -0.335,
                "left_shoulder_roll_joint": -0.008,
                "left_shoulder_yaw_joint": 0.046,
                "left_elbow_pitch_joint": -1.778,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # Right arm
                "right_shoulder_pitch_joint": -0.335,
                "right_shoulder_roll_joint": 0.008,
                "right_shoulder_yaw_joint": -0.047,
                "right_elbow_pitch_joint": -1.778,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # Head
                "head_roll_joint": 0.0,
                "head_pitch_joint": 0.0,
                "head_yaw_joint": 0.0,
                # Waist, legs: all zero
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                # Grippers (both left and right Robotiq joints)
                ".*finger.*": 0.0,
                "right_outer_knuckle_joint.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Waist (locked via very high stiffness)
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*"],
                effort_limit_sim=10000.0,
                velocity_limit_sim=2.61,
                stiffness=10_000_000.0,
                damping=200.0,
            ),
            # Legs (locked via very high stiffness)
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_.*",
                    ".*_knee_.*",
                    ".*_ankle_.*",
                ],
                effort_limit_sim=10000.0,
                velocity_limit_sim=2.61,
                stiffness=10_000_000.0,
                damping=200.0,
            ),
            # Head
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_.*_joint"],
                stiffness=50.0,
                damping=5.0,
            ),
            # Arms
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_.*_joint",
                    ".*_wrist_.*_joint",
                ],
                effort_limit_sim=30.0,
                velocity_limit_sim=5.0,
                stiffness=60.0,
                damping=10.0,
            ),
            # Gripper driver joints (finger_joint + right_outer_knuckle_joint, both grippers)
            "gripper_driver": ImplicitActuatorCfg(
                joint_names_expr=[
                    "finger_joint.*",
                    "right_outer_knuckle_joint.*",
                ],
                stiffness=25.0, #25.0,
                damping=1.0,#1.0,
                effort_limit_sim=4.0,
                velocity_limit_sim=5.0,
            ),
            # Gripper passive/mimic joints (inner finger + inner knuckle, both grippers)
            "gripper_mimic": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_inner_.*",
                ],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=4.0,
                velocity_limit_sim=5.0,
            ),
        },
    )


# ---------------------------------------------------------------------
# Camera config
# ---------------------------------------------------------------------
@configclass
class GR1T2CameraCfg:
    """Camera configuration for GR1T2."""

    head_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_yaw_link/head_camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0,  # ~70 deg FOV
            horizontal_aperture=20.955,
            clipping_range=(0.15, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        height=480,
        width=640,
    )

    left_wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_pitch_link/left_wrist_camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=21.0,  # ~53 deg FOV
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.05),
            rot=(0.4617486, 0.0, 0.8870108, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        height=480,
        width=640,
    )

    right_wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_pitch_link/right_wrist_camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=21.0,  # ~53 deg FOV
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.05),
            rot=(0.4617486, 0.0, 0.8870108, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        height=480,
        width=640,
    )


# ---------------------------------------------------------------------
# Observation config
# ---------------------------------------------------------------------
@configclass
class GR1T2ObservationsCfg:
    """Observation specifications for GR1T2."""

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
            params={"link_name": "left_hand_roll_link", "canonical_frame_offset": _CANONICAL_FRAME_OFFSET},
        )
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"})
        right_eef_pos = ObsTerm(
            func=get_eef_pos_canonical,
            params={"link_name": "right_hand_roll_link", "canonical_frame_offset": _CANONICAL_FRAME_OFFSET},
        )
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"})

        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]},
        )

        # Camera observations (use image_or_zeros so env works even with cameras disabled)
        head_camera_rgb = ObsTerm(
            func=image_or_zeros,
            params={"sensor_name": "head_camera", "data_type": "rgb", "normalize": False},
        )
        left_wrist_camera_rgb = ObsTerm(
            func=image_or_zeros,
            params={"sensor_name": "left_wrist_camera", "data_type": "rgb", "normalize": False},
        )
        right_wrist_camera_rgb = ObsTerm(
            func=image_or_zeros,
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
class GR1T2EventCfg:
    """Event configuration for GR1T2."""

    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")


# ---------------------------------------------------------------------
# Embodiment class
# ---------------------------------------------------------------------
class GR1T2Embodiment(EmbodimentBase):
    """Embodiment for the GR1T2 robot with Robotiq 2F-85 grippers and NullspaceIK control."""

    name = "gr1t2"

    # Offset from robot root to the canonical frame origin.
    # See CanonicalFrameWrapper for details on the canonical coordinate system.
    canonical_frame_offset = _CANONICAL_FRAME_OFFSET

    # Indices of neck joints in the action vector (19D unified space: roll=14, pitch=15, yaw=16).
    # Used by train_policy.py to zero out neck actions in fixedcam mode.
    neck_action_indices = [14, 15, 16]

    # Indices of neck joints in observation.state (= robot_joint_pos, all joints in PhysX order).
    # Verified empirically: head_roll_joint=11, head_pitch_joint=16, head_yaw_joint=21.
    # Used by train_policy.py to zero out neck state in fixedcam mode.
    neck_state_indices = [11, 16, 21]

    # -- Teleop wrapper configuration (used by GenericTorsoExperimentWrapper) --
    teleop_config = {
        # Action indices in the 19D unified action space
        "neck_action_indices": {"pitch": 15, "yaw": 16},  # no roll — defaults to 0
        "left_arm_ik_action_index": 0,
        "right_arm_ik_action_index": 7,
        "left_gripper_action_index": 17,
        "right_gripper_action_index": 18,
        # EEF body names (for reading current EEF pose from sim)
        "left_arm_eef_body_name": "left_hand_roll_link",
        "right_arm_eef_body_name": "right_hand_roll_link",
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
        self.scene_config = GR1T2SceneCfg()
        self.action_config = GR1T2ActionsCfg()
        self.observation_config = GR1T2ObservationsCfg()
        self.event_config = GR1T2EventCfg()
        self.camera_config = GR1T2CameraCfg() if enable_cameras else None
        self.mimic_env = None

    def get_observation_cfg(self):
        """Return observation config directly.

        We override the base class to skip make_camera_observation_cfg() —
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
