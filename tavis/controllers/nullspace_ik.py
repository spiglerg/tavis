from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, Literal

import omni.log

from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import apply_delta_pose, compute_pose_error


class NullSpaceIKController(DifferentialIKController):
    """Differential inverse kinematics controller with nullspace control.

    This controller extends the base differential IK controller by adding nullspace
    control, which allows specifying preferred joint configurations while still
    achieving the desired end-effector pose.

    The control law becomes:
    Δq = J^+ * Δx + (I - J^+ * J) * k_null * (q_target - q_current)

    where:
    - J^+ is the pseudo-inverse of the Jacobian
    - (I - J^+ * J) is the nullspace projection matrix
    - k_null is the nullspace stiffness gain
    - q_target is the desired joint configuration
    - q_current is the current joint configuration
    """

    def __init__(self, cfg: NullSpaceIKControllerCfg, num_envs: int, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            num_envs: The number of environments.
            device: The device to use for computations.
        """
        # Initialize parent class
        super().__init__(cfg, num_envs, device)

        # Store extended config
        self.cfg: NullSpaceIKControllerCfg = cfg

        # Nullspace control will be initialized when joint information is available
        self._nullspace_targets = None
        self._use_nullspace = cfg.use_nullspace_control
        self._nullspace_stiffness = cfg.nullspace_stiffness

    def set_nullspace_targets(self,
                            joint_names: list[str] | None = None,
                            joint_limits: torch.Tensor | None = None,
                            default_joint_pos: torch.Tensor | None = None):
        """Set the nullspace joint targets.

        Args:
            joint_names: List of joint names (needed if targets is a dict in config)
            joint_limits: Joint limits tensor of shape (num_envs, num_joints, 2) for "center" option
            default_joint_pos: Default joint positions tensor for "default" option
        """
        if not self._use_nullspace or self.cfg.nullspace_joint_targets is None:
            self._nullspace_targets = None
            return

        num_joints = len(joint_names)

        if isinstance(self.cfg.nullspace_joint_targets, dict):
            # Initialize a tensor full of NaNs, so that we can track which joints are not specified in the dictionary
            # and for which no nullspace control should be applied.
            self._nullspace_targets = torch.full((self.num_envs, num_joints), float('nan'), device=self._device)

            # Fill in the targets specified in the config dictionary
            for i, joint_name in enumerate(joint_names):
                if joint_name in self.cfg.nullspace_joint_targets:
                    self._nullspace_targets[:, i] = self.cfg.nullspace_joint_targets[joint_name]

        elif self.cfg.nullspace_joint_targets == "zero":
            self._nullspace_targets = torch.zeros(self.num_envs, num_joints, device=self._device)

        elif self.cfg.nullspace_joint_targets == "center":
            if joint_limits is None:
                raise ValueError("Joint limits must be provided for 'center' nullspace option")
            self._nullspace_targets = torch.mean(joint_limits, dim=-1)

        elif self.cfg.nullspace_joint_targets == "default":
            if default_joint_pos is None:
                raise ValueError("Default joint positions must be provided for 'default' nullspace option")
            self._nullspace_targets = default_joint_pos.clone()

        else:
            raise ValueError(f"Invalid nullspace_joint_targets: {self.cfg.nullspace_joint_targets}")

    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Computes target joint positions with nullspace control.

        Args:
            ee_pos: Current end-effector position in shape (N, 3).
            ee_quat: Current end-effector orientation in shape (N, 4).
            jacobian: Geometric jacobian matrix in shape (N, 6, num_joints).
            joint_pos: Current joint positions in shape (N, num_joints).

        Returns:
            Target joint positions in shape (N, num_joints).
        """
        # First compute the primary task delta using parent method
        if "position" in self.cfg.command_type:
            position_error = self.ee_pos_des - ee_pos
            jacobian_task = jacobian[:, 0:3]
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=position_error, jacobian=jacobian_task)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)

        # Add nullspace control if enabled
        if self._use_nullspace and self._nullspace_targets is not None:
            # Compute nullspace projection matrix: N = I - J^+ * J
            num_joints = joint_pos.shape[1]
            I = torch.eye(num_joints, device=self._device).unsqueeze(0).expand(self.num_envs, -1, -1)

            # Get the pseudo-inverse based on method
            if self.cfg.ik_method == "pinv":
                jacobian_pinv = torch.linalg.pinv(jacobian)
            elif self.cfg.ik_method == "svd":
                min_singular_value = self.cfg.ik_params.get("min_singular_value", 1e-5)
                U, S, Vh = torch.linalg.svd(jacobian)
                S_inv = 1.0 / S
                S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
                jacobian_pinv = (
                    torch.transpose(Vh, dim0=1, dim1=2)[:, :, :jacobian.shape[1]]
                    @ torch.diag_embed(S_inv)
                    @ torch.transpose(U, dim0=1, dim1=2)
                )
            elif self.cfg.ik_method == "dls":
                lambda_val = self.cfg.ik_params.get("lambda_val", 0.01)
                jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
                lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
                jacobian_pinv = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix)
            else:
                # For transpose method, we can't compute a proper nullspace
                jacobian_pinv = None

            if jacobian_pinv is not None:
                # Compute nullspace projection matrix
                nullspace_proj = I - torch.bmm(jacobian_pinv, jacobian)

                # Compute nullspace velocity towards targets
                joint_error = self._nullspace_targets - joint_pos

                # Zero out the error for any joint whose target is NaN
                joint_error = torch.where(
                    torch.isnan(self._nullspace_targets), torch.zeros_like(joint_error), joint_error
                )

                nullspace_vel = self._nullspace_stiffness * joint_error

                # Project nullspace velocity and add to delta
                delta_nullspace = torch.bmm(nullspace_proj, nullspace_vel.unsqueeze(-1)).squeeze(-1)

                delta_joint_pos = delta_joint_pos + delta_nullspace

        # Return desired joint positions
        return joint_pos + delta_joint_pos


@configclass
class NullSpaceIKControllerCfg(DifferentialIKControllerCfg):
    """Configuration for differential inverse kinematics controller with nullspace control."""

    class_type: type = NullSpaceIKController
    """The associated controller class."""

    # Nullspace control parameters
    use_nullspace_control: bool = True
    """Whether to use nullspace control. Defaults to True."""

    nullspace_stiffness: float = 5.0
    """Stiffness gain for nullspace control. Higher values make joints move faster to targets."""

    nullspace_joint_targets: Dict[str, float] | Literal["zero", "center", "default"] | None = None
    """Target joint positions for nullspace control.

    Can be:
    - Dict[str, float]: Dictionary mapping joint names to target positions
    - "zero": Use zero positions for all joints
    - "center": Use center of joint limits
    - "default": Use default joint positions
    - None: No nullspace control
    """


class NullSpaceIKAction(DifferentialInverseKinematicsAction):
    """Differential IK action term with nullspace control.

    This action term extends the base differential IK action by adding nullspace
    control capabilities. It allows specifying preferred joint configurations
    that the robot will try to achieve while still satisfying the primary
    end-effector pose task.
    """

    cfg: NullSpaceIKActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: NullSpaceIKActionCfg, env):
        """Initialize the extended differential IK action.

        Args:
            cfg: The configuration for the action term.
            env: The environment instance.
        """
        # Initialize parent class (but we'll override the controller)
        super().__init__(cfg, env)

        self._ik_controller = NullSpaceIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        # Initialize nullspace targets if configured
        self._initialize_nullspace_targets()

    def _initialize_nullspace_targets(self):
        """Initialize the nullspace joint targets based on configuration."""

        # Get joint limits and default positions if needed
        joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids]

        # Set nullspace targets in the controller
        self._ik_controller.set_nullspace_targets(
            joint_names=self._joint_names,
            joint_limits=joint_limits,
            default_joint_pos=default_joint_pos
        )

        # Log nullspace configuration
        if self.cfg.controller.use_nullspace_control:
            omni.log.info(
                f"Initialized nullspace control for {self.__class__.__name__} "
                f"with stiffness {self.cfg.controller.nullspace_stiffness}"
            )
            if isinstance(self.cfg.controller.nullspace_joint_targets, dict):
                omni.log.info(f"Using custom joint targets: {self.cfg.controller.nullspace_joint_targets}")
            else:
                omni.log.info(f"Using '{self.cfg.controller.nullspace_joint_targets}' nullspace targets")


@configclass
class NullSpaceIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for differential IK action with nullspace control."""
    class_type: type[ActionTerm] = NullSpaceIKAction

    controller: NullSpaceIKControllerCfg
