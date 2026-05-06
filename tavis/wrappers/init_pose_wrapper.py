"""
Initial Pose Wrapper — moves the robot to a realistic start configuration on reset.

Problem: env.reset() puts the robot in its default joint configuration (from the USD/
embodiment init_state). But during teleop data collection, the robot is already tracking
the operator's hands by the time recording starts. This creates a distribution gap:
the policy never saw the default reset pose in training, so the first observation is
out-of-distribution and can cause catastrophic mode collapse.

Solution: after env.reset(), this wrapper sends a few warmup actions that drive the
robot's IK to the embodiment's default EEF rest pose (position_zero from teleop_config),
with optional random perturbation. The returned observation is from a state that closely
matches the start of recorded demonstrations.

This is analogous to "homing" in real robot systems — moving to a known start
configuration before executing a policy.

Usage:
    env = make_tavis_env(...)
    env = InitPoseWrapper(env, embodiment, warmup_steps=30)

For teleop, this wrapper is a no-op in practice: the operator's controllers immediately
override the warmup pose. For eval, it ensures the policy starts from a realistic state.
"""

import gymnasium as gym
import torch
import numpy as np


class InitPoseWrapper(gym.Wrapper):
    """Wrapper that moves the robot to a realistic start pose on each reset.

    Reads the embodiment's teleop_config to get the default EEF rest positions
    and orientations, then steps the env with those IK targets (plus optional
    random perturbation) for a fixed number of warmup steps.

    Args:
        env: The wrapped environment (after CanonicalFrameWrapper).
        embodiment: Embodiment instance with teleop_config dict containing
            left/right_arm_position_zero, left/right_arm_pose_zero_quat_xyzw.
        warmup_steps: Number of env steps to run with the warmup action (default: 30).
        position_noise_std: Std of Gaussian noise added to position targets (meters).
            Set to 0 to disable randomization. Default: 0.01.
        gripper_state: Gripper action value during warmup (-1=open, +1=close).
            Default: -1.0 (open).
    """

    def __init__(
        self,
        env,
        embodiment,
        warmup_steps: int = 30,
        position_noise_std: float = 0.01,
        head_noise_std: float | None = None,
        gripper_state: float = -1.0,
    ):
        super().__init__(env)
        self.warmup_steps = warmup_steps
        self.position_noise_std = position_noise_std
        # Head pitch/yaw noise (rad) scales with position noise by default to keep
        # the "OOD aggressiveness" balanced across arm and head axes. Pass an
        # explicit value to decouple. Legacy default was 0.05 rad at
        # position_noise_std=0.01 → ratio of 5.
        self.head_noise_std = (5.0 * position_noise_std) if head_noise_std is None else head_noise_std
        self.gripper_state = gripper_state

        # Read rest poses from embodiment's teleop config
        cfg = embodiment.teleop_config
        self._left_pos = np.array(cfg["left_arm_position_zero"], dtype=np.float32)
        self._right_pos = np.array(cfg["right_arm_position_zero"], dtype=np.float32)

        # Convert scipy xyzw quaternions to wxyz (IsaacLab convention)
        lq = cfg["left_arm_pose_zero_quat_xyzw"]
        rq = cfg["right_arm_pose_zero_quat_xyzw"]
        self._left_quat_wxyz = np.array([lq[3], lq[0], lq[1], lq[2]], dtype=np.float32)
        self._right_quat_wxyz = np.array([rq[3], rq[0], rq[1], rq[2]], dtype=np.float32)

        # Action indices from teleop config
        self._left_ik_idx = cfg["left_arm_ik_action_index"]
        self._right_ik_idx = cfg["right_arm_ik_action_index"]
        self._left_grip_idx = cfg["left_gripper_action_index"]
        self._right_grip_idx = cfg["right_gripper_action_index"]
        self._neck_idx_dict = cfg.get("neck_action_indices") or {}

    def _build_warmup_action(self, num_envs: int, device: torch.device) -> torch.Tensor:
        """Build the warmup action: IK targets at rest pose + per-env noise."""
        action = torch.zeros((num_envs, self.action_space.shape[-1]), device=device)

        li = self._left_ik_idx
        ri = self._right_ik_idx

        # Per-env random position noise
        left_pos = np.tile(self._left_pos, (num_envs, 1))   # (num_envs, 3)
        right_pos = np.tile(self._right_pos, (num_envs, 1))

        if self.position_noise_std > 0:
            left_pos += np.random.normal(0, self.position_noise_std, size=(num_envs, 3)).astype(np.float32)
            right_pos += np.random.normal(0, self.position_noise_std, size=(num_envs, 3)).astype(np.float32)

        action[:, li:li+3] = torch.tensor(left_pos, device=device)
        action[:, li+3:li+7] = torch.tensor(self._left_quat_wxyz, device=device)
        action[:, ri:ri+3] = torch.tensor(right_pos, device=device)
        action[:, ri+3:ri+7] = torch.tensor(self._right_quat_wxyz, device=device)

        # Grippers
        action[:, self._left_grip_idx] = self.gripper_state
        action[:, self._right_grip_idx] = self.gripper_state

        # Per-env head pitch/yaw noise (rad) — scales with position_noise_std by default.
        if self.head_noise_std > 0 and self._neck_idx_dict:
            for key in ("pitch", "yaw"):
                idx = self._neck_idx_dict.get(key)
                if idx is not None:
                    action[:, idx] = torch.tensor(
                        np.random.normal(0, self.head_noise_std, size=num_envs),
                        device=device, dtype=torch.float32)

        return action

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.warmup_steps > 0:
            device = obs['policy']['robot_joint_pos'].device
            num_envs = obs['policy']['robot_joint_pos'].shape[0]
            warmup_action = self._build_warmup_action(num_envs, device)

            for _ in range(self.warmup_steps):
                obs, _, _, _, _ = self.env.step(warmup_action)

        return obs, info
