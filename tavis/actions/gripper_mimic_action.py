"""Custom action term that maps a single 1D gripper action to multiple finger
joints using gearing + offset, bypassing PhysX mimic joints entirely.

Each joint target is computed as::

    joint_target = offset + gearing * driver_target

where *driver_target* comes from the standard affine transform of the raw
action (``scale * action + action_offset``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)


class GripperMimicAction(ActionTerm):
    """Maps a single 1D gripper action to driver + mimic joint position targets."""

    cfg: GripperMimicActionCfg
    _asset: Articulation

    def __init__(self, cfg: GripperMimicActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # Resolve all joint indices: driver first, then mimics
        all_joint_names = [cfg.driver_joint] + list(cfg.mimic_joints.keys())
        self._joint_ids, self._joint_names = self._asset.find_joints(
            all_joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)

        logger.info(
            f"GripperMimicAction: resolved joints {self._joint_names} [{self._joint_ids}]"
        )

        # Build gearing and offset tensors (driver has gearing=1, offset=0)
        gearings = [1.0]
        offsets = [0.0]
        for name in list(cfg.mimic_joints.keys()):
            g, o = cfg.mimic_joints[name]
            gearings.append(g)
            offsets.append(o)

        self._gearings = torch.tensor(gearings, device=self.device, dtype=torch.float32)
        self._offsets = torch.tensor(offsets, device=self.device, dtype=torch.float32)

        # Action transform
        self._scale = cfg.scale
        self._action_offset = cfg.action_offset

        # Buffers
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # Compute driver target from raw action
        driver_target = self._scale * actions + self._action_offset  # (N, 1)
        # Broadcast to all joints and apply gearing + offset
        self._processed_actions = self._offsets + self._gearings * driver_target  # (N, num_joints)

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class GripperMimicActionCfg(ActionTermCfg):
    """Configuration for :class:`GripperMimicAction`."""

    class_type: type = GripperMimicAction

    asset_name: str = "robot"
    driver_joint: str = MISSING
    mimic_joints: dict[str, tuple[float, float]] = MISSING  # {name: (gearing, offset)}
    scale: float = 1.0
    action_offset: float = 0.0
