"""Custom reset event terms for TAVIS.

``reset_scene_to_default_safe`` mirrors
``isaaclab.envs.mdp.events.reset_scene_to_default`` but skips root-velocity
writes on kinematic bodies — both ``kinematic_enabled=True`` RigidObjects
(task signal lights, cue cards, scene fixtures) and fixed-base articulations
(Reachy 2 with ``fix_root_link=True``).

The skipped writes are no-ops at the physics level — PhysX ignores velocity
changes on kinematic bodies — but they emit C++ error logs from
omni.physx.plugin on every reset::

    PhysX error: PxRigidDynamic::setLinearVelocity: Body must be non-kinematic!
    PhysX error: PxRigidDynamic::setAngularVelocity: Body must be non-kinematic!

which clutter benchmark/eval output. This is the same operation upstream
``reset_scene_to_default`` performs, with a one-line guard added per asset
class.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv


def _rigid_object_is_kinematic(rigid_object) -> bool:
    """True iff the RigidObjectCfg set ``rigid_props.kinematic_enabled=True``."""
    spawn = getattr(rigid_object.cfg, "spawn", None)
    rigid_props = getattr(spawn, "rigid_props", None) if spawn is not None else None
    return bool(rigid_props is not None and getattr(rigid_props, "kinematic_enabled", False))


def reset_scene_to_default_safe(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reset_joint_targets: bool = False,
):
    """Drop-in replacement for ``mdp.reset_scene_to_default`` that skips
    root-velocity writes on kinematic rigid objects and fixed-base articulations.

    See module docstring for the rationale.
    """
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        if not _rigid_object_is_kinematic(rigid_object):
            rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    # articulations
    for articulation_asset in env.scene.articulations.values():
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        if not articulation_asset.is_fixed_base:
            articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        if reset_joint_targets:
            articulation_asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target(default_joint_vel, env_ids=env_ids)

    # deformable objects (unchanged from upstream — no kinematic concept)
    for deformable_object in env.scene.deformable_objects.values():
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
