"""
Canonical-frame action wrapper for cross-embodiment compatibility.

Problem
-------
Different robot embodiments have their root link at different locations.
For example, GR1T2's root is at the hips (~0.93 m above ground), while
Reachy2's root is at the wheel base on the floor.  Because the IK
controller operates in the robot root frame, the same IK position target
produces very different physical arm poses depending on the robot.

Solution: the canonical frame
-----------------------------
The **canonical frame** is a robot-independent coordinate system centered
approximately at hip/waist height, with the same orientation as the robot
root frame (pure translation offset, no rotation).  Each embodiment
defines a ``canonical_frame_offset`` -- the 3D position of the canonical
origin in the robot's root frame:

- GR1T2:  ``(0, 0, 0)``     -- root is already at the hips
- Reachy2: ``(-0.01, 0, 0.93)`` -- root is at the floor, hips are ~0.93 m up

With this convention, a policy or teleoperator can output IK targets in
canonical coordinates and get physically equivalent arm poses on any
supported robot.

Where the conversion happens
----------------------------
- **Observations** (EEF positions): converted to canonical frame by
  :func:`~tavis.mdp.observations.get_eef_pos_canonical`, which
  is used in each embodiment's ``ObsTerm`` config.
- **Actions** (IK position targets): converted from canonical to root
  frame by this wrapper's :meth:`CanonicalFrameWrapper.step` method,
  which adds the offset back before the IK controller sees it.

Wrapping order
--------------
This wrapper should be applied *inside* any dataset-recording wrapper
(e.g. ``GenericTorsoExperimentWrapper``), so that the dataset stores
canonical-frame actions (portable across robots) while the IK controller
receives root-frame targets::

    ExperimentWrapper(CanonicalFrameWrapper(isaaclab_env))

``make_tavis_env`` applies this wrapper automatically.

Real-robot deployment
---------------------
When deploying outside IsaacLab (no IK action term), simply add the
embodiment's ``canonical_frame_offset`` to the policy's position outputs
before sending them to the motor controller.
"""

import gymnasium as gym
import torch


class CanonicalFrameWrapper(gym.Wrapper):
    """Thin wrapper that converts canonical-frame actions to root-frame.

    Only the IK position targets (3D xyz per arm) are shifted by the
    embodiment's ``canonical_frame_offset``.  Orientation targets, head
    joints, and gripper commands pass through unchanged (the canonical
    frame shares the root frame's orientation).

    Inherits from :class:`gymnasium.Wrapper` so that downstream wrappers
    (e.g. ``GenericTorsoExperimentWrapper``) see a proper ``gymnasium.Env``.

    Args:
        env: The IsaacLab environment to wrap.
        embodiment: An EmbodimentBase instance with a
            ``canonical_frame_offset`` attribute and a ``teleop_config``
            dict containing ``left_arm_ik_action_index`` and
            ``right_arm_ik_action_index``.
    """

    def __init__(self, env, embodiment):
        super().__init__(env)
        self._offset = torch.tensor(
            embodiment.canonical_frame_offset,
            dtype=torch.float32,
        )
        self._l_idx = embodiment.teleop_config["left_arm_ik_action_index"]
        self._r_idx = embodiment.teleop_config["right_arm_ik_action_index"]

    def step(self, action):
        """Convert canonical-frame IK targets to root-frame targets, then step."""
        offset = self._offset.to(dtype=action.dtype, device=action.device)
        root_action = action.clone()
        root_action[:, self._l_idx:self._l_idx + 3] += offset
        root_action[:, self._r_idx:self._r_idx + 3] += offset
        return self.env.step(root_action)

    def __getattr__(self, name):
        """Proxy IsaacLab-specific attributes (scene, observation_manager, etc.)."""
        return getattr(self.env, name)
