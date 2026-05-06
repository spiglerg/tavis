"""
Shared observation helpers.

State observations are defined in each embodiment's observation_config.
Camera observations are handled by Arena's make_camera_observation_cfg().
"""

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp


def black_image(env, sensor_name: str, data_type="rgb", normalize=False):
    """Always returns a zero (black) image — same shape as the real camera output.

    Drop-in replacement for ``image_or_zeros`` when a camera should be
    disabled at the observation level (e.g. head-camera blackout in
    BlockedClutterPickCubeTask).
    """
    n = env.num_envs
    w, h = (640, 480)
    c = 1 if data_type in ("depth", "segmentation", "distance_to_camera") else 3
    dtype = torch.float32 if normalize or data_type != "rgb" else torch.uint8
    return torch.zeros((n, h, w, c), dtype=dtype, device=env.device)


def image_or_zeros(env, sensor_name: str, data_type="rgb", normalize=False):
    """
    Returns mdp.image(...) if the sensor exists, otherwise a zero image.
    """
    try:
        return mdp.image(
            env,
            sensor_cfg=SceneEntityCfg(sensor_name),
            data_type=data_type,
            normalize=normalize,
        )
    except Exception:
        n = env.num_envs
        w, h = (640, 480)
        c = 1 if data_type in ("depth", "segmentation", "distance_to_camera") else 3
        dtype = torch.float32 if normalize or data_type != "rgb" else torch.uint8
        return torch.zeros((n, h, w, c), dtype=dtype, device=env.device)


def get_eef_pos_canonical(
    env, link_name: str, canonical_frame_offset: tuple = (0.0, 0.0, 0.0)
) -> torch.Tensor:
    """Return end-effector position in the canonical frame.

    The **canonical frame** is a robot-independent reference frame centered
    approximately at hip/waist height.  It is obtained by expressing the
    end-effector position in the robot root frame and then subtracting a
    per-embodiment ``canonical_frame_offset``.

    This makes EEF positions comparable across embodiments whose root
    links sit at very different locations (e.g. GR1T2 root is at the
    hips, Reachy2 root is at the wheel base on the floor).  See
    :class:`~tavis.wrappers.CanonicalFrameWrapper` for the
    corresponding action-side conversion.

    Args:
        env: The IsaacLab environment.
        link_name: Body/link name of the end-effector.
        canonical_frame_offset: ``(x, y, z)`` offset from the robot root
            to the canonical origin, expressed in the root frame.  For
            GR1T2 this is ``(0, 0, 0)``; for Reachy2 ``(-0.01, 0, 0.93)``.

    Returns:
        Tensor of shape ``(num_envs, 3)`` — EEF position in canonical frame.
    """
    idx = env.scene["robot"].data.body_names.index(link_name)
    pos_w = env.scene["robot"].data.body_pos_w[:, idx]
    root_pos_w = env.scene["robot"].data.root_pos_w
    offset = torch.tensor(canonical_frame_offset, dtype=pos_w.dtype, device=pos_w.device)
    return (pos_w - root_pos_w) - offset
