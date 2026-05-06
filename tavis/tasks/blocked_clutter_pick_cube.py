"""
BlockedClutterPickCube — TAVIS-HANDS Task 3.

Direct paired ablation of TAVIS-HEAD's ClutterPickCube.  Same 4-of-5
distractor objects + red cube, same randomization ranges, same fixed prompt.
The ONLY change: the head camera observation is replaced with a black image
via ``modify_env_cfg``, so the policy (and the teleop operator) receive no
useful head-camera feed.  No physical wall in the scene — arms are fully
unobstructed.

With the head camera blacked out, the active head still moves but produces
no useful image; only the wrist cameras can locate the red cube.

Lets us write: "active gaze solves clutter_pick_cube at X% in TAVIS-HEAD;
under head-camera blackout (TAVIS-HANDS) it drops to Y%; wrist cams
recover Z% of the gap."

Implemented as a Python subclass of ClutterPickCubeTask: identical prompt,
distractors, scene, success function, and randomization ranges.  Only the
head_camera_rgb observation term is swapped to ``black_image``.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm

from tavis.mdp.observations import black_image

from .clutter_pick_cube import ClutterPickCubeTask


class BlockedClutterPickCubeTask(ClutterPickCubeTask):
    """TAVIS-HANDS Task 3: blocked clutter pick cube.

    Paired ablation of ClutterPickCubeTask: same prompt, same objects, same
    randomization ranges, same success criterion — but the head camera
    observation is replaced with a black image.  Only the wrist cameras
    can locate the red cube.

    Parameters
    ----------
    task_variant : "id" | "ood_spatial"
        Selects a preset from ``VARIANTS`` (inherited from ClutterPickCubeTask).
    variant_overrides : dict | None
        Patches individual keys of the selected preset.
    """

    def modify_env_cfg(self, env_cfg):
        """Replace head_camera_rgb with a black-image obs term."""
        env_cfg.observations.policy.head_camera_rgb = ObsTerm(
            func=black_image,
            params={"sensor_name": "head_camera", "data_type": "rgb", "normalize": False},
        )
        return env_cfg
