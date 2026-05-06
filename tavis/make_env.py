"""
Environment factory using IsaacLab-Arena's ArenaEnvBuilder.

Composes an embodiment + task + scene into a running ManagerBasedRLEnv.
"""

import argparse

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene


def make_tavis_env(
    embodiment,
    task,
    scene=None,
    num_envs=1,
    enable_cameras=True,
    robot_ctrl_rate=20,
    episode_length_s=100,
):
    """
    Create an IsaacLab Arena environment for tavis.

    Args:
        embodiment: An EmbodimentBase instance (e.g. AgibotA2DEmbodiment).
        task: A TaskBase instance (e.g. LiftTask).
        scene: Optional Arena Scene with extra assets. Defaults to empty Scene().
        num_envs: Number of parallel environments.
        enable_cameras: Whether to enable cameras on the embodiment.
        robot_ctrl_rate: Control frequency in Hz.
        episode_length_s: Maximum episode length in seconds.

    Returns:
        env: A ManagerBasedRLEnv instance, reset and ready to use.
    """
    # Fetch robot and task USDs from Hugging Face on first use. No-op if
    # the assets are already on disk; runs once per machine otherwise.
    from .download_assets import check_assets, download_assets
    if not check_assets():
        download_assets()

    if scene is None:
        scene = Scene()

    # Override task episode length
    task.episode_length_s = episode_length_s

    # Ensure embodiment camera setting matches
    embodiment.enable_cameras = enable_cameras
    if enable_cameras and embodiment.camera_config is None:
        print("Warning: enable_cameras=True but embodiment has no camera_config. Adding default front-facing RGB camera.")

    # Build the Arena environment descriptor
    arena_env = IsaacLabArenaEnvironment(
        name="tavis_env",
        embodiment=embodiment,
        scene=scene,
        task=task,
    )

    # Build a synthetic args namespace for the builder
    device = "cpu" if num_envs == 1 else "cuda:0"
    args = argparse.Namespace(
        device=device,
        num_envs=num_envs,
        disable_fabric=False,
        mimic=False,
    )

    builder = ArenaEnvBuilder(arena_env, args)

    # Compose and build the environment configuration
    name, cfg = builder.build_registered()

    # Set decimation based on control rate (sim runs at 60 Hz)
    sim_freq = 1.0 / cfg.sim.dt
    cfg.decimation = max(1, int(round(sim_freq / robot_ctrl_rate)))
    cfg.sim.render_interval = cfg.decimation

    # Force extra renders after reset so RTX cameras reflect the new scene state.
    # Default is 0, which leaves sensor data stale on the first frame after reset
    # (camera returns the previous episode's image). This corrupts policy obs history
    # at episode boundaries. See IsaacLab manager_based_rl_env.py:224 and the docs
    # on `num_rerenders_on_reset`.
    cfg.num_rerenders_on_reset = 4

    # Create the environment
    import gymnasium as gym
    env = gym.make(name, cfg=cfg).unwrapped

    env.reset()  # first reset syncs env.task via _sample_target event

    # Wrap with canonical frame remapping so that IK position targets
    # are expected in the canonical frame (hip-level, robot-independent)
    # and automatically converted to the robot's root frame before
    # reaching the IK controller.  See CanonicalFrameWrapper docstring.
    from tavis.wrappers import CanonicalFrameWrapper
    env = CanonicalFrameWrapper(env, embodiment)

    #from tavis.wrappers import InitPoseWrapper
    #env = InitPoseWrapper(env, embodiment, warmup_steps=15, position_noise_std=0.02, head_noise_std=0.05)

    return env
