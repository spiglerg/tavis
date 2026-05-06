#!/usr/bin/env python3
"""
Single-task policy evaluation with optional visualization.

For systematic benchmark evaluation across suites, use eval_benchmark.py instead.
This script is primarily useful for debugging with visualization (headless=False).

Usage:
    python eval_policy.py --checkpoint <path> [--model diffusion|act|smolvla|pi0]
                          [--camera headcam|fixedcam] [--episodes 50] [--headless] [--lora]
                          [--robot gr1t2|reachy2] [--task clutter_pick_lift]

Examples:
    python eval_policy.py --checkpoint checkpoints/gr1t2_clutter_pick_lift_pi0_headcam --headless
    python eval_policy.py --checkpoint checkpoints/my_policy --model diffusion --camera fixedcam --episodes 20
    python eval_policy.py --checkpoint checkpoints/my_policy --lora --headless
"""

# Parse args before IsaacSim init (AppLauncher must come before most imports)
import argparse
parser = argparse.ArgumentParser(description="Evaluate a trained policy in Isaac Lab")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained policy checkpoint")
parser.add_argument("--model", type=str, default="",
                    help="Policy architecture: diffusion|act|smolvla|pi0 (default: auto-detect from checkpoint)")
parser.add_argument("--camera", type=str, default="", choices=["headcam", "fixedcam", ""],
                    help="Camera mode (default: auto-detect from checkpoint)")
parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes (default: 100)")
parser.add_argument("--robot", type=str, default="",
                    help="Robot embodiment: gr1t2|reachy2 (default: auto-detect from checkpoint)")
parser.add_argument("--task", type=str, default="clutter_pick_lift",
                    choices=["clutter_pick_lift", "clutter_pick_cube", "conditional_pick", "wait_then_act", "multi_shelf_scan"],
                    help="Task to evaluate on (default: clutter_pick_lift)")
parser.add_argument("--headless", action="store_true", help="Run without visualization")
parser.add_argument("--num-envs", type=int, default=1,
                    help="Number of parallel environments for headless mode (default: 1). Must divide --episodes.")
parser.add_argument("--lora", action="store_true", help="Load a LoRA-finetuned checkpoint (pi0 only)")
parser.add_argument("--robot-ctrl-rate", type=int, default=20,
                    help="Robot control rate in Hz (default: 20). Must match the effective rate "
                         "used to train the policy. Training defaults to downsample-factor=3 → 20Hz "
                         "effective from 60Hz teleop, matching LIBERO/RoboMimic. Use 60 for "
                         "policies trained without downsampling, or 30 for --downsample-factor=2.")
args = parser.parse_args()

# Initialize IsaacSim
from isaaclab.app import AppLauncher
num_envs = args.num_envs if args.headless else 1
app_launcher = AppLauncher(num_envs=num_envs, enable_cameras=True, headless=args.headless)
simulation_app = app_launcher.app

####

import time
import numpy as np

from tavis import make_tavis_env
from tavis.tasks import TASK_MAP
from tavis.robots import ROBOT_MAP
from tavis.eval.core import (
    load_policy, build_input_batch, run_eval_episodes, compute_summary,
)

robot_ctrl_rate = args.robot_ctrl_rate

# Load policy (auto-detects model/camera/robot from checkpoint name if not specified)
policy_bundle = load_policy(
    args.checkpoint, model=args.model, camera=args.camera, robot=args.robot, lora=args.lora,
)
#policy_bundle.policy.config.n_action_steps = 30

camera_mode = policy_bundle.camera_mode
print(policy_bundle.policy.config)

# Create embodiment and task
embodiment = ROBOT_MAP[policy_bundle.robot_name](enable_cameras=True)
task = TASK_MAP[args.task](episode_length_s=20)
isaaclab_env = make_tavis_env(embodiment=embodiment, task=task, num_envs=num_envs, robot_ctrl_rate=robot_ctrl_rate, episode_length_s=20)

try:
    if args.headless:
        # Headless mode: delegate to shared eval core
        episodes = run_eval_episodes(
            isaaclab_env, policy_bundle, camera_mode, embodiment, args.episodes,
            robot_ctrl_rate, num_envs=num_envs,
        )
        summary = compute_summary(episodes)

        print("=" * 50)
        print(f"Evaluation Results ({summary['n_episodes']} episodes):")
        print(f"  Success rate: {summary['success_rate']:.2%} ({summary['n_successful']}/{summary['n_episodes']})")
        if summary["avg_length_s_successful"] is not None:
            print(f"  Avg episode length (successful): "
                  f"{summary['avg_length_steps_successful']:.1f} steps "
                  f"({summary['avg_length_s_successful']:.1f} +/- {summary['std_length_s_successful']:.1f}s "
                  f"@ {robot_ctrl_rate} Hz)")
        else:
            print("  Avg episode length: N/A (no successful episodes)")
        print("=" * 50)

    else:
        # Visualization mode: inline loop with cv2 display
        # Supports num_envs >= 1. Use left/right arrows to cycle viewed env.
        import cv2
        import torch

        view_idx = 0  # which env to display
        successes = 0
        total_episodes = 0
        successful_episode_lengths = []

        if num_envs > 1 and args.episodes % num_envs != 0:
            print(f"Warning: --episodes ({args.episodes}) not divisible by --num-envs ({num_envs}), "
                  f"rounding down to {args.episodes - args.episodes % num_envs}")
            args.episodes = args.episodes - args.episodes % num_envs
        num_rounds = max(1, args.episodes // num_envs)

        for round_idx in range(num_rounds):
            print(f"\nRound {round_idx + 1}/{num_rounds} "
                  f"({num_envs} env{'s' if num_envs > 1 else ''})")
            obs, _ = isaaclab_env.reset()
            policy_bundle.policy.reset()

            env_device = obs['policy']['robot_joint_pos'].device
            env_done = torch.zeros(num_envs, dtype=torch.bool, device=env_device)
            env_success = torch.zeros(num_envs, dtype=torch.bool, device=env_device)
            env_timesteps = torch.zeros(num_envs, dtype=torch.long, device=env_device)

            timestep = 0
            round_start = time.time()
            force_reset = False

            while not env_done.all() and not force_reset:
                if hasattr(isaaclab_env, 'task') and hasattr(isaaclab_env.task, '_env_state') and isaaclab_env.task._env_state:
                    task_prompt = [isaaclab_env.task.get_prompt(i) for i in range(num_envs)]
                elif hasattr(isaaclab_env, 'task'):
                    task_prompt = isaaclab_env.task.get_prompt()
                else:
                    task_prompt = ""

                input_batch = build_input_batch(obs, policy_bundle, camera_mode, embodiment, task_prompt)

                action = policy_bundle.policy.select_action(input_batch)
                action = policy_bundle.postprocessor(action)

                if camera_mode == "fixedcam":
                    for idx in embodiment.neck_action_indices:
                        action[:, idx] = 0.0

                obs, reward, terminated, truncated, info = isaaclab_env.step(action)
                timestep += 1

                # Track per-env termination
                step_done = (terminated | truncated).squeeze(-1)
                newly_done = step_done & ~env_done
                if newly_done.any():
                    for eid in newly_done.nonzero(as_tuple=False).squeeze(-1):
                        eid = eid.item()
                        env_done[eid] = True
                        env_success[eid] = bool(terminated[eid].item())
                        env_timesteps[eid] = timestep
                        status = 'SUCCESS' if env_success[eid] else 'truncated'
                        print(f"  env {eid}: {status} at step {timestep}")

                # Display cameras from viewed env
                prompt_str = ""
                if isinstance(task_prompt, list):
                    prompt_str = task_prompt[view_idx]
                elif isinstance(task_prompt, str):
                    prompt_str = task_prompt

                cv2.imshow('left', cv2.cvtColor(obs['policy']['left_wrist_camera_rgb'][view_idx].cpu().numpy(), cv2.COLOR_RGB2BGR))
                cv2.imshow('right', cv2.cvtColor(obs['policy']['right_wrist_camera_rgb'][view_idx].cpu().numpy(), cv2.COLOR_RGB2BGR))
                head_display = cv2.cvtColor(obs['policy']['head_camera_rgb'][view_idx].cpu().numpy(), cv2.COLOR_RGB2BGR).copy()
                cv2.putText(head_display, f"[env {view_idx}/{num_envs}] {prompt_str}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow('camera', head_display)
                cv2.imshow('fixed', cv2.cvtColor(obs['policy']['fixed_camera_rgb'][view_idx].cpu().numpy(), cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord(' '), ord('r')):
                    print("Reset pressed")
                    force_reset = True
                elif key == 81 or key == ord(','):  # left arrow or comma
                    view_idx = (view_idx - 1) % num_envs
                    print(f"Viewing env {view_idx}")
                elif key == 83 or key == ord('.'):  # right arrow or period
                    view_idx = (view_idx + 1) % num_envs
                    print(f"Viewing env {view_idx}")

            round_time = time.time() - round_start

            # Record per-env results
            for eid in range(num_envs):
                success = bool(env_success[eid].item())
                steps = int(env_timesteps[eid].item())
                if success:
                    successes += 1
                    successful_episode_lengths.append(steps)
                total_episodes += 1

            round_successes = int(env_success.sum().item())
            current_sr = successes / total_episodes
            print(f"  Round result: {round_successes}/{num_envs} succeeded, "
                  f"wall-clock: {round_time:.1f}s")
            if successful_episode_lengths:
                avg_len_s = np.mean(successful_episode_lengths) / robot_ctrl_rate
                print(f"  -> [{total_episodes}/{args.episodes}] cumulative SR: {current_sr:.2%} "
                      f"({successes}/{total_episodes}), avg success length: {avg_len_s:.1f}s")
            else:
                print(f"  -> [{total_episodes}/{args.episodes}] cumulative SR: {current_sr:.2%} "
                      f"({successes}/{total_episodes})")

        # Final summary
        success_rate = successes / total_episodes if total_episodes > 0 else 0
        print("=" * 50)
        print(f"Evaluation Results ({total_episodes} episodes):")
        print(f"  Success rate: {success_rate:.2%} ({successes}/{total_episodes})")
        if successful_episode_lengths:
            avg_len = np.mean(successful_episode_lengths)
            std_len = np.std(successful_episode_lengths)
            print(f"  Avg episode length (successful): {avg_len:.1f} +/- {std_len:.1f} steps "
                  f"({avg_len / robot_ctrl_rate:.1f} +/- {std_len / robot_ctrl_rate:.1f}s @ {robot_ctrl_rate} Hz)")
        else:
            print("  Avg episode length: N/A (no successful episodes)")
        print("=" * 50)
except Exception as e:
    import traceback
    error_msg = traceback.format_exc()
    print(f"\n{RED}ERROR during evaluation: {error_msg}{RESET}")

finally:
    if not args.headless:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
    isaaclab_env.close()
    simulation_app.close()
