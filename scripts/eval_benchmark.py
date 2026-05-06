#!/usr/bin/env python3
"""
TAVIS Benchmark Evaluation Script

Evaluates a trained policy across all tasks in a TAVIS benchmark suite,
writing per-combo JSON result files for downstream analysis.

Usage:
    # Full suite — all 5 tasks, ID + OOD-spatial:
    python eval_benchmark.py --checkpoint path/to/ckpt --suite tavis-head --robot gr1t2

    # Only headcam, only ID:
    python eval_benchmark.py --checkpoint path/to/ckpt --suite tavis-head --robot gr1t2 \
        --camera headcam --eval-modes id

    # Single task from suite:
    python eval_benchmark.py --checkpoint path/to/ckpt --suite tavis-head --robot gr1t2 \
        --tasks clutter_pick_lift

    # Ad-hoc tasks (no suite name):
    python eval_benchmark.py --checkpoint path/to/ckpt --robot gr1t2 \
        --tasks clutter_pick_lift conditional_pick --eval-modes id

    # Single-task checkpoint (convenience):
    python eval_benchmark.py --checkpoint path/to/ckpt --tasks clutter_pick_lift \
        --eval-modes id ood_spatial --episodes 200
"""

# Parse args before IsaacSim init
import argparse

parser = argparse.ArgumentParser(description="Evaluate a policy across a TAVIS benchmark suite")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained policy checkpoint")
parser.add_argument("--suite", type=str, default=None,
                    help="Suite name (e.g., 'tavis-head'). Selects all tasks + default eval modes.")
parser.add_argument("--tasks", type=str, nargs="+", default=None,
                    help="Task name(s) to evaluate. Filters --suite, or used standalone.")
parser.add_argument("--eval-modes", type=str, nargs="+", default=None,
                    help="Eval modes (default: suite's default, or ['id'] without suite)")
parser.add_argument("--robot", type=str, default="",
                    help="Robot embodiment: gr1t2|reachy2 (default: auto-detect from checkpoint)")
parser.add_argument("--camera", type=str, default="", choices=["headcam", "fixedcam", ""],
                    help="Camera mode (default: auto-detect from checkpoint)")
parser.add_argument("--model", type=str, default="",
                    help="Policy architecture (default: auto-detect from checkpoint)")
parser.add_argument("--episodes", type=int, default=100,
                    help="Episodes per (task, eval_mode) combo (default: 100)")
parser.add_argument("--lora", action="store_true", help="Load a LoRA-finetuned checkpoint (pi0 only)")
parser.add_argument("--output", type=str, default="results",
                    help="Output directory for JSON results (default: results/)")
parser.add_argument("--episode-length", type=float, default=20.0,
                    help="Episode length in seconds (default: 20)")
parser.add_argument("--num-envs", type=int, default=1,
                    help="Number of parallel environments (default: 1). Must divide --episodes.")
parser.add_argument("--robot-ctrl-rate", type=int, default=20,
                    help="Robot control rate in Hz (default: 20). Must match the effective rate "
                         "used to train the policy. The training script defaults to downsample-factor=3 "
                         "which gives 20Hz effective from 60Hz teleop data (matching LIBERO/RoboMimic). "
                         "Use 60 for policies trained without downsampling (--downsample-factor=1), "
                         "or 30 for --downsample-factor=2.")
args = parser.parse_args()

if not args.suite and not args.tasks:
    parser.error("Specify --suite or --tasks (or both)")

if args.episodes % args.num_envs != 0:
    parser.error(f"--episodes ({args.episodes}) must be divisible by --num-envs ({args.num_envs}).")

# Initialize IsaacSim (always headless for benchmark evaluation)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(num_envs=args.num_envs, enable_cameras=True, headless=True, kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error")
simulation_app = app_launcher.app

####

import contextlib
import io
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

import omni.physx
import omni.usd
from isaaclab.sim import SimulationContext

from tavis import make_tavis_env
from tavis.robots import ROBOT_MAP
from tavis.benchmark.suites import build_eval_combos
from tavis.eval.core import load_policy, run_eval_episodes, compute_summary

# Suppress IsaacLab Python loggers (raw print() calls are handled via redirect_stdout)
logging.getLogger("isaaclab").setLevel(logging.ERROR)
logging.getLogger("omni").setLevel(logging.ERROR)


# ANSI colors
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"

robot_ctrl_rate = args.robot_ctrl_rate

# Load policy (once for all combos)
policy_bundle = load_policy(
    args.checkpoint, model=args.model, camera=args.camera, robot=args.robot, lora=args.lora,
)

camera_mode = policy_bundle.camera_mode
robot_name = policy_bundle.robot_name

# Build evaluation combos
combos = build_eval_combos(suite=args.suite, tasks=args.tasks, eval_modes=args.eval_modes)

# Print run plan
checkpoint_name = Path(args.checkpoint).name
print("\n" + "=" * 60)
print(f"TAVIS Benchmark Evaluation")
print(f"=" * 60)
print(f"Checkpoint: {checkpoint_name}")
print(f"Model: {policy_bundle.model_name} | Camera: {camera_mode} | Robot: {robot_name}")
print(f"Episodes per combo: {args.episodes} | Episode length: {args.episode_length}s | Parallel envs: {args.num_envs}")
if args.suite:
    print(f"Suite: {args.suite}")
print(f"\nRunning {len(combos)} evaluation(s):")
for i, (task_name, _, eval_mode) in enumerate(combos):
    print(f"  [{i + 1}/{len(combos)}] {task_name} ({eval_mode})")
# Worst-case time estimate (assuming ~25 steps/s and all episodes timeout)
max_steps = int(args.episode_length * robot_ctrl_rate)
worst_total_steps = max_steps * args.episodes * len(combos) / args.num_envs
worst_hours = worst_total_steps / 25 / 3600  # ~25 steps/s
print(f"\nWorst-case estimate (all timeouts, ~25 steps/s): ~{worst_hours:.1f}h")
print("=" * 60 + "\n")

# Prepare output directory
output_dir = Path(args.output) / checkpoint_name
output_dir.mkdir(parents=True, exist_ok=True)

# Run evaluations
all_results = []
benchmark_start = time.time()

try:
    for combo_idx, (task_name, TaskCls, eval_mode) in enumerate(combos):
        print(f"\n{BOLD}{CYAN}[{combo_idx + 1}/{len(combos)}] Evaluating {task_name} ({eval_mode}){RESET}")
        print("-" * 40)

        result = {
            "meta": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "model": policy_bundle.model_name,
                "camera": camera_mode,
                "robot": robot_name,
                "suite": args.suite,
                "task": task_name,
                "eval_mode": eval_mode,
                "episodes_requested": args.episodes,
                "episode_length_s": args.episode_length,
                "robot_ctrl_rate": robot_ctrl_rate,
            },
        }

        env = None
        try:
            print("  Creating embodiment...", flush=True)
            embodiment = ROBOT_MAP[robot_name](enable_cameras=True)
            print("  Creating task...", flush=True)
            # ood_init_pose uses the "id" task variant; the OOD perturbation is applied at
            # the env level via InitPoseWrapper (random rest-pose perturbation on reset).
            task_variant = "id" if eval_mode == "ood_init_pose" else eval_mode
            task = TaskCls(task_variant=task_variant, episode_length_s=args.episode_length)
            print("  Creating env...", flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                env = make_tavis_env(
                    embodiment=embodiment, task=task,
                    num_envs=args.num_envs,
                    robot_ctrl_rate=robot_ctrl_rate, episode_length_s=args.episode_length,
                )
                if eval_mode == "ood_init_pose":
                    from tavis.wrappers.init_pose_wrapper import InitPoseWrapper
                    warmup_steps = max(5, int(0.5 * robot_ctrl_rate))
                    env = InitPoseWrapper(
                        env, embodiment,
                        warmup_steps=warmup_steps,
                        position_noise_std=0.10,   # arms: 10cm std
                        head_noise_std=0.175,      # head: ~10° std (decoupled from arms)
                    )
            print("  Env ready. Running episodes...", flush=True)

            episodes = run_eval_episodes(
                env, policy_bundle, camera_mode, embodiment, args.episodes,
                robot_ctrl_rate, num_envs=args.num_envs,
            )
            summary = compute_summary(episodes)

            result["episodes"] = episodes
            result["summary"] = summary

            # Print combo summary
            sr = summary["success_rate"]
            n_ok = summary["n_successful"]
            avg_s = summary["avg_length_s_successful"]
            sr_color = GREEN if sr >= 0.5 else YELLOW if sr >= 0.2 else RED
            print(f"\n  => SR: {sr_color}{sr:.1%}{RESET} ({n_ok}/{args.episodes})", end="")
            if avg_s is not None:
                print(f" | Avg success length: {avg_s:.1f}s")
            else:
                print()

        except Exception:
            error_msg = traceback.format_exc()
            print(f"\n  {RED}ERROR: {error_msg}{RESET}", flush=True)
            result["error"] = error_msg
        finally:
            # --- Full teardown for sequential env creation ---
            # 1. Close env (may partially fail, that's ok)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
                # Mark underlying env as closed to prevent __del__ crash
                unwrapped = getattr(env, 'unwrapped', None)
                if unwrapped is not None:
                    unwrapped._is_closed = True
                del env
                env = None

            # 2. Force-clear SimulationContext singleton (env.close() may
            #    have crashed before reaching clear_instance)
            sim = SimulationContext.instance()
            if sim is not None:
                try:
                    sim.clear_all_callbacks()
                except Exception:
                    pass
                SimulationContext.clear_instance()

            # 3. Detach PhysX, close USD stage, then re-open a fresh one
            #    (SimulationContext.__init__ requires a live stage)
            try:
                omni.physx.get_physx_simulation_interface().detach_stage()
            except Exception:
                pass
            try:
                omni.usd.get_context().close_stage()
            except Exception:
                pass

            # 4. Pump event loop so async stage close completes
            for _ in range(50):
                simulation_app.update()

            # 5. Open a fresh empty stage for the next env
            try:
                omni.usd.get_context().new_stage()
            except Exception:
                pass
            for _ in range(20):
                simulation_app.update()

        # Write JSON result
        result_path = output_dir / f"{task_name}_{eval_mode}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {result_path}")

        all_results.append(result)

        # Print ETA based on elapsed time
        elapsed = time.time() - benchmark_start
        combos_done = combo_idx + 1
        combos_remaining = len(combos) - combos_done
        if combos_remaining > 0:
            avg_per_combo = elapsed / combos_done
            eta_s = avg_per_combo * combos_remaining
            eta_m = eta_s / 60
            print(f"  {DIM}ETA: ~{eta_m:.0f}min ({combos_remaining} combo(s) remaining, "
                  f"{avg_per_combo:.0f}s/combo avg){RESET}")

    # Print overall summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Task':<25} {'Mode':<15} {'SR':>8} {'N':>5}")
    print("-" * 55)

    for r in all_results:
        task = r["meta"]["task"]
        mode = r["meta"]["eval_mode"]
        if "summary" in r:
            sr = r["summary"]["success_rate"]
            n = r["summary"]["n_episodes"]
            print(f"{task:<25} {mode:<15} {sr:>7.1%} {n:>5}")
        else:
            print(f"{task:<25} {mode:<15} {'ERROR':>8} {'':>5}")

    total_time = time.time() - benchmark_start
    print(f"\nTotal wall-clock: {total_time/60:.1f}min ({total_time/3600:.2f}h)")
    print("=" * 60)

finally:
    simulation_app.close()
