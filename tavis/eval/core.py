"""Shared evaluation utilities for policy evaluation in Isaac Lab.

Import this module AFTER IsaacSim AppLauncher initialization.
"""

import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors

from tavis.eval.galt import GaltParams, compute_galt


POLICY_CLASS_MAP = {
    "diffusion": DiffusionPolicy,
    "act": ACTPolicy,
    "smolvla": SmolVLAPolicy,
    "pi0": PI0Policy,
}

GREEN = "\033[92m"
RESET = "\033[0m"

PolicyBundle = namedtuple("PolicyBundle", [
    "policy", "preprocessor", "postprocessor",
    "image_transforms", "policy_class", "model_name",
    "camera_mode", "robot_name",
])


def auto_detect_from_checkpoint(checkpoint_path, model="", camera="", robot=""):
    """Auto-detect model, camera mode, and robot from checkpoint name.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        model: Explicit model name, or "" to auto-detect.
        camera: Explicit camera mode, or "" to auto-detect.
        robot: Explicit robot name, or "" to auto-detect.

    Returns:
        Tuple of (model, camera, robot) with auto-detected values filled in.

    Raises:
        ValueError: If auto-detection fails and no explicit value given.
    """
    checkpoint_name = Path(checkpoint_path).name

    if not model:
        if "pi0" in checkpoint_name:
            model = "pi0"
        elif "diffusion" in checkpoint_name:
            model = "diffusion"
        elif "smolvla" in checkpoint_name:
            model = "smolvla"
        elif "_act" in checkpoint_name:
            model = "act"
        else:
            raise ValueError(
                f"Cannot auto-detect model from checkpoint name '{checkpoint_name}'. "
                f"Please specify --model explicitly (diffusion, act, smolvla, pi0)."
            )
        print(f"{GREEN}[AUTO] model={model}  (inferred from checkpoint name){RESET}")

    if not robot:
        if "gr1t2" in checkpoint_name:
            robot = "gr1t2"
        elif "reachy2" in checkpoint_name:
            robot = "reachy2"
        else:
            raise ValueError(
                f"Cannot auto-detect robot from checkpoint name '{checkpoint_name}'. "
                f"Please specify --robot explicitly (gr1t2 or reachy2)."
            )
        print(f"{GREEN}[AUTO] robot={robot}  (inferred from checkpoint name){RESET}")

    if not camera:
        if "headcam" in checkpoint_name:
            camera = "headcam"
        elif "fixedcam" in checkpoint_name:
            camera = "fixedcam"
        else:
            raise ValueError(
                f"Cannot auto-detect camera mode from checkpoint name '{checkpoint_name}'. "
                f"Please specify --camera explicitly (headcam or fixedcam)."
            )
        print(f"{GREEN}[AUTO] camera={camera}  (inferred from checkpoint name){RESET}")

    return model, camera, robot


def load_policy(checkpoint_path, model="", camera="", robot="", lora=False):
    """Load a trained policy and its pre/post processors.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        model: Policy architecture name, or "" to auto-detect.
        camera: Camera mode, or "" to auto-detect.
        robot: Robot name, or "" to auto-detect.
        lora: Whether to load as LoRA-finetuned checkpoint (pi0 only).

    Returns:
        PolicyBundle namedtuple with all components needed for evaluation.
    """
    model, camera, robot = auto_detect_from_checkpoint(checkpoint_path, model, camera, robot)
    policy_class = POLICY_CLASS_MAP[model]

    # Enable TF32
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True

    # Load policy
    if lora and model == 'pi0':
        from peft import PeftModel
        from lerobot.configs.policies import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
        policy = PI0Policy.from_pretrained('lerobot/pi0_base', config=cfg, strict=False)
        policy = PeftModel.from_pretrained(policy, checkpoint_path)
        policy = policy.merge_and_unload()
    else:
        policy = policy_class.from_pretrained(checkpoint_path)

    policy.eval()
    policy.to('cuda')

    # Load preprocessor/postprocessor from checkpoint
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint_path,
    )

    # Image transforms (Pi0 handles its own resize to 224x224 internally)
    if policy_class == PI0Policy:
        image_transforms = None
    else:
        image_transforms = v2.Compose([v2.Resize((240, 320))])

    return PolicyBundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        image_transforms=image_transforms,
        policy_class=policy_class,
        model_name=model,
        camera_mode=camera,
        robot_name=robot,
    )


def build_input_batch(obs, policy_bundle, camera_mode, embodiment, task_prompt=""):
    """Convert raw environment observations into a policy input batch.

    Args:
        obs: Raw observation dict from env.step() or env.reset().
        policy_bundle: PolicyBundle from load_policy().
        camera_mode: "headcam" or "fixedcam".
        embodiment: Embodiment instance (used for neck_state_indices).
        task_prompt: Language prompt string or list of per-env strings for VLA models.

    Returns:
        Input batch dict ready for policy.select_action() (after preprocessor).
    """
    policy_class = policy_bundle.policy_class
    image_transforms = policy_bundle.image_transforms

    # Convert images from HWC to CHW and to float32 [0, 1]
    head_img = obs['policy']['head_camera_rgb'].permute(0, 3, 1, 2).float() / 255
    left_img = obs['policy']['left_wrist_camera_rgb'].permute(0, 3, 1, 2).float() / 255
    right_img = obs['policy']['right_wrist_camera_rgb'].permute(0, 3, 1, 2).float() / 255
    fixed_img = obs['policy']['fixed_camera_rgb'].permute(0, 3, 1, 2).float() / 255

    # Apply resize transform (Pi0 handles its own resize)
    if image_transforms is not None:
        head_img = image_transforms(head_img)
        left_img = image_transforms(left_img)
        right_img = image_transforms(right_img)
        fixed_img = image_transforms(fixed_img)

    state = obs['policy']['robot_joint_pos']

    # SmolVLA expects state size <= 32; skip first joint element
    if policy_class == SmolVLAPolicy:
        state = state[:, 1:]

    input_batch = {
        'observation.state': state,
        'observation.images.OBS_HEAD': head_img,
        'observation.images.OBS_WRIST_LEFT': left_img,
        'observation.images.OBS_WRIST_RIGHT': right_img,
        'observation.images.OBS_FIXED': fixed_img,
        'action_is_pad': False,
    }

    # Fixedcam: zero out neck state
    if camera_mode == "fixedcam":
        for idx in embodiment.neck_state_indices:
            input_batch["observation.state"][:, idx] = 0.0

    # Policy-specific adjustments
    batch_size = state.shape[0]
    if policy_class == ACTPolicy:
        input_batch["observation.state"] = input_batch["observation.state"].squeeze(1)
    elif policy_class in (SmolVLAPolicy, PI0Policy):
        if isinstance(task_prompt, list):
            input_batch["task"] = task_prompt
        else:
            input_batch["task"] = [task_prompt] * batch_size

    # Apply preprocessor (normalization + device placement)
    input_batch = policy_bundle.preprocessor(input_batch)

    return input_batch


def run_eval_episodes(env, policy_bundle, camera_mode, embodiment, n_episodes,
                      robot_ctrl_rate=60, num_envs=1):
    """Run evaluation episodes and return per-episode results.

    Supports parallel environments via num_envs > 1. Episodes are run in
    synchronized rounds of num_envs episodes each.

    Args:
        env: Isaac Lab environment (wrapped with CanonicalFrameWrapper).
        policy_bundle: PolicyBundle from load_policy().
        camera_mode: "headcam" or "fixedcam".
        embodiment: Embodiment instance (used for neck indices).
        n_episodes: Number of episodes to run.
        robot_ctrl_rate: Control frequency in Hz (for time conversion).
        num_envs: Number of parallel environments.

    Returns:
        List of dicts, one per episode:
            {"success": bool, "length_steps": int, "length_s": float,
             "wall_clock_s": float, "extra": dict}
    """
    if n_episodes % num_envs != 0:
        raise ValueError(
            f"n_episodes ({n_episodes}) must be divisible by num_envs ({num_envs}). "
            f"Try {n_episodes - (n_episodes % num_envs)} or "
            f"{n_episodes + (num_envs - n_episodes % num_envs)}."
        )

    num_rounds = n_episodes // num_envs
    episodes = []
    eval_start = time.time()

    for round_idx in range(num_rounds):
        obs, _ = env.reset()
        policy_bundle.policy.reset()





        # DEBUGGING CODE; REMOVE from final repository
        """
        import cv2                                                                                        
        from pathlib import Path
        _dump_dir = Path("/tmp/eval_cam_dump"); _dump_dir.mkdir(exist_ok=True)                            
        for cam_key in ("head_camera_rgb", "left_wrist_camera_rgb",                                       
                        "right_wrist_camera_rgb", "fixed_camera_rgb"):                                    
            imgs = obs['policy'][cam_key]  # (num_envs, H, W, 3) uint8                                    
            for i in range(imgs.shape[0]):                                                                
                bgr = cv2.cvtColor(imgs[i].cpu().numpy(), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(_dump_dir / f"{cam_key}_env{i}.png"), bgr)                                
        print(f"[dump] wrote cam images to {_dump_dir}")  
        #"""
 


        env_device = obs['policy']['robot_joint_pos'].device
        env_done = torch.zeros(num_envs, dtype=torch.bool, device=env_device)
        env_success = torch.zeros(num_envs, dtype=torch.bool, device=env_device)
        env_timesteps = torch.zeros(num_envs, dtype=torch.long, device=env_device)

        # GALT per-env buffers (populated only for headcam runs).
        # Actions are appended step-by-step for each not-yet-done env; on a
        # successful termination we stack and pass to compute_galt once.
        action_buffers = [[] for _ in range(num_envs)]
        env_galt = [None] * num_envs
        galt_params = GaltParams()

        round_start = time.time()
        timestep = 0

        while not env_done.all():
            # Build per-env prompts when task supports it, else single string
            if hasattr(env, 'task') and hasattr(env.task, '_env_state') and env.task._env_state:
                task_prompt = [env.task.get_prompt(i) for i in range(num_envs)]
            elif hasattr(env, 'task'):
                task_prompt = env.task.get_prompt()
            else:
                task_prompt = ""
            input_batch = build_input_batch(obs, policy_bundle, camera_mode, embodiment, task_prompt)

            action = policy_bundle.policy.select_action(input_batch)
            action = policy_bundle.postprocessor(action)

            # Fixedcam: zero out neck actions before stepping
            if camera_mode == "fixedcam":
                for idx in embodiment.neck_action_indices:
                    action[:, idx] = 0.0

            # Buffer actions for GALT (headcam only; skip terminated envs).
            if camera_mode == "headcam":
                act_np = action.detach().cpu().numpy()
                active = (~env_done).nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in active:
                    action_buffers[i].append(act_np[i])

            obs, reward, terminated, truncated, info = env.step(action)
            timestep += 1

            if timestep % 100 == 0:
                elapsed = time.time() - round_start
                print(f"    step {timestep}, elapsed {elapsed:.1f}s, "
                      f"{timestep / elapsed:.1f} steps/s, "
                      f"{timestep * num_envs / elapsed:.1f} env-steps/s")

            # Check per-env termination (only record first completion per env)
            step_done = (terminated | truncated).squeeze(-1)
            newly_done = step_done & ~env_done

            if newly_done.any():
                for env_idx in newly_done.nonzero(as_tuple=False).squeeze(-1):
                    env_idx = env_idx.item()
                    env_done[env_idx] = True
                    env_success[env_idx] = bool(terminated[env_idx].item())
                    env_timesteps[env_idx] = timestep
                    # GALT on successful headcam episodes only.
                    if camera_mode == "headcam" and env_success[env_idx] \
                            and action_buffers[env_idx]:
                        act_arr = np.stack(action_buffers[env_idx])
                        env_galt[env_idx] = compute_galt(
                            act_arr, fps=float(robot_ctrl_rate),
                            params=galt_params,
                        )
                    if num_envs > 1:
                        print(f"  env {env_idx}: "
                              f"{'SUCCESS' if env_success[env_idx] else 'truncated'} "
                              f"at step {timestep}")

        round_time = time.time() - round_start

        # Record per-env results
        for env_idx in range(num_envs):
            success = bool(env_success[env_idx].item())
            steps = int(env_timesteps[env_idx].item())
            extra = {}
            gr = env_galt[env_idx]
            if gr is not None:
                extra["galt_s"] = gr.galt_s
                extra["galt_reason"] = gr.reason
                extra["galt_arm"] = gr.arm
                extra["galt_t_head_arrival_s"] = gr.t_head_arrival_s
                extra["galt_t_hand_arrival_s"] = gr.t_hand_arrival_s
                extra["galt_arm_reach_s"] = gr.arm_reach_s
            episodes.append({
                "success": success,
                "length_steps": steps,
                "length_s": steps / robot_ctrl_rate,
                "wall_clock_s": round(round_time, 2),
                "extra": extra,
            })

        # Print progress
        ep_so_far = len(episodes)
        sr_so_far = sum(e["success"] for e in episodes) / ep_so_far
        round_successes = int(env_success.sum().item())

        # ETA based on elapsed time
        elapsed_total = time.time() - eval_start
        rounds_remaining = num_rounds - (round_idx + 1)
        avg_per_round = elapsed_total / (round_idx + 1)
        eta_s = avg_per_round * rounds_remaining

        if num_envs > 1:
            print(f"  Round {round_idx + 1}/{num_rounds}: "
                  f"{round_successes}/{num_envs} succeeded, "
                  f"wall-clock: {round_time:.1f}s")
            print(f"  -> [{ep_so_far}/{n_episodes}] cumulative SR: {sr_so_far:.1%}"
                  f" | ETA: {eta_s/60:.1f}min\n")
        else:
            ep = episodes[-1]
            print(f"  Episode {ep_so_far}/{n_episodes}: "
                  f"{'SUCCESS' if ep['success'] else 'FAIL'} "
                  f"({ep['length_steps']} steps, {round_time:.1f}s) | "
                  f"Running SR: {sr_so_far:.1%}"
                  f" | ETA: {eta_s/60:.1f}min")

    return episodes


def compute_summary(episodes):
    """Compute summary statistics from per-episode results.

    Episode length stats are computed over successful episodes only
    (failures always timeout, and are already captured by success_rate).

    Args:
        episodes: List of episode result dicts from run_eval_episodes().

    Returns:
        Dict with summary statistics.
    """
    n = len(episodes)
    n_successful = sum(e["success"] for e in episodes)
    success_rate = n_successful / n if n > 0 else 0.0

    summary = {
        "success_rate": round(success_rate, 4),
        "n_episodes": n,
        "n_successful": n_successful,
    }

    successful = [e for e in episodes if e["success"]]
    if successful:
        lengths_s = [e["length_s"] for e in successful]
        lengths_steps = [e["length_steps"] for e in successful]
        summary["avg_length_s_successful"] = round(float(np.mean(lengths_s)), 2)
        summary["std_length_s_successful"] = round(float(np.std(lengths_s)), 2)
        summary["avg_length_steps_successful"] = round(float(np.mean(lengths_steps)), 1)
    else:
        summary["avg_length_s_successful"] = None
        summary["std_length_s_successful"] = None
        summary["avg_length_steps_successful"] = None

    # GALT stats — only emit keys if any episode has GALT data (i.e., headcam run).
    has_galt = any("galt_reason" in e.get("extra", {}) for e in episodes)
    if has_galt:
        galt_vals = []
        reason_counts = {}
        for e in successful:
            ex = e.get("extra", {})
            reason = ex.get("galt_reason")
            if reason is not None:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            galt_s = ex.get("galt_s")
            if galt_s is not None:
                galt_vals.append(galt_s)
        summary["galt_reason_counts"] = reason_counts
        summary["galt_n_valid"] = len(galt_vals)
        if galt_vals:
            summary["galt_mean_s"] = round(float(np.mean(galt_vals)), 4)
            summary["galt_median_s"] = round(float(np.median(galt_vals)), 4)
            summary["galt_std_s"] = round(float(np.std(galt_vals)), 4)
        else:
            summary["galt_mean_s"] = None
            summary["galt_median_s"] = None
            summary["galt_std_s"] = None

    return summary
