#!/usr/bin/env python3
"""
Policy Training Script

Usage:
    python train_policy.py --dataset <path> [--model diffusion|act|smolvla|pi0] [--camera headcam|fixedcam]
                           [--steps 10000] [--lora] [--task <name>]

    --dataset   Path to the LeRobot dataset directory (required)
    --model     Policy architecture (default: pi0)
    --camera    Camera mode: headcam or fixedcam (default: headcam)
    --steps     Number of training steps (default: 10000)
    --lora      Enable LoRA finetuning for pi0 (reduces VRAM, enables larger batch sizes)
    --task      Filter the dataset to a single task class (e.g. 'clutter_pick_lift'). The
                released TAVIS datasets are multi-task suites (one per robot); use --task
                to train on a single task within a suite. Omit to train multi-task on all
                tasks present in --dataset.

Examples:
    # Multi-task training on a whole suite:
    python train_policy.py --dataset datasets/tavis-head-gr1t2

    # Single-task training (filter the multi-task suite to one task):
    python train_policy.py --dataset datasets/tavis-head-gr1t2 \\
        --task clutter_pick_lift --model diffusion --steps 30000

    # Fixed-camera ablation:
    python train_policy.py --dataset datasets/tavis-head-gr1t2 \\
        --task clutter_pick_lift --camera fixedcam
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2


def set_seed(seed: int, deterministic: bool = False):
    """Seed all RNGs used by the training stack for reproducible runs.

    Covers: Python `random`, NumPy, PyTorch (CPU + CUDA), `PYTHONHASHSEED`.
    With `deterministic=True` also forces cuDNN deterministic kernels (slower
    but fully reproducible on the same hardware). Default False because the
    main use case is *seed variance measurement*, not bit-exact repro — small
    non-deterministic CUDA ops are fine and keep training fast.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[seed] set_seed({seed}, deterministic={deterministic})")

# Head camera augmentations for headcam mode (applied in training loop)
# Note: rotation disabled since head roll is locked during data collection
#head_cam_augment = v2.Compose([
#    v2.RandomAffine(
#        degrees=0,                 # no rotation (head roll locked during collection)
#        translate=(0.05, 0.05),    # ±5% xy translation for gaze variability
#        fill=0,
#    ),
#    #v2.ColorJitter(
#    #    brightness=(0.9, 1.1),
#    #    contrast=(0.9, 1.1),
#    #),
#])

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

from tavis.math_utils import fix_quat_stats, fix_constant_dims

# Neck joint indices per robot, duplicated here to avoid importing the embodiment files
# which require isaaclab (not available on training-only clusters).
# Canonical source: GR1T2Embodiment / Reachy2Embodiment in tavis/robots/
# TODO: refactor into a shared lightweight constants file under robots/
NECK_INDICES = {
    "gr1t2":  {"action": [14, 15, 16], "state": [11, 16, 21]},
    "reachy2": {"action": [14, 15, 16], "state": [5, 8, 9]},
}


# Enable bfloat16 training
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    print(torch.backends.cudnn.allow_tf32 ,"*******")
    torch.backends.cudnn.allow_tf32 = True
    print(torch.backends.cudnn.allow_tf32 ,"*******")
print('--')
if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
    print(torch.backends.cuda.matmul.allow_tf32 ,"*******")
    torch.backends.cuda.matmul.allow_tf32 = True
    print(torch.backends.cuda.matmul.allow_tf32 ,"*******")
train_bf16 = True  # recommended for pi0 finetuning; PaLiGemma is pretrained in bf16


parser = argparse.ArgumentParser(description="Train a policy on a LeRobot dataset")
parser.add_argument("--dataset", type=str, required=True, help="Path to the LeRobot dataset directory")
parser.add_argument("--model", type=str, default="pi0", choices=["diffusion", "act", "smolvla", "pi0"],
                    help="Policy architecture (default: pi0)")
parser.add_argument("--camera", type=str, default="headcam", choices=["headcam", "fixedcam"],
                    help="Camera mode (default: headcam)")
parser.add_argument("--robot", type=str, default="",
                    help="Robot embodiment: 'gr1t2' or 'reachy2'. If omitted, auto-detected from "
                         "the dataset path by searching for 'gr1t2' or 'reachy2' as a substring. "
                         "Errors out if neither is found. Only used in fixedcam mode to zero out "
                         "the correct neck joints in state and action.")
parser.add_argument("--task", type=str, default=None,
                    help="Filter the dataset to a single task class. Accepts either a snake-case "
                         "key from tavis.tasks.TASK_MAP (e.g. 'clutter_pick_lift') or the class "
                         "name as recorded in the dataset (e.g. 'ClutterPickLiftTask'). The TAVIS "
                         "release ships multi-task suite datasets; pass --task to train on a single "
                         "task within a suite. Omit to train on all tasks present in --dataset.")
parser.add_argument("--lora", action="store_true", help="Enable LoRA finetuning for pi0 (reduces VRAM, enables larger batch sizes)")
parser.add_argument("--steps", type=int, default=10000, help="Number of training steps (default: 10000)")
parser.add_argument("--downsample-factor", type=int, default=3,
                    help="Downsample data by sampling every Nth frame from the original dataset, "
                         "training as if the data were at fps/N Hz. Default: 3 (i.e. 60Hz → 20Hz, "
                         "matching LIBERO/RoboMimic). At eval, the env's robot_ctrl_rate must be "
                         "set to fps/N to match. Set to 1 to disable downsampling.")
parser.add_argument("--output_dir", type=str, default="",
                    help="Output directory for checkpoints. If omitted, defaults to "
                         "checkpoints/{dataset_name}_{model}_{camera}")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for Python/NumPy/PyTorch (default: 42). Change "
                         "to measure training-run variance.")
parser.add_argument("--deterministic", action="store_true",
                    help="Force bit-exact reproducibility (cuDNN deterministic kernels, "
                         "benchmark=False). Slower. Only needed for exact repro; for "
                         "seed-variance measurements, leave off.")
args = parser.parse_args()

# Seed everything as early as possible (before dataset / policy construction)
set_seed(args.seed, deterministic=args.deterministic)
DOWNSAMPLE_FACTOR = args.downsample_factor
assert DOWNSAMPLE_FACTOR >= 1, "--downsample-factor must be >= 1"

root = args.dataset
repo_id = Path(root).name
model = args.model
camera_mode = args.camera

if not args.robot:
    dataset_name = Path(root).name
    if "gr1t2" in dataset_name:
        args.robot = "gr1t2"
    elif "reachy2" in dataset_name:
        args.robot = "reachy2"
    else:
        parser.error(f"Cannot auto-detect robot from dataset name '{dataset_name}'. "
                     f"Please specify --robot explicitly (gr1t2 or reachy2).")
# neck_action_indices / neck_state_indices are only used in fixedcam mode to zero out
# neck joints so the policy learns to ignore head movement.
neck_action_indices = NECK_INDICES[args.robot]["action"]
neck_state_indices = NECK_INDICES[args.robot]["state"]



# Features to use based on camera mode (main camera first, then wrists, then state)
if camera_mode == "headcam":
    features_to_use = ["observation.images.OBS_HEAD",
                       "observation.images.OBS_WRIST_LEFT",
                       "observation.images.OBS_WRIST_RIGHT",
                       "observation.state"]
elif camera_mode == "fixedcam":
    features_to_use = ["observation.images.OBS_FIXED",
                       "observation.images.OBS_WRIST_LEFT",
                       "observation.images.OBS_WRIST_RIGHT",
                       "observation.state"]

if args.output_dir:
    output_path = args.output_dir
else:
    task_suffix = f"__{args.task}" if args.task else ""
    output_path = "checkpoints/"+repo_id+task_suffix+"_"+model+"_"+camera_mode


## Action chunk lengths assume 60Hz data, then scaled by DOWNSAMPLE_FACTOR.
## At default DOWNSAMPLE_FACTOR=3, we get 20Hz effective (matching LIBERO/RoboMimic):
## predict ~0.8s (16 frames), execute ~0.4s (8 frames) — same as the original
## Diffusion Policy paper. The 48/24 base is chosen so horizon stays divisible by 8
## (required by diffusion's U-Net with 3 downsampling layers) at common factors:
##   factor=1: 48/24 (60Hz)   factor=2: 24/12 (30Hz)   factor=3: 16/8 (20Hz)
HORIZON_60HZ = 48          # ~0.8s of prediction at 60Hz
N_ACTION_STEPS_60HZ = 24   # ~0.4s of execution at 60Hz

_horizon = max(8, HORIZON_60HZ // DOWNSAMPLE_FACTOR)
_n_action_steps = max(1, N_ACTION_STEPS_60HZ // DOWNSAMPLE_FACTOR)

## DIFFUSION POLICY
if model == 'diffusion':
    n_obs_steps = 2
    horizon = _horizon
    n_action_steps = _n_action_steps
    drop_n_last_frames = (horizon - n_action_steps - 2 + 1) * DOWNSAMPLE_FACTOR

    down_dims = (512,1024,2048) # 271M params
    #down_dims = (256, 512, 512) # 46M params  but not really faster
    #down_dims = (128, 256, 256) # 21M params   but not really faster


## ACT POLICY
if model == 'act':
    n_obs_steps = 1
    chunk_size = _horizon
    n_action_steps = _n_action_steps

if model == 'smolvla':
    n_obs_steps = 1
    chunk_size = _horizon
    n_action_steps = max(1, _horizon // 3)  # ~1/3 of chunk
    max_state_dim = 48

if model == 'pi0':
    n_obs_steps = 1
    chunk_size = _horizon
    n_action_steps = _n_action_steps




downscale_images = True

# Batch sizes per model (H100, 3x images, 320x240 downscaled except pi0 which resizes internally to 224x224)
# diffusion: <10GB VRAM, ~75% GPU utilization; 32 fits easily
# act:       TODO
# smolvla:   TODO
# pi0:       ~67GB VRAM; gradient checkpointing enabled; 16 is a good tradeoff
BATCH_SIZES = {"diffusion": 16, "act": 16, "smolvla": 16, "pi0": 16}
batch_size = BATCH_SIZES[model]

training_steps = args.steps
log_freq = 100



def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path(output_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda")

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root, force_cache_sync=False)

    # Optional single-task filter for multi-task suite datasets. Reads the
    # per-episode `tasks` field from meta/episodes/*.parquet and keeps only
    # episodes whose recorded task class matches. `filtered_episodes` is
    # passed to LeRobotDataset(episodes=...) below; None = use all episodes.
    filtered_episodes = None
    if args.task:
        import pandas as pd
        from tavis.tasks import TASK_MAP
        target_class = TASK_MAP[args.task].__name__ if args.task in TASK_MAP else args.task
        ep_files = sorted(Path(root).glob("meta/episodes/**/*.parquet"))
        if not ep_files:
            parser.error(f"--task requires meta/episodes/*.parquet under {root}")
        df = pd.concat([pd.read_parquet(f) for f in ep_files])
        def _has_target(tasks_field):
            xs = list(tasks_field) if not isinstance(tasks_field, str) else [tasks_field]
            return target_class in xs
        mask = df["tasks"].apply(_has_target)
        filtered_episodes = sorted(df.loc[mask, "episode_index"].astype(int).tolist())
        if not filtered_episodes:
            seen = sorted({
                t for tasks_field in df["tasks"]
                for t in (list(tasks_field) if not isinstance(tasks_field, str) else [tasks_field])
            })
            parser.error(f"--task: no episodes for class '{target_class}' in {root}. "
                         f"Available task classes: {seen}")
        print(f"--task: kept {len(filtered_episodes)} of {len(df)} episodes for class '{target_class}'.")

    # Check if dataset has stats available
    print(f"Dataset stats available: {dataset_metadata.stats is not None}")
    if dataset_metadata.stats is not None:
        print(f"Stats keys: {list(dataset_metadata.stats.keys())}")
    else:
        print("Warning: No pre-computed stats found. Normalization will be skipped or you need to compute stats.")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: features[key] for key in features_to_use if key in features and key not in output_features}

    # Update image feature dimensions to match the resize transform
    # Pi0 handles its own resize to 224x224 internally via resize_with_pad
    if downscale_images and model not in ('pi0',):
        for key, feature in features.items():
            if key.startswith("observation.images."):
                feature.shape = (3, 240, 320)  # Match your resize transform

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    #input_features['observation.state'].shape = (14,)  # joints state replaced with left_eef_pos/quat+right_eef_pos/quat; not however that no velocities are used

    if model == 'diffusion':
        cfg = DiffusionConfig(input_features=input_features, output_features=output_features,
                            n_obs_steps=n_obs_steps,
                            horizon=horizon,
                            n_action_steps=n_action_steps,
                            drop_n_last_frames=drop_n_last_frames,
                            crop_shape=None,
                            down_dims=down_dims)

    elif model == 'act':
        cfg = ACTConfig(input_features=input_features, output_features=output_features,
                        n_obs_steps=n_obs_steps,
                        chunk_size=chunk_size,
                        n_action_steps=n_action_steps)

    elif model == 'smolvla':
        input_features['observation.state'].shape = (32,)

        cfg = SmolVLAConfig(input_features=input_features, output_features=output_features,
                            n_obs_steps=n_obs_steps,
                            chunk_size=chunk_size,
                            n_action_steps=n_action_steps)

    elif model == 'pi0':
        # Pi0 pads state/action to max_dim; set to match our dims so pretrained state_proj loads
        state_dim = input_features['observation.state'].shape[0]
        action_dim = output_features['action'].shape[0]
        cfg = PI0Config(input_features=input_features, output_features=output_features,
                        n_obs_steps=n_obs_steps,
                        chunk_size=chunk_size,
                        n_action_steps=n_action_steps,
                        max_state_dim=max(state_dim, 32),
                        max_action_dim=max(action_dim, 32),
                        gradient_checkpointing=True)

    # Instantiate policy (no longer accepts dataset_stats in new lerobot version).
    # Normalization is now handled by separate preprocessor/postprocessor.

    # Fix quaternion stats to use identity normalization (preserves geometric structure)
    fixed_stats = fix_quat_stats(dataset_metadata.stats)
    fixed_stats = fix_constant_dims(fixed_stats)

    if model == 'diffusion':
        policy = DiffusionPolicy(cfg)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=fixed_stats)
    elif model == 'act':
        policy = ACTPolicy(cfg)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=fixed_stats)
    elif model == 'smolvla':
        # Slice the state normalization stats to match the sliced state vector
        modified_stats = fixed_stats
        # skip the first element of the state stats (it's the robot_joint_pos)
        if 'observation.state' in modified_stats:
            if 'mean' in modified_stats['observation.state']:
                modified_stats['observation.state']["mean"] = modified_stats['observation.state']["mean"][1:]
            if 'std' in modified_stats['observation.state']:
                modified_stats['observation.state']["std"] = modified_stats['observation.state']["std"][1:]
            if 'min' in modified_stats['observation.state']:
                modified_stats['observation.state']["min"] = modified_stats['observation.state']["min"][1:]
            if 'max' in modified_stats['observation.state']:
                modified_stats['observation.state']["max"] = modified_stats['observation.state']["max"][1:]

        policy = SmolVLAPolicy(cfg)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=modified_stats)
        # Only load compatible weights (skip normalization layers)
        state_dict = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base').state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items()
                                if 'normalize_inputs' not in k and 'normalize_targets' not in k
                                and 'unnormalize_outputs' not in k}
        policy.load_state_dict(filtered_state_dict, strict=False)

    elif model == 'pi0':
        # Load pretrained pi0_base with our config (strict=False for state_proj dim mismatch)
        policy = PI0Policy.from_pretrained('lerobot/pi0_base', config=cfg, strict=False)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=fixed_stats)



    # Apply LoRA if requested (pi0 only)
    # LoRA on VLM only (PaLiGemma vision + language) to prevent catastrophic forgetting
    # Expert (Gemma 300M) + projections are fully trained since they need to learn new action space
    if args.lora and model == 'pi0':
        from peft import LoraConfig, get_peft_model
        # Only target VLM attention, not expert attention
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=r".*paligemma\..*self_attn\.(q|v)_proj",  # LoRA on VLM only
            modules_to_save=[                                         # fully train expert + projections
                "state_proj", "action_in_proj", "action_out_proj",
                "action_time_mlp_in", "action_time_mlp_out",
            ],
            lora_dropout=0.05,
        )
        policy = get_peft_model(policy, lora_config)
        # Also unfreeze the entire expert so it trains fully
        for name, param in policy.named_parameters():
            if "gemma_expert" in name and "lora" not in name:
                param.requires_grad = True
        policy.print_trainable_parameters()

    policy.train()
    policy.to(device)

    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    observation_delta_indices = cfg.observation_delta_indices
    if observation_delta_indices is None:
        observation_delta_indices = [0]  # pi0 returns None; use single current frame
    if model == 'act':
        observation_delta_indices = [0]
    # Multiply delta indices by DOWNSAMPLE_FACTOR so we sample every Nth frame
    # from the original dataset, effectively training at fps/N Hz.
    delta_timestamps = {
        "observation.images.OBS_HEAD": [DOWNSAMPLE_FACTOR * i / dataset_metadata.fps for i in observation_delta_indices],
        "observation.images.OBS_FIXED": [DOWNSAMPLE_FACTOR * i / dataset_metadata.fps for i in observation_delta_indices],
        "observation.images.OBS_WRIST_LEFT": [DOWNSAMPLE_FACTOR * i / dataset_metadata.fps for i in observation_delta_indices],
        "observation.images.OBS_WRIST_RIGHT": [DOWNSAMPLE_FACTOR * i / dataset_metadata.fps for i in observation_delta_indices],
        "observation.state": [DOWNSAMPLE_FACTOR * i / dataset_metadata.fps for i in observation_delta_indices],
        "observation.left_eef_pos": [DOWNSAMPLE_FACTOR * i / dataset_metadata.fps for i in observation_delta_indices],
        "observation.left_eef_quat": [DOWNSAMPLE_FACTOR * i / dataset_metadata.fps for i in observation_delta_indices],
        "observation.right_eef_pos": [DOWNSAMPLE_FACTOR * i / dataset_metadata.fps for i in observation_delta_indices],
        "observation.right_eef_quat": [DOWNSAMPLE_FACTOR * i / dataset_metadata.fps for i in observation_delta_indices],
        "action": [DOWNSAMPLE_FACTOR * (i+1) / dataset_metadata.fps for i in cfg.action_delta_indices], # TODO: IMPORTANT!
    }
    print('*Delta indices, obs and action: ', cfg.observation_delta_indices, cfg.action_delta_indices)

    # Pi0 handles its own resize to 224x224 internally
    if model == 'pi0':
        image_transforms = None
    else:
        image_transforms = v2.Compose([v2.Resize((240, 320))]) if downscale_images else None

    # We can then instantiate the dataset with these delta_timestamps configuration.
    # `episodes=filtered_episodes` is None for multi-task training (use all
    # episodes) or a list of episode indices when --task selected one task.
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        video_backend="pyav",
        episodes=filtered_episodes,
    )
    # Seeded generator so batch-sampling order is deterministic under the CLI seed.
    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)

    def _worker_init_fn(worker_id):
        # Each dataloader worker gets its own deterministic RNG derived from base seed.
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=16,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        generator=dl_generator,
        worker_init_fn=_worker_init_fn,
    )

    print(f"\nDataset length: {len(dataset)}")


    # Then we create our optimizer and dataloader for offline training.
    if model == 'pi0':
        # Use pi0's recommended optimizer preset (lr=2.5e-5, weight_decay=0.01)
        optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    # Run training loop.
    step = 0
    done = False
    last_t = time.time()

    em_loss = 0
    em_alpha = 0.9

    while not done:
        for batch in dataloader:
            # Filter to only features we want to use
            batch = {k:v for k, v in batch.items() if k in features_to_use + ["action", "language_instruction"]}

            # Apply augmentations ONLY to head camera (gaze variability robustness)
            #if camera_mode == "headcam" and "observation.images.OBS_HEAD" in batch:
            #    imgs = batch["observation.images.OBS_HEAD"]
            #    B, T, C, H, W = imgs.shape
            #    imgs_aug = head_cam_augment(imgs.view(B * T, C, H, W))
            #    batch["observation.images.OBS_HEAD"] = imgs_aug.view(B, T, C, H, W)

            batch["action_is_pad"] = torch.zeros(batch["action"].shape[:-1], dtype=torch.bool)

            # Apply fixedcam mode modifications: zero out head movement data
            if camera_mode == "fixedcam":
                # Zero out neck actions
                if neck_action_indices is not None:
                    for idx in neck_action_indices:
                        batch["action"][:, :, idx] = 0.0
                # Zero out neck state
                if neck_state_indices is not None:
                    for idx in neck_state_indices:
                        batch["observation.state"][:, :, idx] = 0.0

            if model == 'act':
                batch["observation.state"] = batch["observation.state"].squeeze(1)
            elif model == 'smolvla':
                batch["task"] = batch["language_instruction"]
                batch["observation.state"] = batch["observation.state"][:, :, 1:]
            elif model == 'pi0':
                batch["task"] = batch["language_instruction"]
                # Pi0 expects [B, D] state and [B, C, H, W] images, not [B, T, ...] from dataloader
                batch["observation.state"] = batch["observation.state"].squeeze(1)  # [B,1,D] -> [B,D]
                for img_key in [k for k in batch if k.startswith("observation.images.")]:
                    batch[img_key] = batch[img_key].squeeze(1)  # [B,1,C,H,W] -> [B,C,H,W]

            if step==0:
                print("Batch keys before preprocessor:", batch.keys())

            # Apply preprocessor (handles ALL normalization and device placement)
            batch = preprocessor(batch)

            if step==0:
                print("Batch keys after preprocessor:", batch.keys())

            # Use autocast for mixed precision training with bfloat16
            if train_bf16:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss, _ = policy.forward(batch)
            else:
                loss, _ = policy.forward(batch)

            # Standard backward pass (no scaling needed for bfloat16)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            em_loss = em_alpha * em_loss + (1 - em_alpha) * loss.item()

            if step % log_freq == 0:
                current_time = time.time()
                time_s = (training_steps - step) / (log_freq / (current_time - last_t))
                time_h_h = time_s // 3600
                time_h_m = (time_s % 3600) // 60
                print(f"step: {step} loss: {loss.item():.3f} em_loss: {em_loss:.3f} time: {current_time - last_t:.3f}s    est remaining: {time_s:.0f}s ({time_h_h}h {time_h_m}m)")
                last_t = current_time
            step += 1
            if step >= training_steps:
                done = True
                break

            """
            checkpoint_freq = 15000 if model == 'pi0' else 50000
            if step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"step_{step}"
                policy.save_pretrained(checkpoint_dir)
                if args.lora:  # PEFT save doesn't include base config
                    cfg.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
            """

    # Save a policy checkpoint along with preprocessor and postprocessor.
    policy.save_pretrained(output_directory)
    if args.lora:  # PEFT save doesn't include base config
        cfg.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

if __name__ == "__main__":
    main()
