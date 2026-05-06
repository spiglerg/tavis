"""GALT (Gaze-Action Lead Time) detector — proprio variant.

Operates on a single episode's commanded-action trajectory. No sim state, no
target-object knowledge required → real-robot compatible.

Primary metric:
    GALT = t_hand_arrival - t_head_arrival          (arrival-arrival, seconds)

Anchor for t_hand_arrival:
    latest gripper state change on the active arm (close event for grasp-
    and-lift tasks, open event for place tasks — uniform definition).

Multi-grasp extension (not implemented here, trivial to add later):
    iterate over all gripper state changes as anchors and return a list of
    GALTs; one per grasp/place event. Useful for long-horizon tasks. Current
    implementation takes the *latest* event only.

Reason codes returned in GaltResult.reason:
    "ok"                - valid GALT
    "ok_post_anchor"    - valid, but head arrival occurred after hand arrival
                          (no pre-anchor fixation found; non-anticipatory)
    "no_grasp"          - no gripper state change on either arm
    "no_hand_onset"     - no stable-to-motion transition before anchor
    "no_fixation"       - no head settling in search window
    "outlier_low"       - GALT < outlier_min_s
    "outlier_high"      - GALT > outlier_max_s
    "ambiguous_arms"    - both arms produced a valid GALT (should be rare)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Parameter & result dataclasses ────────────────────────────────────────────


@dataclass
class GaltParams:
    v_hand_thresh: float = 0.05          # m/s — commanded EE linear velocity floor
    v_sac_thresh: float = 0.10           # rad/s — commanded neck angular velocity floor
    K_fix_s: float = 0.080               # arrival persistence
    min_stable_for_onset_s: float = 0.300
    lookback_s: float = 3.0              # backward search horizon from anchor
    forward_slack_s: float = 0.5         # post-anchor search slack
    outlier_min_s: float = -0.5
    outlier_max_s: float = 4.0
    arrival_margin_rad: float = 0.05     # L-inf joint-space tolerance for refinement
                                         # (≈2.9°; head is considered "at final fixation"
                                         # once its commanded joints are within this of
                                         # their settled value).


@dataclass
class ActionLayout:
    """Index mapping for the TAVIS 19D unified action space."""
    left_ee_xyz: tuple = (0, 1, 2)
    right_ee_xyz: tuple = (7, 8, 9)
    neck: tuple = (14, 15, 16)
    left_gripper: int = 17
    right_gripper: int = 18


@dataclass
class GaltResult:
    galt_s: Optional[float]
    reason: str
    arm: Optional[str]                   # "left" | "right" | None
    t_head_arrival_s: Optional[float] = None
    t_hand_arrival_s: Optional[float] = None
    t_hand_onset_s: Optional[float] = None
    arm_reach_s: Optional[float] = None  # t_hand_arrival - t_hand_onset
    per_arm: dict = field(default_factory=dict)  # debug: per-arm attempt record


# ── Internal helpers ─────────────────────────────────────────────────────────


def _velocity(pos: np.ndarray, fps: float) -> np.ndarray:
    """Per-step speed magnitude from a position/angle trajectory.

    pos: (T, D) or (T,). Returns (T,) speed (L2 norm of per-step derivative).
    Uses forward difference, padded at the end to keep shape T.
    """
    if pos.ndim == 1:
        pos = pos[:, None]
    diff = np.diff(pos, axis=0)                       # (T-1, D)
    speed = np.linalg.norm(diff, axis=1) * fps        # (T-1,)
    speed = np.concatenate([speed, speed[-1:]])       # pad to T
    return speed


def _find_gripper_events(grip_cmd: np.ndarray) -> list[int]:
    """Indices where the gripper command crosses 0 (sign change).

    Returns the index of the *post-crossing* step.
    """
    # Binarize at 0 (negative → 0, non-negative → 1).
    # Avoids sign(0)=0 pitfalls.
    b = (grip_cmd >= 0).astype(np.int8)
    changes = np.where(np.diff(b) != 0)[0] + 1
    return changes.tolist()


def _backward_search_hand_onset(
    ee_speed: np.ndarray,
    anchor: int,
    v_thresh: float,
    min_stable_steps: int,
) -> Optional[int]:
    """Walk backward from anchor. Find the latest stable-to-motion transition.

    A "stable interval" is a run of ≥ min_stable_steps steps with speed
    below v_thresh. Onset = the step right after such an interval ends,
    i.e., the first motion step after the pre-approach rest.

    Returns None if no such transition exists before the anchor.
    """
    if anchor <= 0:
        return None
    stable_run = 0
    # Walk backward from anchor-1; the *first* time we accumulate enough stable
    # steps, the onset is the step that ended that stability (one past).
    for t in range(anchor - 1, -1, -1):
        if ee_speed[t] < v_thresh:
            stable_run += 1
            if stable_run >= min_stable_steps:
                # motion onset = step after the stable block ended.
                # The block ended at t (inclusive), so motion resumed at t + stable_run
                # going forward — but we want the first motion step after this block,
                # which is (t + stable_run) in the forward direction.
                # Equivalently: t + min_stable_steps.
                onset = t + stable_run
                return onset if onset < anchor else None
        else:
            stable_run = 0
    return None


def _find_settled_intervals(
    neck_speed: np.ndarray,
    v_thresh: float,
    k_persist: int,
    start_t: int,
    end_t: int,
) -> list[int]:
    """Start indices of settled intervals within [start_t, end_t).

    A settled interval starts at the first step of a ≥ k_persist run with
    speed < v_thresh. We return the *start* of each such run.
    """
    start_t = max(0, start_t)
    end_t = min(len(neck_speed), end_t)
    if end_t <= start_t:
        return []
    # Find boolean mask of "is stable"
    stable = neck_speed[start_t:end_t] < v_thresh
    starts = []
    run_len = 0
    run_start = None
    recorded = False
    for i, s in enumerate(stable):
        if s:
            if run_len == 0:
                run_start = i + start_t
                recorded = False
            run_len += 1
            if run_len == k_persist and not recorded:
                starts.append(run_start)
                recorded = True
        else:
            run_len = 0
            run_start = None
            recorded = False
    return starts


def _refine_arrival_by_position(
    neck_pos: np.ndarray,
    settled_t: int,
    margin_rad: float,
) -> int:
    """Walk back from ``settled_t`` to the earliest timestep ``t'`` such that
    from ``t'`` through ``settled_t`` the commanded neck joints stayed within
    ``margin_rad`` (L-inf) of their settled value ``neck_pos[settled_t]``.

    Handles the common "smooth deceleration" case: the head keeps moving
    slowly long after it's essentially at the target direction, so a pure
    velocity-threshold detector fires late. This refinement anchors on the
    target direction instead — "when did the head first get close to where
    it ends up, and stay there".
    """
    q_final = neck_pos[settled_t]
    for t in range(settled_t - 1, -1, -1):
        if np.max(np.abs(neck_pos[t] - q_final)) >= margin_rad:
            return t + 1
    return 0


def _match_head_arrival(
    neck_pos: np.ndarray,
    neck_speed: np.ndarray,
    anchor: int,
    fps: float,
    params: GaltParams,
) -> tuple[Optional[int], bool]:
    """Find best head-arrival candidate in the window around the anchor.

    Two-stage detection:
      1. Identify the best settled interval (speed < v_sac_thresh for K_fix)
         within the window; prefer pre-anchor candidates, closest to anchor.
      2. Refine its start by walking back in joint-position space: the true
         arrival is the earliest timestep from which the head *stayed* close
         to its ultimate fixation direction (within arrival_margin_rad).

    Returns (t_head_arrival, is_pre_anchor) or (None, False) if no settled
    interval exists in the window.
    """
    lookback = int(round(params.lookback_s * fps))
    slack = int(round(params.forward_slack_s * fps))
    k_persist = max(1, int(round(params.K_fix_s * fps)))

    window_start = anchor - lookback
    window_end = anchor + slack + 1  # exclusive

    candidates = _find_settled_intervals(
        neck_speed, params.v_sac_thresh, k_persist, window_start, window_end
    )
    if not candidates:
        return None, False

    pre = [c for c in candidates if c <= anchor]
    best_settled = (min(pre, key=lambda c: abs(c - anchor))
                    if pre
                    else min(candidates, key=lambda c: abs(c - anchor)))

    refined = _refine_arrival_by_position(
        neck_pos, best_settled, params.arrival_margin_rad
    )
    # is_pre_anchor is recomputed on the refined time. Refinement can pull a
    # post-anchor settling back into pre-anchor territory (head was heading
    # to final direction before anchor but didn't fully stop pre-anchor).
    return refined, refined <= anchor


# ── Main entry point ─────────────────────────────────────────────────────────


def compute_galt(
    action: np.ndarray,
    fps: float,
    params: Optional[GaltParams] = None,
    layout: Optional[ActionLayout] = None,
) -> GaltResult:
    """Compute GALT_proprio for a single episode.

    Args:
        action: (T, 19) commanded-action trajectory.
        fps: sampling rate (Hz).
        params: GaltParams; None → defaults.
        layout: ActionLayout; None → TAVIS 19D.

    Returns:
        GaltResult with galt_s, reason, arm, and auxiliary timings.
    """
    if params is None:
        params = GaltParams()
    if layout is None:
        layout = ActionLayout()

    T = action.shape[0]
    if T < 10:
        return GaltResult(None, "too_short", None)

    # Neck angular speed (rad/s on joint space, L2 over 3 joints).
    neck_pos = action[:, list(layout.neck)]
    neck_speed = _velocity(neck_pos, fps)

    per_arm = {}
    arm_configs = {
        "left":  (layout.left_ee_xyz,  layout.left_gripper),
        "right": (layout.right_ee_xyz, layout.right_gripper),
    }

    results = {}
    for arm_name, (ee_idx, grip_idx) in arm_configs.items():
        rec = {}
        ee_speed = _velocity(action[:, list(ee_idx)], fps)

        events = _find_gripper_events(action[:, grip_idx])
        rec["n_gripper_events"] = len(events)
        if not events:
            rec["reason"] = "no_grasp"
            per_arm[arm_name] = rec
            continue

        anchor = events[-1]                                  # latest event
        rec["anchor"] = anchor

        min_stable_steps = max(1, int(round(params.min_stable_for_onset_s * fps)))
        t_hand_onset = _backward_search_hand_onset(
            ee_speed, anchor, params.v_hand_thresh, min_stable_steps
        )
        rec["t_hand_onset"] = t_hand_onset
        if t_hand_onset is None:
            rec["reason"] = "no_hand_onset"
            per_arm[arm_name] = rec
            continue

        t_head_arrival, is_pre = _match_head_arrival(
            neck_pos, neck_speed, anchor, fps, params
        )
        rec["t_head_arrival"] = t_head_arrival
        rec["is_pre_anchor"] = is_pre
        if t_head_arrival is None:
            rec["reason"] = "no_fixation"
            per_arm[arm_name] = rec
            continue

        galt_s = (anchor - t_head_arrival) / fps
        rec["galt_s"] = galt_s
        if galt_s < params.outlier_min_s:
            rec["reason"] = "outlier_low"
            per_arm[arm_name] = rec
            continue
        if galt_s > params.outlier_max_s:
            rec["reason"] = "outlier_high"
            per_arm[arm_name] = rec
            continue

        rec["reason"] = "ok" if is_pre else "ok_post_anchor"
        results[arm_name] = rec
        per_arm[arm_name] = rec

    # Enforce at most one valid arm
    if len(results) == 0:
        # Pick the most informative per-arm reason: prefer "no_fixation" >
        # "no_hand_onset" > "no_grasp" (deeper into pipeline = more informative).
        priority = {"no_fixation": 3, "no_hand_onset": 2, "outlier_low": 3,
                    "outlier_high": 3, "no_grasp": 1}
        reason = "no_grasp"
        if per_arm:
            reason = max(
                (r["reason"] for r in per_arm.values()),
                key=lambda r: priority.get(r, 0),
            )
        return GaltResult(None, reason, None, per_arm=per_arm)

    if len(results) == 2:
        return GaltResult(None, "ambiguous_arms", None, per_arm=per_arm)

    # Exactly one
    arm_name, rec = next(iter(results.items()))
    return GaltResult(
        galt_s=rec["galt_s"],
        reason=rec["reason"],
        arm=arm_name,
        t_head_arrival_s=rec["t_head_arrival"] / fps,
        t_hand_arrival_s=rec["anchor"] / fps,
        t_hand_onset_s=rec["t_hand_onset"] / fps,
        arm_reach_s=(rec["anchor"] - rec["t_hand_onset"]) / fps,
        per_arm=per_arm,
    )


# ── Tiny self-test on synthetic data ─────────────────────────────────────────

def _synth_episode(T=300, fps=60.0, head_arrival_s=1.0, hand_onset_s=1.4,
                   hand_arrival_s=2.0, arm="right"):
    """Build a synthetic episode with known event times."""
    t = np.arange(T) / fps
    act = np.zeros((T, 19), dtype=np.float32)

    # Neck: smoothly ramp angles during [0, head_arrival_s], then hold.
    mask = t < head_arrival_s
    ramp = np.linspace(0, 0.8, mask.sum())
    act[mask, 15] = ramp                         # pitch
    act[~mask, 15] = 0.8

    # Arm EE: hold at origin until hand_onset_s, then ramp to (0.5, -0.2, 0.3)
    ee_idx = [7, 8, 9] if arm == "right" else [0, 1, 2]
    grip_idx = 18 if arm == "right" else 17
    tgt = np.array([0.5, -0.2, 0.3])
    mid = (hand_onset_s <= t) & (t < hand_arrival_s)
    if mid.sum() > 0:
        ramp = np.linspace(0, 1, mid.sum())[:, None] * tgt[None, :]
        act[mid][:, :] = 0  # silence linter
        act[mid, ee_idx[0]] = ramp[:, 0]
        act[mid, ee_idx[1]] = ramp[:, 1]
        act[mid, ee_idx[2]] = ramp[:, 2]
    after = t >= hand_arrival_s
    act[after, ee_idx[0]] = tgt[0]
    act[after, ee_idx[1]] = tgt[1]
    act[after, ee_idx[2]] = tgt[2]

    # Gripper: open (-1) until hand_arrival_s, then close (+1).
    act[:, grip_idx] = -1.0
    act[after, grip_idx] = 1.0
    # Non-grasping arm: gripper always open.
    other_grip = 17 if arm == "right" else 18
    act[:, other_grip] = -1.0

    return act


if __name__ == "__main__":
    # Quick sanity test.
    act = _synth_episode(head_arrival_s=1.0, hand_onset_s=1.4, hand_arrival_s=2.0)
    res = compute_galt(act, fps=60.0)
    print(f"result: {res}")
    assert res.reason == "ok", f"expected ok, got {res.reason}"
    assert res.arm == "right"
    assert abs(res.galt_s - 1.0) < 0.1, f"galt off: {res.galt_s}"
    print(f"OK — GALT={res.galt_s:.3f}s (expected ~1.0s)")
