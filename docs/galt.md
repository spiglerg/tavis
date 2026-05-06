# GALT: Gaze–Action Lead Time

GALT is a kinematic metric that quantifies *anticipatory gaze* in an
imitation-learning policy: how far in advance the head settles on the
target before the hand arrives at it. In its current form, it is computed 
from a single episode's commanded-action trajectory — no simulator state, no
target-object knowledge, no observations required.

```
GALT = t_hand_arrival − t_head_arrival              (seconds)
```

The detector is implemented in `tavis/eval/galt.py` and is wired into
`scripts/eval_benchmark.py` so every benchmark rollout produces a GALT
estimate alongside the success rate.

## Why "proprioceptive" GALT?

Most analyses of anticipatory gaze use eye-tracking or scene-relative
fixation targets — both unavailable in a closed-loop IL setting where
the policy outputs commanded actions. We instead define arrival
events purely from the action stream:

* `t_head_arrival` — the head joints stop moving and stay within an
  L∞ tolerance of their final commanded value for ≥ K frames.
* `t_hand_arrival` — anchored on the *latest* gripper state change
  (close for a grasp, open for a place); the hand arrival is then
  the start of the rest interval that immediately precedes that
  gripper event.
* `t_hand_onset` — the latest stable-to-motion transition of the
  end-effector before the anchor (used to compute `arm_reach_s`).

This makes GALT robot-portable, demonstration-portable, and policy-
agnostic.

## The 19-D canonical action layout

GALT consumes the commanded-action trajectory as an `(T, 19)` array.
The default index mapping (`ActionLayout` in `tavis/eval/galt.py`) is:

| indices       | meaning                            |
|---------------|------------------------------------|
| `0–2`         | left arm IK target — Cartesian xyz |
| `3–6`         | left arm IK target — quaternion wxyz |
| `7–9`         | right arm IK target — xyz          |
| `10–13`       | right arm IK target — quaternion wxyz |
| `14–16`       | head joints (roll, pitch, yaw)     |
| `17`          | left gripper scalar in `[-1, 1]`   |
| `18`          | right gripper scalar in `[-1, 1]`  |

GR1T2 and Reachy2 both expose this 19-D layout natively, so policies
share a single action space across robots.

## Porting GALT to a new robot

The detector only inspects four channel groups: left/right EE position,
head joints, and left/right gripper scalar. To run on a new robot,
override `ActionLayout` with that robot's action indices:

```python
from tavis.eval.galt import compute_galt, ActionLayout, GaltParams

layout = ActionLayout(
    left_ee_xyz=(2, 3, 4),
    right_ee_xyz=(11, 12, 13),
    neck=(0, 1, 18),
    left_gripper=10,
    right_gripper=19,
)

result = compute_galt(actions, fps=20.0, layout=layout)
print(result.galt_s, result.reason)
```

If the robot has a different DoF layout (e.g., a 7-DoF neck instead of
3, or a multi-finger hand instead of a single gripper scalar), pick a
"head pose" channel set whose joint-velocity norm captures fixation
onset/offset, and a single "gripper" channel that crosses zero on
grasp/release. The remaining detector logic is unchanged.

## Tuning thresholds

`GaltParams` exposes the velocity floors and arrival-persistence
window. Defaults come from physical noise-floor analysis on TAVIS
teleoperation data and are reported in the paper:

* `v_hand_thresh = 0.05 m/s`
* `v_sac_thresh = 0.10 rad/s`
* `K_fix_s = 0.080 s` (arrival persistence)
* `arrival_margin_rad = 0.05` (≈ 2.9°)

Re-tune on your own data only if the noise floor differs substantially 
(e.g., a much faster control rate, a much smaller robot).

## Reason codes

Every result carries a reason string that tells you why a GALT could
or could not be returned: `ok`, `ok_post_anchor`, `no_grasp`,
`no_hand_onset`, `no_fixation`, `outlier_low`, `outlier_high`,
`ambiguous_arms`. Aggregate analyses can report the per-code
breakdown alongside summary statistics.

## Multi-event extension

`compute_galt` returns a single GALT, anchored on the *latest*
gripper event. Long-horizon or multi-grasp tasks need a thin
wrapper loop iterating over every gripper event as anchor. The
single-event core is unchanged; only the loop is missing. See the
module docstring at `tavis/eval/galt.py`.
