# Evaluation modes

`scripts/eval_benchmark.py` evaluates each (robot × task) cell under
one or more *eval modes*. Each mode tags the rollout JSON distinctly
so id and ood episodes never mix in downstream analysis.

The three modes used in the paper:

| Mode             | Object placement     | Robot reset pose                                                |
|------------------|----------------------|-----------------------------------------------------------------|
| `id`             | training range       | default                                                         |
| `ood_spatial`    | wider than training  | default                                                         |
| `ood_init_pose`  | training range       | Gaussian: σ=10 cm on EEF positions, σ≈10° on neck pitch and yaw |

`id` and `ood_spatial` are implemented task-side via the `VARIANTS`
preset (the `task_variant` string is passed to the task constructor
and selects a preset of object-placement bounds).

`ood_init_pose` is implemented env-side via
`tavis/wrappers/init_pose_wrapper.py`. `eval_benchmark.py` wires it
in automatically when `--eval-modes ood_init_pose` is passed; the
task-side code still uses the `id` `VARIANTS` preset.

> In the current pipeline `ood_init_pose` is **not** combinable with
> `ood_spatial` — the three modes are separate, not orthogonal axes
> that compose. However, since `ood_init_pose` simply adds an initial 
> pose wrapper to the chosen env, it is very simple to modify `eval_benchmark.py`
> to combine it with `ood_spatial`.

## Adding a new eval mode

**Task-side** (different object distribution): add a key under
`VARIANTS` in the task file and either pass it via `--eval-modes`
or extend `DEFAULT_EVAL_MODES` in `tavis/benchmark/suites.py`.

```python
VARIANTS = {
    "id":          {...},
    "ood_spatial": {...},
    "ood_clutter": {              # new mode
        "pose_range":     {...},
        "n_distractors":  8,
    },
}
```

**Env-side** (start state, scene perturbation, …): wire the new
behaviour in `eval_benchmark.py` next to the `ood_init_pose` branch
and use a distinct eval-mode string so the rollout JSON is tagged
separately.
