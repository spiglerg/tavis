# Adding a new task

A TAVIS task is a subclass of `isaaclab_arena.tasks.task_base.TaskBase`
that knows how to:

* lay out the scene (table, objects, distractors, occluders, …),
* sample initial object placements per episode,
* emit a language prompt naming the target,
* score the trajectory (`success` and `done` terms).

The eight reference tasks live in `tavis/tasks/`. Each task file is
~250–500 lines, and most of that is the scene description; the
control flow is the same across all tasks and is shared via
`tasks/_common.py` (table cfg, YCB object factory, common termination
helpers).

## Convention every task follows

Two class-level pieces are required:

1. A `VARIANTS` class attribute (a dict-of-dicts) with at least the
   `"id"` and `"ood_spatial"` keys. Standard fields:

   ```python
   VARIANTS = {
       "id": {
           "pose_range":     {"x": (...), "y": (...), "z": (...), "yaw": (...)},
           "min_separation": 0.10,
           # task-specific keys (e.g. probe_color, n_distractors) go here
       },
       "ood_spatial": {
           "pose_range":     {"x": (...wider...), "y": (...wider...)},
           "min_separation": 0.05,
       },
   }
   ```

2. A constructor that accepts the standard signature:

   ```python
   def __init__(
       self,
       task_variant: str = "id",
       variant_overrides: dict | None = None,
       episode_length_s: float = 100.0,
   ):
   ```

   `task_variant` selects a preset, and `variant_overrides` patches
   individual keys at instantiation time (only the supplied keys are
   overridden).

## Object placement, the shared way

Use `_common.make_object_cfg(name)` to spawn a YCB object by canonical
name (`soup_can`, `cracker_box`, …). Use the rejection sampler in
`_common` for collision-free initial placements; if you need a new
sampler (e.g. on a shelf rather than a table), keep it in the task
file but mirror the API.

## Registering the task

Add the class to `tavis/tasks/__init__.py`:

```python
from .my_task import MyTask

__all__ = [..., "MyTask"]

TASK_MAP = {
    ...,
    "my_task": MyTask,
}
```

Then decide whether the task belongs in an existing suite or a new
one. Suite membership lives in `tavis/benchmark/suites.py`:

```python
SUITES = {
    "tavis-head":  ["clutter_pick_lift", ..., "my_task"],
    "tavis-hands": [...],
}
```
