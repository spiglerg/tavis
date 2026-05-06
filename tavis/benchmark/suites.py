"""TAVIS benchmark suite definitions.

A suite is a named list of task names. Robot, camera mode, and eval mode
are orthogonal axes specified at evaluation time.
"""

from tavis.tasks import TASK_MAP
from tavis.robots import ROBOT_MAP  # noqa: F401 — re-export for convenience

SUITES = {
    "tavis-head": [
        "clutter_pick_lift",
        "clutter_pick_cube",
        "conditional_pick",
        "wait_then_act",
        "multi_shelf_scan",
    ],
    "tavis-hands": [
        "peeking_box",
        "occluded_reach",
        "blocked_clutter_pick_cube",
    ],
    # Future: "tavis-social": [...]
}

DEFAULT_EVAL_MODES = {
    "tavis-head": ["id", "ood_spatial"],
    "tavis-hands": ["id", "ood_spatial"],
}


def build_eval_combos(suite=None, tasks=None, eval_modes=None):
    """Build list of (task_name, task_class, eval_mode) combos.

    Args:
        suite: Suite name (e.g., "tavis-head"). Selects all tasks in suite.
        tasks: List of task names. Filters suite, or used standalone.
        eval_modes: List of eval mode strings. Defaults to suite's default or ["id"].

    Returns:
        List of (task_name, task_class, eval_mode) tuples.
    """
    if suite:
        task_names = list(SUITES[suite])
        if tasks:
            task_names = [t for t in task_names if t in tasks]
    elif tasks:
        task_names = list(tasks)
    else:
        raise ValueError("Specify --suite or --tasks")

    task_list = [(name, TASK_MAP[name]) for name in task_names]

    if eval_modes is None:
        if suite:
            eval_modes = DEFAULT_EVAL_MODES.get(suite, ["id"])
        else:
            eval_modes = ["id"]

    return [(name, cls, mode) for name, cls in task_list for mode in eval_modes]
