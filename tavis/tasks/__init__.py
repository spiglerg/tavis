"""TAVIS task library.

Each task class follows these conventions:

``VARIANTS`` class attribute (dict of dicts)
    Presets keyed by task_variant string.  Minimum required keys:

        pose_range     : dict with x/y/z/yaw range tuples for object placement.
        min_separation : float, minimum centre-to-centre distance (m).

    Add further keys as task complexity grows; document them in the class
    docstring.  Standard variants: ``"id"``, ``"ood_spatial"``.

Constructor signature::

    def __init__(
        self,
        task_variant: str = "id",
        variant_overrides: dict | None = None,
        episode_length_s: float = 100.0,
    ):

``task_variant`` selects the preset; ``variant_overrides`` patches individual
keys (only specified keys are overridden, others keep the preset value).
"""

from .blocked_clutter_pick_cube import BlockedClutterPickCubeTask
from .clutter_pick_cube import ClutterPickCubeTask
from .clutter_pick_lift import ClutterPickLiftTask
from .conditional_pick import ConditionalPickTask
from .multi_shelf_scan import MultiShelfScanTask
from .occluded_reach import OccludedReachTask
from .peeking_box import PeekingBoxTask
from .wait_then_act import WaitThenActTask

__all__ = [
    "BlockedClutterPickCubeTask",
    "ClutterPickCubeTask",
    "ClutterPickLiftTask",
    "ConditionalPickTask",
    "MultiShelfScanTask",
    "OccludedReachTask",
    "PeekingBoxTask",
    "WaitThenActTask",
    "TASK_MAP",
]

TASK_MAP = {
    # TAVIS-HEAD
    "clutter_pick_lift": ClutterPickLiftTask,
    "clutter_pick_cube": ClutterPickCubeTask,
    "conditional_pick": ConditionalPickTask,
    "wait_then_act": WaitThenActTask,
    "multi_shelf_scan": MultiShelfScanTask,
    # TAVIS-HANDS
    "peeking_box": PeekingBoxTask,
    "occluded_reach": OccludedReachTask,
    "blocked_clutter_pick_cube": BlockedClutterPickCubeTask,
}
