# Adding a new robot

TAVIS decouples robots from tasks. A robot is an `Embodiment` —
a subclass of `isaaclab_arena.embodiments.embodiment_base.EmbodimentBase`
that bundles an articulation cfg, an action space, a camera config,
and a teleop config into a single object. The same task code runs
unchanged across embodiments.

The two reference embodiments live in `tavis/robots/`:

* `gr1t2.py` — Fourier GR1T2, 7-DoF arms, Robotiq 2F-85 grippers.
* `reachy2.py` — Pollen Reachy2, 7-DoF arms, single-DoF parallel grippers.

Both expose the same 19-D unified action layout (see `docs/galt.md`),
so the action space and the GALT metric are robot-agnostic.
Observation states are not — joint order and dimension differ between
robots — so a policy trained on one robot's state cannot be re-used
on another without retraining.

## Required pieces of a new embodiment

1. **USD / articulation file.** During development, place the USD
   tree under `tavis/assets/robots/<your_robot>/` locally. For
   public release, upload the same tree to the Hugging Face asset
   repo `tavis-benchmark/tavis-assets`; `tavis/download_assets.py`
   will fetch it on first env construction.

2. **Subclass `EmbodimentBase`.** Mirror the structure of
   `tavis/robots/gr1t2.py`. The class needs to define:
   * `articulation_cfg` — IsaacLab `ArticulationCfg` pointing at the USD.
   * `actions` — a 19-D action layout. Use `NullSpaceIKActionCfg`
     (from `tavis/controllers`) for the arm IK targets, plus a head
     joint-position action and a gripper action term.
   * `observations` — joint state, optional cameras (head, left/right
     wrist, fixed scene cam).
   * `camera_config` — the head-mounted camera (egocentric).
   * `teleop_config` — rest poses, action-index mapping for the Quest
     teleop loop (`tavis/teleop/quest_teleop.py`).
   * `neck_state_indices` and `neck_action_indices` — class attributes
     used by the fixed-vs-head-camera ablation. **Also** add the same
     values to the `NECK_INDICES` dict at the top of
     `scripts/train_policy.py` (it's duplicated there to avoid
     importing the embodiment files on training-only clusters that
     don't have IsaacLab installed).

3. **Canonical-frame offset.** TAVIS uses a single hip-level canonical
   frame for IK targets. Each robot reports a translation between its
   root link and the canonical origin via `embodiment.canonical_frame_offset`;
   `tavis/wrappers/canonical_frame_wrapper.py` applies the conversion
   before commands reach the IK controller. GR1T2's hips are already
   at canonical height (zero offset); Reachy2 sits taller, so it
   declares a non-zero z offset.

4. **Register in `tavis/robots/__init__.py`** by adding an entry to
   `ROBOT_MAP`:

   ```python
   ROBOT_MAP = {
       "gr1t2": GR1T2Embodiment,
       "reachy2": Reachy2Embodiment,
       "myrobot": MyRobotEmbodiment,
   }
   ```

   `eval_benchmark.py --robot myrobot` and the suite scripts will pick
   it up automatically.

## Action-space convention

For best use in TAVIS, new robots should use absolute IK tracking of 
the arms, and head yaw/pitch/roll angles, together with a single gripper 
command per arm.

## Verifying the integration

A minimal smoke test is to construct the embodiment, place it in an
empty scene, and step the env once:

```python
from tavis import make_tavis_env
from tavis.robots import MyRobotEmbodiment
from tavis.tasks import ClutterPickLiftTask

embodiment = MyRobotEmbodiment(enable_cameras=True)
env = make_tavis_env(embodiment=embodiment, task=ClutterPickLiftTask())
obs, _ = env.reset()
print(obs.keys())
```

If the IK controller, cameras, and observation manager all initialise
cleanly, the embodiment is ready for teleoperation and policy training.

It is helpful to also run your robot on `eval_policy.py` (non-headless mode)
to verify camera placement, and `teleop_main.py` to make sure teleoperation maps
correctly (especially to fine-tune the canonical frame).
