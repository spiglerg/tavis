# Data collection (VR teleoperation)

TAVIS demonstrations are collected in simulation with a Meta Quest
headset. The user sees the robot's egocentric (head-mounted) RGB
stream live in the headset; left-controller pose drives the left
arm IK target, right-controller pose drives the right arm IK target,
and the operator's head pose drives the robot's head joints. Both
parallel grippers are bound to the corresponding controller grip
trigger.

The host sends video frames and receives pose data over TCP; the
protocol is JSON, documented in `quest_app/README.md`.

## Hardware

* Meta Quest 2 / 3 with developer mode enabled.
* USB-C cable for ADB.
* A workstation with a recent ray-tracing-capable NVIDIA GPU; data
  collection uses RTX rendering. (Training itself does not need RTX.)

## Software set-up

1. **Install IsaacSim and IsaacLab** as documented in the top-level
   `README.md`. Then `pip install -e ".[train]"` from the repo root.
2. **Pre-built APK.** The `quest_teleop.apk` is published as a
   release artifact. Download and install:

   ```bash
   adb install -r quest_teleop.apk
   ```

3. **Forward the streaming ports** every time you reconnect the
   Quest:

   ```bash
   adb devices
   adb forward tcp:9500 tcp:9500     # video stream (host → headset)
   adb forward tcp:9501 tcp:9501     # pose data    (headset → host)
   ```

4. **Launch the headset app:**

   ```bash
   adb shell am force-stop com.airlab.quest_teleop
   adb shell monkey -p com.airlab.quest_teleop \
       -c android.intent.category.LAUNCHER 1
   ```

   > **Bundle-ID note for the pre-built APK.** Depending on the
   > build revision, the APK published under Releases may carry a
   > different bundle ID than the one shown above. If `adb monkey`
   > reports `package not found`, run
   > `aapt dump badging quest_teleop.apk` (Android SDK build-tools)
   > to read the actual ID and substitute it into the launch command,
   > or rebuild a clean APK from the included Unity project — see
   > "Building the Quest APK from source" below. The full Unity source
   > is shipped in this repository for exactly that purpose.

5. **Run the host script:**

   ```bash
   python scripts/teleop_main.py
   ```

   Edit the file to pick the robot, task, and dataset name (the top
   half of `teleop_main.py` is a switch table of configurations).

## Controller mapping

| Control                                       | Effect                                                                            |
|-----------------------------------------------|-----------------------------------------------------------------------------------|
| Y on left controller                          | Toggle hand tracking on / off                                                     |
| X on left controller (short press, < 0.5 s)   | Recalibrate table height (place controller flat on the table, then short-press X) |
| X on left controller (long press, ≥ 0.5 s)    | Toggle wrist-camera picture-in-picture overlay on the head-cam view               |
| Right grip                                    | Start / stop recording the current episode                                        |
| A on right controller                         | Save the current episode                                                          |
| B on right controller                         | Discard the current episode                                                       |
| Left grip                                     | Reset the environment                                                             |

## Recording

Episodes are written to a LeRobot v3.0 dataset on disk. The default
location is `datasets/<dataset_name>/`. Saving an episode is
incremental; the full dataset is finalised on shutdown.

> **Critical:** stop teleoperation with `Ctrl-C` in the terminal
> running `teleop_main.py`. The signal handler triggers
> `dataset.finalize()`, which closes video files and writes the
> manifest. Killing the process any other way (`kill -9`, closing
> the terminal, power-off) will corrupt the dataset.

## Building the Quest APK from source

The Unity project under `quest_app/source/Quest_Teleop/` is included
for users who want to modify the controller mapping or the streaming
protocol. Open with Unity 2022.3 LTS or later, switch to the Android
build target, install the Meta XR SDK, and build a development APK.
The bundle ID is `com.airlab.quest_teleop` — keep it stable
or update the `adb shell monkey` command above accordingly.

## Sanity checks

If the headset shows a black screen instead of the simulation:

* Verify both ports are forwarded (`adb forward --list`).
* Check that the host script logs incoming controller frames
  (`debug=True` in `teleop_main.py`).
* On a slow workstation, drop `robot_ctrl_rate` (set debug=True
  at the end of the file to print expected and real per-step time
  to make sure your workstation is able to track teleop at the
  right speed).
