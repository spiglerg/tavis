# Quest Teleoperation App

This directory contains the Meta Quest VR application for teleoperation.

## Pre-built APK

The pre-built APK is available under Releases (`quest_teleop.apk`).

### Installation

1. Connect your Meta Quest to your computer via USB
2. Enable developer mode on your Quest
3. Install the APK:

```bash
adb install -r quest_teleop.apk
```

### Running

Before each teleoperation session:

```bash
# Check device connection
adb devices

# Forward required ports
adb forward tcp:9500 tcp:9500
adb forward tcp:9501 tcp:9501

# Launch the app
adb shell am force-stop com.airlab.quest_teleop
adb shell monkey -p com.airlab.quest_teleop -c android.intent.category.LAUNCHER 1
```

Then run the teleoperation script on your computer:

```bash
python scripts/teleop_main.py
```

## Source Code

The Unity project source is in `source/Quest_Teleop/`.

### Requirements for Building

- Unity 2022.3 LTS or later
- Meta Quest development SDK
- Android Build Support module


## Communication Protocol

The app communicates with the host computer via TCP:

- **Port 9500**: Video stream (JPEG frames from simulation)
- **Port 9501**: Pose data (JSON with head and controller poses)

### Pose Data Format

```json
{
  "head": {
    "pos_xyz": [x, y, z],
    "quat_wxyz": [w, x, y, z]
  },
  "leftController": {
    "pos_xyz": [x, y, z],
    "quat_wxyz": [w, x, y, z],
    "trigger": 0.0-1.0,
    "grip": 0.0-1.0,
    "X": true/false,
    "Y": true/false
  },
  "rightController": {
    "pos_xyz": [x, y, z],
    "quat_wxyz": [w, x, y, z],
    "trigger": 0.0-1.0,
    "grip": 0.0-1.0,
    "A": true/false,
    "B": true/false
  },
  "timestamp": 1234567890
}
```

Coordinate system (Unity):
- X: positive right
- Y: positive up
- Z: positive forward
