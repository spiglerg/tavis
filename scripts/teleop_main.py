#!/usr/bin/env python3
"""
Quest Teleoperation Main Script


First time running:
  Install the APK on your Meta Quest device:
  $ adb install -r quest_app/builds/quest_teleop.apk

Before each session, run the following when you connect the Quest to the computer:
    $ adb devices
    $ adb forward tcp:9500 tcp:9500 && adb forward tcp:9501 tcp:9501

Open the Quest app:
    $ adb shell am force-stop com.airlab.quest_teleop
    $ adb shell monkey -p com.airlab.quest_teleop -c android.intent.category.LAUNCHER 1

When running from a new computer, you may want to run with debug=True and different robot_ctrl_rate,
to make sure that the system is capable of keeping the correct loop time.

Watch out: you better use headless=True unless you are debugging the code.

On a 4090+16-core desktop: 60Hz (~17ms per step) in headless mode; GUI mode is 30Hz (~33ms per step).

VERY IMPORTANT: Kill the app to stop recording (preferably with Ctrl+C in the terminal running this
script, to ensure proper dataset finalization).
"""

# Initialize IsaacSim
#headless = False
headless = True

# Teleop runs at 60Hz for high-fidelity demos. Training/eval default to 20Hz
# (via train_policy.py --downsample-factor=3), matching LIBERO/RoboMimic. The extra
# 60Hz resolution gives the data loader free temporal augmentation (every original
# frame is a valid anchor for the 20Hz subsequence). If your computer can't keep up
# with 60Hz teleop (debug=True will show timing warnings), it is OK to drop this
# to 20, but then train with --downsample-factor=1 for matched rates.
robot_ctrl_rate = 60


from isaaclab.app import AppLauncher 
app_launcher = AppLauncher(num_envs=1, enable_cameras=True, headless=headless)
simulation_app = app_launcher.app

####


import signal

from tavis import make_tavis_env

from tavis.robots import GR1T2Embodiment, Reachy2Embodiment

from tavis.tasks import ClutterPickLiftTask, ConditionalPickTask, ClutterPickCubeTask, WaitThenActTask, MultiShelfScanTask, \
                                 PeekingBoxTask, OccludedReachTask, BlockedClutterPickCubeTask


from tavis.wrappers import GenericTorsoExperimentWrapper

from tavis.teleop import QuestTeleopRun



# Create embodiment and task - GR1
embodiment = GR1T2Embodiment(enable_cameras=True)

# fixedcam-teleop ablation
#embodiment.teleop_config['fpv_camera_obs_key'] = 'fixed_camera_rgb'

task = ConditionalPickTask() #(task_variant="ood_spatial")
dataset_name = 'gr1t2_conditional_pick'
dataset_description = "GR1T2 robot must look at the card; if red, look and then lift the left object; if green, look and then lift the right object.  Tests gaze for information gathering." #"Ablation: teleoperation from fixed camera. To train fixedcam, discarding neck motion."

#task = WaitThenActTask() #(task_variant="ood_spatial")
#dataset_name = 'gr1t2_wait_then_act'
#dataset_description = "GR1T2 robot must look at the light until it turns green. Then, gaze at the object and pick it up."

#task = ClutterPickCubeTask() #(task_variant="ood_spatial")
#dataset_name = 'gr1t2_clutter_pick_cube'
#dataset_description = "GR1T2 robot must look at the cube and then pick it up, ignoring the distractors."

#task = ClutterPickLiftTask() #(task_variant="ood_spatial")
#dataset_name = 'gr1t2_clutter_pick_lift'
#dataset_description = "GR1T2 robot must look for the target object and pick it up."

#task = MultiShelfScanTask() #(task_variant="ood_spatial")
#dataset_name = 'gr1t2_multi_shelf_scan'
#dataset_description = "GR1T2 robot must scan the shelves top to bottom, locating the target object. The object is then taken from the shelf."


# TAVIS-HANDS
#task = PeekingBoxTask() #(task_variant="ood_spatial")
#dataset_name = 'gr1t2_peeking_box' 
#dataset_description = "GR1T2 robot must use its wrist cameras to find which side (left or right) of the box the object is in, and then pick it up. Tests use of wrist cameras for hidden object localization."

#task = OccludedReachTask() #(task_variant="ood_spatial")
#dataset_name = 'gr1t2_occluded_reach'
#dataset_description = "GR1T2 robot must reach to the target object, which is partially occluded by a barrier. Tests reaching under partial occlusion."

#task = BlockedClutterPickCubeTask() #(task_variant="ood_spatial")
#dataset_name = 'gr1t2_blocked_clutter_pick_cube'
#dataset_description = "GR1T2 robot must pick up the target cube, which is surrounded by distractor cubes, while the head camera is disabled. Tests reaching and grasping under heavy clutter, while relying exclusively on wrist cameras."



# Create embodiment and task - Reachy
#embodiment = Reachy2Embodiment(enable_cameras=True)

#task = ConditionalPickTask() #(task_variant="ood_spatial")
#dataset_name = 'reachy2_conditional_pick'
#dataset_description = "Reachy2 robot must look at the card; if red, look and then lift the left object; if green, look and then lift the right object.  Tests gaze for information gathering."

#task = WaitThenActTask() #(task_variant="ood_spatial")
#dataset_name = 'reachy2_wait_then_act'
#dataset_description = "Reachy2 robot must look at the light until it turns green. Then, gaze at the object and pick it up."

#task = ClutterPickCubeTask() #(task_variant="ood_spatial")
#dataset_name = 'reachy2_clutter_pick_cube'
#dataset_description = "Reachy2 robot must look at the cube and then pick it up, ignoring the distractors."

#task = ClutterPickLiftTask() #(task_variant="ood_spatial")
#dataset_name = 'reachy2_clutter_pick_lift'
#dataset_description = "Reachy2 robot must look for the target object and pick it up."

#task = MultiShelfScanTask() #(task_variant="ood_spatial")
#dataset_name = 'reachy2_multi_shelf_scan'
#dataset_description = "Reachy2 robot must scan the shelves top to bottom, locating the target object. The object is then taken from the shelf."


# TAVIS-HANDS
#task = PeekingBoxTask() #(task_variant="ood_spatial")
#dataset_name = 'reachy2_peeking_box' 
#dataset_description = "Reachy2 robot must use its wrist cameras to find which side (left or right) of the box the object is in, and then pick it up. Tests use of wrist cameras for hidden object localization."

#task = OccludedReachTask() #(task_variant="ood_spatial")
#dataset_name = 'reachy2_occluded_reach'
#dataset_description = "Reachy2 robot must reach to the target object, which is partially occluded by a barrier. Tests reaching under partial occlusion."

#task = BlockedClutterPickCubeTask() #(task_variant="ood_spatial")
#dataset_name = 'reachy2_blocked_clutter_pick_cube'
#dataset_description = "Reachy2 robot must pick up the target cube, which is surrounded by distractor cubes, while the head camera is disabled. Tests reaching and grasping under heavy clutter, while relying exclusively on wrist cameras."




# Create environment via Arena builder
isaaclab_env = make_tavis_env(
    embodiment=embodiment,
    task=task,
    robot_ctrl_rate=robot_ctrl_rate,
)


# We then wrap the isaaclab environment in a Quest teleoperation wrapper to handle the Quest device controls and experiment logic.
# Robot-specific parameters (action indices, EEF body names, rest poses, etc.)
# are read from embodiment.teleop_config.
env = GenericTorsoExperimentWrapper(isaaclab_env,
                                    embodiment=embodiment,
                                    dataset_kwargs = {
                                        'dataset_repo_id' : "tavis-benchmark/"+dataset_name,
                                        'dataset_root_dir' : "datasets/"+dataset_name,
                                        'dataset_description' : dataset_description,
                                        'dataset_fps' : robot_ctrl_rate,
                                        'dataset_obs_keys_to_record' : {'head_camera_rgb':'OBS_HEAD', 'left_wrist_camera_rgb':'OBS_WRIST_LEFT', 'right_wrist_camera_rgb':'OBS_WRIST_RIGHT', 'fixed_camera_rgb':'OBS_FIXED', 'robot_joint_pos':'state', 'left_eef_pos':None, 'left_eef_quat':None, 'right_eef_pos':None, 'right_eef_quat':None},
                                    })


# A bit hacky, but we need to make sure we intercept ctrl+c to cleanly close the dataset and Isaac Sim
original_sigint_handler = signal.getsignal(signal.SIGINT)
def signal_handler(sig, frame):
    global env # Need global to access module-level variables
    print("\nCaught Ctrl+C - cleaning up...")

    if hasattr(env, 'dataset'):
        print("Finalizing dataset before exit...")
        env.dataset.finalize()

    print("Closing Isaac Sim...")
    if original_sigint_handler and callable(original_sigint_handler):
        print('close isaac orig')
        original_sigint_handler(sig, frame)
    else:
        print('sys exit')
        import sys
        sys.exit(0)


import signal
signal.signal(signal.SIGINT, signal_handler)

# Finally, we can run the Quest teleoperation main function with the environment we created.
QuestTeleopRun(env,
                robot_ctrl_rate=robot_ctrl_rate,
                video_port=9500,
                pose_port=9501,
                frame_width=640,
                frame_height=480,
                jpeg_quality=80,
                debug=False,
                )
