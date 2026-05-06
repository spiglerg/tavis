import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from tavis import math_utils
from tavis.wrappers.experiment_wrapper import AbstractExperimentWrapper

# Default Quest controller orientations (neutral pose, thumbsticks up).
# These are Quest-specific, not robot-specific, so they live here.
_DEFAULT_LEFT_CONTROLLER_ZERO_R = R.from_quat([0.0386093, -0.62003238, -0.01515786, 0.78347904])
_DEFAULT_RIGHT_CONTROLLER_ZERO_R = R.from_quat([-0.11884408, -0.56864608, -0.1144551, 0.8058646])
_DEFAULT_QUEST_XYZ_SCALE = 1.3


class GenericTorsoExperimentWrapper(AbstractExperimentWrapper):
    """
    Generic experiment wrapper with support for fixed robot torsos with moving neck (its joints can be set to None to
     disable neck control), and grippers with 2 arms.

    Robot-specific parameters (action indices, EEF body names, rest poses, z offset,
    etc.) are read from ``embodiment.teleop_config``.  Any key can be overridden via
    ``**overrides`` for quick experiments.

    Basic interface implemented:
        - On start, by default, controller movement won't move the arms (though head movement and grippers are enabled).
          Press the left 'Y' button to start tracking (after positioning your hands to roughly match the robot's, controllers' thumbstick up); press again to stop tracking.

        - 'X' button (short press < 0.5s): put your left controller close to the table surface and
          short-press 'X' to recompute the calibration z offset (triggers on release).
        - 'X' button (long press >= 0.5s): toggles wrist camera picture-in-picture overlay on the
          head camera feed (left/right wrist thumbnails in the bottom corners).

    # TODO: add support for simple moving bases on wheels (real or virtual)
    """
    def __init__(self,
                 env,
                 embodiment,
                 dataset_kwargs: dict,
                 left_controller_zero_R=None,
                 right_controller_zero_R=None,
                 quest_xyz_scale_factor=None,
                 **overrides):
        """
        Args:
            env: The IsaacLab environment to wrap.
            embodiment: An EmbodimentBase instance whose ``teleop_config`` dict
                provides robot-specific parameters.  Required keys:
                  neck_action_indices, left_arm_ik_action_index,
                  right_arm_ik_action_index, left_gripper_action_index,
                  right_gripper_action_index, left_arm_eef_body_name,
                  right_arm_eef_body_name, left_arm_position_zero,
                  right_arm_position_zero, left_arm_pose_zero_quat_xyzw,
                  right_arm_pose_zero_quat_xyzw, default_z_offset,
                  fpv_camera_obs_key
            dataset_kwargs: Keyword arguments forwarded to the dataset recorder.
            left_controller_zero_R / right_controller_zero_R: Optional scipy
                Rotation overrides for the Quest controller neutral orientation.
            quest_xyz_scale_factor: Optional float override for the Quest
                position scaling factor.
            **overrides: Any key present in ``teleop_config`` can be overridden
                here for quick experiments.
        """
        if not hasattr(embodiment, 'teleop_config'):
            raise ValueError(
                f"Embodiment {type(embodiment).__name__} does not define a "
                f"'teleop_config' dict.  GenericTorsoExperimentWrapper requires "
                f"teleop_config to be set on the embodiment."
            )

        # Merge: embodiment defaults <- explicit overrides
        cfg = {**embodiment.teleop_config, **overrides}

        left_arm_ik_action_index = cfg["left_arm_ik_action_index"]
        right_arm_ik_action_index = cfg["right_arm_ik_action_index"]

        # Compute quaternion indices for 6D rotation conversion in dataset
        # Quaternions start at position index + 3 (after xyz position)
        action_quat_indices = [left_arm_ik_action_index + 3, right_arm_ik_action_index + 3]

        super().__init__(env, action_quat_indices=action_quat_indices, **dataset_kwargs)

        self.last_frame = None

        self.neck_action_indices = cfg["neck_action_indices"]

        self.left_arm_ik_action_index = left_arm_ik_action_index
        self.right_arm_ik_action_index = right_arm_ik_action_index
        self.left_gripper_action_index = cfg["left_gripper_action_index"]
        self.right_gripper_action_index = cfg["right_gripper_action_index"]
        self.fpv_camera_obs_key = cfg["fpv_camera_obs_key"]

        self.left_arm_position_zero = cfg["left_arm_position_zero"]
        self.right_arm_position_zero = cfg["right_arm_position_zero"]

        # default arm poses in the environment, in scipy Rotation format
        self.left_arm_pose_zero_R = R.from_quat(cfg["left_arm_pose_zero_quat_xyzw"])
        self.right_arm_pose_zero_R = R.from_quat(cfg["right_arm_pose_zero_quat_xyzw"])

        # Quest controller defaults
        self.quest_xyz_scale = quest_xyz_scale_factor if quest_xyz_scale_factor is not None else _DEFAULT_QUEST_XYZ_SCALE

        self.z_offset = cfg["default_z_offset"]

        self.left_arm_eef_body_name = cfg["left_arm_eef_body_name"]
        self.right_arm_eef_body_name = cfg["right_arm_eef_body_name"]

        # Canonical frame z offset (for table-height calibration)
        self._canonical_z_offset = getattr(embodiment, 'canonical_frame_offset', (0, 0, 0))[2]

        self.left_controller_zero_R = left_controller_zero_R if left_controller_zero_R is not None else _DEFAULT_LEFT_CONTROLLER_ZERO_R
        self.right_controller_zero_R = right_controller_zero_R if right_controller_zero_R is not None else _DEFAULT_RIGHT_CONTROLLER_ZERO_R

        self.controllers_tracking_active = False  # whether arm pose commands will be sent, or whether to use the default configuration
        self.last_y_pressed_status = False

        # X button long-press state (short < 0.5s = table recalib, long >= 0.5s = wrist cam toggle)
        self._x_press_start: float | None = None
        self._x_long_press_triggered = False
        self._show_wrist_cams = False
        self._X_LONG_PRESS_S = 0.5  # seconds threshold

        self.text_to_display = (
            "Place your hands to roughly match the robot's, press 'Y' to start tracking.\n"
            "Right 'grip' start/stop recording.\n"
            "'A' save episode\n"
            "'B' discard episode.\n"
            "'X' short press: recalibrate table height | long press: toggle wrist cams"
        )

    def save_fpv_frame(self, last_obs):
        frame = last_obs['policy'][self.fpv_camera_obs_key][0].cpu().numpy().copy()

        if self._show_wrist_cams:
            # --- PiP overlay: wrist cam thumbnails in the bottom corners ---
            # Thumbnail size — change these to adjust overlay size
            WRIST_PIP_W = 300   # thumbnail width  (pixels)
            WRIST_PIP_H = 225   # thumbnail height (pixels)
            WRIST_PIP_PAD = 10   # padding from frame edge (pixels)

            for key, x_anchor in [
                ('left_wrist_camera_rgb',  WRIST_PIP_PAD),                              # bottom-left
                ('right_wrist_camera_rgb', frame.shape[1] - WRIST_PIP_W - WRIST_PIP_PAD),  # bottom-right
            ]:
                wrist = last_obs['policy'][key][0].cpu().numpy()
                thumb = cv2.resize(wrist, (WRIST_PIP_W, WRIST_PIP_H))
                y_anchor = frame.shape[0] - WRIST_PIP_H - WRIST_PIP_PAD
                frame[y_anchor:y_anchor + WRIST_PIP_H,
                      x_anchor:x_anchor + WRIST_PIP_W] = thumb

        self.last_frame = frame

    def quest_to_env_action(self, data):
        env_action = np.zeros(self.env.action_space.shape, dtype=np.float32)

        # Process the controllers quaternions to prevent crashing if a controller is not detected.
        pose_data = data.copy()
        if np.linalg.norm(pose_data['leftController']['quat_wxyz']) < 1e-3:
            pose_data['leftController']['quat_wxyz'] = [1, 0, 0, 0]
        if np.linalg.norm(pose_data['rightController']['quat_wxyz']) < 1e-3:
            pose_data['rightController']['quat_wxyz'] = [1, 0, 0, 0]


        # X button: short press (< 0.5s) = table height recalibration on release,
        #           long press (>= 0.5s) = toggle wrist cam PiP display.
        table_height_in_canonical = 1.0 - self.env.unwrapped.scene["robot"].data.root_pos_w[0,2].item() - self._canonical_z_offset
        x_pressed = pose_data['leftController']['X']

        if x_pressed and self._x_press_start is None:
            # X just pressed — start timer
            self._x_press_start = time.monotonic()
            self._x_long_press_triggered = False

        if x_pressed and self._x_press_start is not None:
            # X held — check for long press
            if not self._x_long_press_triggered and (time.monotonic() - self._x_press_start) >= self._X_LONG_PRESS_S:
                self._show_wrist_cams = not self._show_wrist_cams
                self._x_long_press_triggered = True
                print(f"Wrist cam PiP: {'ON' if self._show_wrist_cams else 'OFF'}")

        if not x_pressed and self._x_press_start is not None:
            # X just released
            if not self._x_long_press_triggered:
                # Short press — table height recalibration
                self.z_offset = table_height_in_canonical - self.quest_xyz_scale * pose_data['leftController']['pos_xyz'][1]
                print(f"Table recalib: z_offset={self.z_offset:.4f}")
            self._x_press_start = None
            self._x_long_press_triggered = False


        # Reset 'zero' position of the controllers by placing them as you prefer and pressing 'Y'; this will make the default robot arm pose to match the controllers' orientation
        # TODO: perhaps, subsequent presses to 'Y' should leave the robot's arms where they are, so pressing y and y again can be used for minor adjustments
        y_pressed = pose_data['leftController']['Y']

        if y_pressed and self.last_y_pressed_status==0:
            self.controllers_tracking_active = not self.controllers_tracking_active
            print('controllers_tracking_active: ', self.controllers_tracking_active)

            # Store updated controller zero rotations for the controllers (i.e., take current orientation of the controllers and use it as new zero pose)
            self.left_controller_zero_R = math_utils._unity_quat_to_isaac_scipy_R(pose_data['leftController']['quat_wxyz'])
            self.right_controller_zero_R = math_utils._unity_quat_to_isaac_scipy_R(pose_data['rightController']['quat_wxyz'])

            print(f'New default rotation for left ({self.left_controller_zero_R.as_quat()}) and right ({self.right_controller_zero_R.as_quat()}) controllers set to the current pose.')

        self.last_y_pressed_status = y_pressed


        # MOVE HEAD
        head_quat_wxyz = pose_data['head']['quat_wxyz']
        head_pitch, head_yaw, head_roll = R.from_quat([head_quat_wxyz[1], head_quat_wxyz[2], head_quat_wxyz[3], head_quat_wxyz[0]]).as_euler('xyz', degrees=True)

        tgt_yaw = -head_yaw / 180.0 * np.pi
        tgt_pitch = head_pitch / 180.0 * np.pi
        tgt_roll = -head_roll / 180.0 * np.pi

        if self.neck_action_indices is not None:
            for key, joint_idx in self.neck_action_indices.items():
                if key == 'pitch':
                    env_action[0, joint_idx] = tgt_pitch
                elif key == 'yaw':
                    env_action[0, joint_idx] = tgt_yaw
                elif key == 'roll':
                    env_action[0, joint_idx] = tgt_roll


        # MOVE ARMS (absolute pose control: position xyz + quaternion wxyz)
        if self.controllers_tracking_active:
            # Absolute position tracking
            env_action[0, self.left_arm_ik_action_index+0] = self.quest_xyz_scale*pose_data['leftController']['pos_xyz'][2]                    # forward
            env_action[0, self.left_arm_ik_action_index+1] = -self.quest_xyz_scale*pose_data['leftController']['pos_xyz'][0]                   # right
            env_action[0, self.left_arm_ik_action_index+2] = self.quest_xyz_scale*pose_data['leftController']['pos_xyz'][1] + self.z_offset    # up

            env_action[0, self.right_arm_ik_action_index+0] = self.quest_xyz_scale*pose_data['rightController']['pos_xyz'][2]                  # forward
            env_action[0, self.right_arm_ik_action_index+1] = -self.quest_xyz_scale*pose_data['rightController']['pos_xyz'][0]                 # right
            env_action[0, self.right_arm_ik_action_index+2] = self.quest_xyz_scale*pose_data['rightController']['pos_xyz'][1] + self.z_offset  # up

            # Clip forward position to be positive (arms shouldn't go behind the robot)
            env_action[0, self.left_arm_ik_action_index+0] = np.clip(env_action[0, self.left_arm_ik_action_index+0], 0, np.inf)
            env_action[0, self.right_arm_ik_action_index+0] = np.clip(env_action[0, self.right_arm_ik_action_index+0], 0, np.inf)

            # Absolute rotation tracking
            left_hand_quat_wxyz = pose_data['leftController']['quat_wxyz']
            right_hand_quat_wxyz = pose_data['rightController']['quat_wxyz']

            # Convert controller orientation to Isaac coordinate frame
            left_target_R = math_utils._unity_quat_to_isaac_scipy_R(left_hand_quat_wxyz)
            right_target_R = math_utils._unity_quat_to_isaac_scipy_R(right_hand_quat_wxyz)

            # Compute rotation delta from controller zero pose
            delta_left_target_R = left_target_R * self.left_controller_zero_R.inv()
            delta_right_target_R = right_target_R * self.right_controller_zero_R.inv()

            # Apply delta to arm zero pose to get target rotation in global coordinates
            left_target_transformed_R = delta_left_target_R * self.left_arm_pose_zero_R
            right_target_transformed_R = delta_right_target_R * self.right_arm_pose_zero_R

            # Convert to quaternion (scipy uses xyzw, IsaacLab uses wxyz)
            left_target_quat_xyzw = left_target_transformed_R.as_quat()
            right_target_quat_xyzw = right_target_transformed_R.as_quat()

            env_action[0, (self.left_arm_ik_action_index+3):(self.left_arm_ik_action_index+7)] = [left_target_quat_xyzw[3],   # w
                                                                                                  left_target_quat_xyzw[0],   # x
                                                                                                  left_target_quat_xyzw[1],   # y
                                                                                                  left_target_quat_xyzw[2]]   # z
            env_action[0, (self.right_arm_ik_action_index+3):(self.right_arm_ik_action_index+7)] = [right_target_quat_xyzw[3],  # w
                                                                                                    right_target_quat_xyzw[0],  # x
                                                                                                    right_target_quat_xyzw[1],  # y
                                                                                                    right_target_quat_xyzw[2]]  # z

        else:
            # Set default pose when not tracking
            env_action[0, self.left_arm_ik_action_index:(self.left_arm_ik_action_index+3)] = self.left_arm_position_zero
            env_action[0, self.right_arm_ik_action_index:(self.right_arm_ik_action_index+3)] = self.right_arm_position_zero

            left_quat_xyzw = self.left_arm_pose_zero_R.as_quat()
            right_quat_xyzw = self.right_arm_pose_zero_R.as_quat()

            env_action[0, (self.left_arm_ik_action_index+3):(self.left_arm_ik_action_index+7)] = [left_quat_xyzw[3],   # w
                                                                                                  left_quat_xyzw[0],   # x
                                                                                                  left_quat_xyzw[1],   # y
                                                                                                  left_quat_xyzw[2]]   # z
            env_action[0, (self.right_arm_ik_action_index+3):(self.right_arm_ik_action_index+7)] = [right_quat_xyzw[3],  # w
                                                                                                    right_quat_xyzw[0],  # x
                                                                                                    right_quat_xyzw[1],  # y
                                                                                                    right_quat_xyzw[2]]  # z


        # GRIPPERS
        left_hand_close = pose_data['leftController']['trigger']
        right_hand_close = pose_data['rightController']['trigger']

        env_action[0, self.left_gripper_action_index] = 2*left_hand_close - 1  # -1 is open, 1 is close
        env_action[0, self.right_gripper_action_index] = 2*right_hand_close - 1  # -1 is open, 1 is close

        return env_action


# TODO: add to documentation, timing of data.  e.g.,   questdata->action   ->  env step ->  take image and show it to the quest display, and take obs to store in the dataset
#                for relative actions, current action =   pose of controller right before env.step  minus  previous pose, so it's technically the action we executed seeing the previous image
# TODO:   currently, data frame (experimentwrapper) puts together the action for the last env.step and the observations AFTER env.step, which is not correct (we are fixing it during training but changing the indices of actions/obs, but this should be better)
