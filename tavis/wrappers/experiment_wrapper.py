from abc import ABC, abstractmethod
import os
import shutil
from pathlib import Path

import cv2
import gymnasium as gym
import torch

import omni.log

from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import make_policy



class AbstractExperimentWrapper(gym.Wrapper):
    """
    Base experiment wrapper class for Quest teleoperation in IsaacLab environments.
    The wrapper wraps an isaaclab environment to act as 1) interface between oculus quest controls and the environment,
    and 2) to handle the experiment logic, such as recording data, handling buttons, etc.

    The wrapper is meant for automatic use by QuestTeleopMain, which will call the step method with the quest data.
    When using the environment directly, for example to run a trained policy, no wrapper is needed, and the environment
    can be used directly.

    IMPORTANT:  your derived wrapper must implement the following:
        - 'quest_to_env_action(quest_data)':  method to convert the quest pose data to environment actions
        - 'save_fpv_frame(last_obs)' : a method to save the first-person view frame to the experiment recording into the env/wrapper
                             field  'last_frame' (this is used by the Quest streaming utility to send to the Quest device).
                            Note:  last_frame can be a list with 2 images, in which case stereoscopic vision is enabled.

        - in the constructor, you can override the field 'self.text_to_display' to display a custom text on the frame until the first
          recording starts, for example, to provide wrapper-specific interface instructions.

    Intercepted commands / interface:
        - Press right 'grip' to start/stop recording an episode. On 'start', previous unsaved data is discarded. If 'done', recording automatically stops.
        - Then, press 'A' to save the episode.
        - Press 'B' to discard the last recorded data.

        - If dataset file already exists, this should keep appending demonstrations to the existing dataset.
    """
    def __init__(self,
                 env,
                 dataset_repo_id:str,
                 dataset_root_dir:str,
                 dataset_description:str,
                 dataset_fps:int,
                 dataset_obs_keys_to_record:dict[str, str],
                 action_quat_indices:list[int] = None):
        """
            dataset_repo_id:  repo id for the lerobot dataset; this is used to load the dataset
            dataset_root_dir:  root directory for the lerobot dataset; if the path exists, new episodes will be appended to the existing dataset
            dataset_description:  short description of the dataset; this will be saved to the dataset
            dataset_fps:  fps for the dataset; this will be saved to the dataset
            dataset_obs_keys_to_record:  dict of observation keys to record; these will be saved to the dataset. Keys correspond to environment observation keys,
                                        and values correspond to the dataset key to save the observation under. If None, the same name will be used.
                                        Dataset keys will be named 'observation.state.[key]' for vectors, and
                                        'observation.images.[key]' for images.
            action_quat_indices:  list of starting indices for quaternions (wxyz) in the action vector.
                                  If provided, quaternions will be converted to 6D rotation representation
                                  before storing to dataset. E.g., [3, 10] means quats at indices 3-6 and 10-13.
            TODO: rename images in a standardzied way, e.g.,    OBS_IMAGE_1 for main camera, OBS_IMAGE_2 and OBS_IMAGE_3 for wrists
        """
        super().__init__(env)

        self.dataset_repo_id = dataset_repo_id
        self.dataset_root_dir = dataset_root_dir
        self.dataset_description = dataset_description
        self.dataset_fps = dataset_fps
        self.dataset_obs_keys_to_record = dataset_obs_keys_to_record
        self.action_quat_indices = action_quat_indices

        # Check if path exists but dataset is empty
        dataset_path = Path(self.dataset_root_dir)
        if dataset_path.exists():
            contents = [p.name for p in dataset_path.iterdir()]

            if not contents or contents == ['meta']:
                print(f'Dataset {self.dataset_repo_id} is empty, creating new dataset')
                shutil.rmtree(dataset_path)

        if os.path.exists(self.dataset_root_dir):
            print(f'Resuming dataset {self.dataset_repo_id} from {self.dataset_root_dir}')
            self.resume_dataset = True
        else:
            print(f'Creating new dataset {self.dataset_repo_id} in {self.dataset_root_dir}')
            self.resume_dataset = False

        # Interface utilities
        self.last_a_press_status = False
        self.last_b_press_status = False
        self.last_reset_press_status = False
        self.last_grip_press_status = False
        self.is_recording_episode = False

        self.text_to_display = ''

        # Build the observation, cameras, and action features. If loading an existing dataset, we will check that the features match.
        env_action_shape = self.env.action_space.shape[1::]
        dataset_action_shape = env_action_shape

        action_features = {'action': {'dtype': 'float32',
                                        'shape': dataset_action_shape,
                                        'names': 'env actions'} }

        # Compute actual observations to get correct keys and shapes
        # (observation_space may be incomplete for sensors attached to robot links).
        # Use .unwrapped to reach the Isaac Lab env — gymnasium 1.x Wrapper
        # no longer delegates attribute access through the wrapper chain.
        actual_obs = self.env.unwrapped.observation_manager.compute()

        obs_features = {}
        has_cameras = False
        for key in self.dataset_obs_keys_to_record:
            if key in actual_obs.get('policy', {}).keys():
                obs_shape = actual_obs['policy'][key].shape[1::]
                cam_prefix = 'images.' if len(obs_shape)==3 else ''
                dtype = 'video' if len(obs_shape)==3 else 'float32'  # image
                obs_name = self.dataset_obs_keys_to_record[key] if self.dataset_obs_keys_to_record[key] is not None else key
                obs_features['observation.'+cam_prefix+obs_name] = {'dtype': dtype,
                                        'shape': obs_shape,
                                        'names': ['height', 'width', 'channels']}
                if cam_prefix == 'images.':
                    has_cameras = True
            else:
                available_keys = list(actual_obs.get('policy', {}).keys())
                raise ValueError(f'Observation key {key} not found in actual observations. Available keys: {available_keys}')

        extra_features = {
            'language_instruction': {'dtype': 'string', 'shape': (1,), 'names': None},
        }

        dataset_features = {**action_features, **obs_features, **extra_features}

        if self.resume_dataset:
            # Load existing dataset
            self.dataset = LeRobotDataset(
                self.dataset_repo_id,
                root=self.dataset_root_dir,
                force_cache_sync=False, # force using local data instead of cloud hf dataset
            )

            if has_cameras:
                self.dataset.start_image_writer(
                    num_processes=0, # 0=use threads only;  cfg.dataset.num_image_writer_processes,
                    num_threads=12,    #cfg.dataset.num_image_writer_threads_per_camera (4) * len(robot.cameras),
                )
            # TODO: modify to avoid using LeRobot Robot class: https://github.com/huggingface/lerobot/blob/519b76110efeea55a4f919895d0029dc0df41e8b/src/lerobot/utils/control_utils.py#L197
            #sanity_check_dataset_robot_compatibility(dataset, robot, self.dataset_fps, dataset_features)

        else:
            # Create new dataset
            self.dataset = LeRobotDataset.create(
                repo_id=self.dataset_repo_id,
                fps=self.dataset_fps,
                root=self.dataset_root_dir,
                #robot_type=self.env.robot.name,
                features=dataset_features,
                #use_videos=True, # default true
                image_writer_processes=0, # 0=use threads only;  cfg.dataset.num_image_writer_processes,
                image_writer_threads=12,    #cfg.dataset.num_image_writer_threads_per_camera (4) * len(robot.cameras),
            )

    def _clear_episode_buffer_with_video_cleanup(self):
        """
        Workaround for a lerobot bug: `LeRobotDataset.clear_episode_buffer()` only deletes
        temporary PNG dirs for `image_keys`, NOT `video_keys`. When an episode is discarded
        (via 'B' or by toggling recording), leftover PNGs from the previous attempt at the
        same `episode_index` remain on disk. The next (shorter) episode then encodes those
        leftover frames into the video, producing a video longer than the data buffer.

        This helper clears the buffer and then manually removes the temp PNG dir for each
        video key at the current `episode_index`, before any new frames are written.
        """
        episode_index = self.dataset.episode_buffer["episode_index"]
        if hasattr(episode_index, "item"):
            episode_index = episode_index.item() if getattr(episode_index, "size", 1) == 1 else int(episode_index[0])
        self.dataset.clear_episode_buffer()
        for cam_key in self.dataset.meta.video_keys:
            img_dir = self.dataset._get_image_file_dir(episode_index, cam_key)
            if img_dir.is_dir():
                shutil.rmtree(img_dir)

    @abstractmethod
    def save_fpv_frame(self, last_obs):
        """
        Save the first-person view frame to the experiment recording.
        This must save the frame to self.last_frame, which will be used by QuestTeleopMain to stream the frame to the Quest device.
        """
        pass

    @abstractmethod
    def quest_to_env_action(self, quest_data):
        """
        Convert quest pose data to environment actions.

        Args:
            quest_pose_data: dictionary with keys 'head', 'leftController', 'rightController', 'timestamp'
                            * 'head', 'leftController', and 'rightController' all have 'pos_xyz' and 'quat_wxyz'
                            * left/right controllers also have buttons and thumbstick:
                                left controller: 'X', 'Y', 'trigger', 'grip'
                                right controller: 'A', 'B' 'trigger', 'grip'
                                thumbstick is a Vector2 with values [-1, 1]; first value is horizontal left to right;
                                                                             second value is vertical bottom to top

                            * Quaternion positions and orientations are in Unity frame of reference:
                                    x pos right, y pos up, z pos forward

                                    pitch pos down, yaw pos right, roll pos tilt left

                            * IsaacLab's frame of reference (try to keep this consistent across environments!):
                                    x forward, y left, z up

        Returns:
            env_action: numpy array of actions for the environment; this must match the specific environment used
        """
        pass

    def step(self, quest_data):
        """
        Step the environment with the given quest_data action.
        """
        action = self.quest_to_env_action(quest_data)

        observation, reward, terminated, truncated, info = self.env.step(torch.Tensor(action))
        self.save_fpv_frame(observation)

        if terminated or truncated:
            # Automatically stop recording if episode is done
            self.is_recording_episode = False
            self.text_to_display = "stopped\n'A' to save, 'B' to discard\nright 'grip' to start new\nleft 'grip' to reset env"

        # TODO: handle experiment logic by reading quest_data button presses, consider 'done' and 'info' (to see if
        #     the episode ended successfully or in a failure); eventually override self.env.last_frame with
        #     diagnostic text, after copying the frame to the experiment recording, etc...

        reset_pressed = quest_data['leftController']['grip']
        if reset_pressed and self.last_reset_press_status==0:
            self.env.reset()
        self.last_reset_press_status = reset_pressed

        grip_pressed = quest_data['rightController']['grip']
        if grip_pressed and self.last_grip_press_status==0:
            # If starting new recording, discard previous unsaved data
            if not self.is_recording_episode:
                if (self.dataset.episode_buffer is not None) and (self.dataset.episode_buffer["size"] > 0):
                    # Last buffer was not cleared or saved, so we need to clear it
                    self._clear_episode_buffer_with_video_cleanup()
            self.is_recording_episode = not self.is_recording_episode
            self.text_to_display = "recording..." if self.is_recording_episode else "stopped\n'A' to save, 'B' to discard\nright 'grip' to start new"

        self.last_grip_press_status = grip_pressed

        a_pressed = quest_data['rightController']['A']
        if a_pressed and self.last_a_press_status==0:
            # Check if there is unsaved data
            omni.log.info(f"Saving episode with size {self.dataset.episode_buffer['size']}")
            if self.dataset.episode_buffer["size"] > 0:
                self.is_recording_episode = False
                self.dataset.save_episode()
                # CRITICAL: Wait for all images to be written to disk before continuing
                if self.dataset.image_writer is not None:
                    self.dataset.image_writer.wait_until_done()
                self.text_to_display = "SAVED"
                print('SAVED! [Remember to quit the app with CTRL-C if you wish to save the recorded data correctly.]')
        self.last_a_press_status = a_pressed

        b_pressed = quest_data['rightController']['B']
        if b_pressed and self.last_b_press_status==0:
            # Stop recording and discard episode
            self.is_recording_episode = False
            self.text_to_display = "DISCARDED"
            self._clear_episode_buffer_with_video_cleanup()
        self.last_b_press_status = b_pressed

        if self.is_recording_episode:
            action_frame = {'action': action[0,:].copy()}  # we are controlling a single environment
            task = self.env.unwrapped.task if hasattr(self.env.unwrapped, 'task') else None
            task_class = type(task).__name__ if task is not None else ""
            prompt = task.get_prompt() if task is not None else ""
            task_frame = {'task': task_class, 'language_instruction': prompt}
            observation_frame = {}
            for key in self.dataset_obs_keys_to_record:
                if key in observation['policy'].keys():
                    obs_shape = observation['policy'][key].shape[1::]
                    cam_prefix = 'images.' if len(obs_shape) == 3 else ''
                    obs_name = self.dataset_obs_keys_to_record[key] if self.dataset_obs_keys_to_record[key] is not None else key
                    observation_frame['observation.'+cam_prefix+obs_name] = observation['policy'][key][0,:].cpu().numpy()
            frame = {**observation_frame, **action_frame, **task_frame}
            self.dataset.add_frame(frame)

        self.last_frame = self.last_frame.copy()

        # Display status text at the top (red)
        text_lines = self.text_to_display.split('\n')
        line_height = 25
        for i, line in enumerate(text_lines):
            (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = (self.last_frame.shape[1] - text_width) // 2
            text_y = line_height * (i + 1)
            cv2.putText(self.last_frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)

        # Display language prompt at the bottom
        if hasattr(self.env.unwrapped, 'task'):
            prompt = self.env.unwrapped.task.get_prompt()
            if prompt:
                (pw, _), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                px = (self.last_frame.shape[1] - pw) // 2
                py = self.last_frame.shape[0] - 15
                ####cv2.putText(self.last_frame, prompt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

        # Draw a little blue fixation pixel in the center for alignment
        center_x = self.last_frame.shape[1] // 2
        center_y = self.last_frame.shape[0] // 2
        cv2.circle(self.last_frame, (center_x, center_y), 1, (0, 0, 255), -1)

        return observation, reward, terminated, truncated, info
