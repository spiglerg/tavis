"""
First time running:
  install the apk on your device

$ adb install -r quest_teleop.apk




Don't forget to run the following when you connect the Quest to the computer:
    [may need   $ adb kill-server,     $ sudo adb start-server]
        check with "$ adb devices" that the Quest is connected

    $ adb forward tcp:9500 tcp:9500
    $ adb forward tcp:9501 tcp:9501

Open the quest app.

$ adb shell am force-stop com.airlab.quest_teleop
$ adb shell monkey -p com.airlab.quest_teleop -c android.intent.category.LAUNCHER 1


Then, from your own script, call QuestTeleopMain() with the correct arguments.

    try:
        QuestTeleopMain(env,
                        robot_ctrl_rate=30,
                        video_port=9500,
                        pose_port=9501,
                        frame_width=640,
                        frame_height=480,
                        jpeg_quality=80)
    except KeyboardInterrupt:
        print("\\nInterrupted by user")

"""

import threading
import cv2
import json
import socket
import struct
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym


####################################
# The code below automatically handles everything, provided an 'env' wrapped with the proper ExperimentWrapper
# has been created.
####################################

class QuestTCPStreamer:
    def __init__(self,
                 env,
                 ctrl_rate,
                 video_port=9500,
                 pose_port=9501,
                 frame_width=640,
                 frame_height=480,
                 jpeg_quality=80,
                 debug=False):
        self.running = False
        self.video_socket = None
        self.pose_socket = None

        self.env = env
        self.ctrl_rate = ctrl_rate
        self.video_port = video_port
        self.pose_port = pose_port
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.jpeg_quality = jpeg_quality

        self.last_data = None

        self.data_lock = threading.Lock()
        self.threads = []

        self.debug = debug

    def start(self):
        """Main entry point"""
        self.running = True

        try:
            # Connect to Quest
            print(f"Connecting to Quest on ports {self.video_port} and {self.pose_port}...")

            # Video socket
            self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.video_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            self.video_socket.connect(('127.0.0.1', self.video_port))
            print("Video socket connected")

            # Pose socket
            self.pose_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.pose_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.pose_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            self.pose_socket.connect(('127.0.0.1', self.pose_port))
            print("Pose socket connected")

            print("Connected! Streaming...")

            # Start threads
            video_thread = threading.Thread(target=self.video_send_loop, daemon=True)
            pose_thread = threading.Thread(target=self.pose_receive_loop, daemon=True)

            self.threads = [video_thread, pose_thread]

            for thread in self.threads:
                thread.start()

            self.main_loop()

        except Exception as e:
            print(f"Error: {e}")
            self.cleanup()
            raise

    def video_send_loop(self):
        """Send video frames to Quest"""
        frame_id = 0

        while self.running:
            # print('DEBUG: video send loop')

            try:
                # Read frame
                if self.env.last_frame is None:
                    time.sleep(0.001)
                    continue

                frame = self.env.last_frame.copy()

                # Convert to BGR for internal processing by OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Create header: [frame_id:4][timestamp:8][size:4]
                frame_id += 1
                timestamp = int(time.time() * 1000)

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

                # if frame is a list, use it as single/stereo camera frames; else proceed as single camera
                if isinstance(frame, list):
                    if len(frame) == 2:
                        num_cams = 2
                    elif len(frame) == 1:
                        num_cams = 1
                        frame = frame[0]
                else:
                    num_cams = 1

                if num_cams == 1:
                    # Single camera: only left frame
                    if frame.shape[:2] != (self.frame_width, self.frame_height):
                        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                    _, jpeg_data = cv2.imencode('.jpg', frame, encode_param)
                    jpeg_bytes = jpeg_data.tobytes()

                elif num_cams == 2:
                    if frame[0].shape[:2] != (self.frame_width, self.frame_height):
                        frame[0] = cv2.resize(frame[0], (self.frame_width, self.frame_height))
                    if frame[1].shape[:2] != (self.frame_width, self.frame_height):
                        frame[1] = cv2.resize(frame[1], (self.frame_width, self.frame_height))
                    _, jpeg_data_left = cv2.imencode('.jpg', frame[0], encode_param)
                    jpeg_bytes_left = jpeg_data_left.tobytes()
                    _, jpeg_data_right = cv2.imencode('.jpg', frame[1], encode_param)
                    jpeg_bytes_right = jpeg_data_right.tobytes()

                if num_cams == 1:
                    header = struct.pack('<IQBI',
                                         frame_id,
                                         timestamp,
                                         num_cams,
                                         len(jpeg_bytes))
                    # Send header + data
                    data_to_send = header + jpeg_bytes
                    self.video_socket.sendall(data_to_send)

                elif num_cams == 2:
                    header = struct.pack('<IQBII',
                                         frame_id,
                                         timestamp,
                                         num_cams,
                                         len(jpeg_bytes_left),
                                         len(jpeg_bytes_right))
                    # Send header + left data + right data
                    data_to_send = header + jpeg_bytes_left + jpeg_bytes_right
                    self.video_socket.sendall(data_to_send)

                # Display the image with cv2
                # cv2.imshow('Video Feed', frame)
                # cv2.waitKey(1)  # Allow OpenCV to process the window

                # Target 60 FPS
                # max 60 fps
                time.sleep(0.011)

            except Exception as e:
                print(f"Video send error: {e}")
                break

    def pose_receive_loop(self):
        """Receive pose data from Quest"""
        buffer = bytearray()

        while self.running:
            # print('DEBUG: pose receive loop')

            try:
                # Read data
                data = self.pose_socket.recv(4096)
                if not data:
                    time.sleep(0.001)
                    continue

                buffer.extend(data)

                # Process complete messages
                while len(buffer) >= 4:
                    # Read size
                    size = struct.unpack('<I', buffer[:4])[0]
                    if len(buffer) < 4 + size:
                        break  # Wait for more data

                    # Extract message
                    json_data = buffer[4:4+size]
                    buffer = buffer[4+size:]

                    # Parse and print pose data occasionally
                    try:
                        # data keys 'head', 'leftController', 'rightController', 'timestamp';
                        #       all have 'pos_xyz' and 'quat_wxyz'
                        #       left/right controllers also have buttons:   XY left, AB right, grip and trigger
                        # head pitch:  pos down;  yaw: pos right;  roll: pos tilt left
                        # xyz left controller:   x pos right, y pos up, z pos forward
                        # head_quat_wxyz = data['head']['quat_wxyz']
                        # pitch, yaw, roll = R.from_quat([head_quat_wxyz[1],
                        #                                 head_quat_wxyz[2],
                        #                                 head_quat_wxyz[3],
                        #                                 head_quat_wxyz[0]]).as_euler('xyz', degrees=True)
                        data = json.loads(json_data.decode('utf-8'))

                        with self.data_lock:
                            self.last_data = data.copy()

                    except json.JSONDecodeError:
                        pass

            except Exception as e:
                print(f"Pose receive error: {e}")
                break

    def main_loop(self):
        period = 1.0 / self.ctrl_rate
        last_step_time = time.monotonic()

        try:
            while True:
                current_time = time.monotonic()
                if current_time - last_step_time >= period:
                    if self.debug:
                        print(f'Loop time (it should be {period}): {current_time-last_step_time}')

                    with self.data_lock:
                        if self.last_data is not None:
                            data_copy = self.last_data.copy()
                        else:
                            data_copy = None

                    if data_copy is not None:
                        self.env.step(data_copy)

                    last_step_time = current_time
                #else:
                #    time.sleep(0.1/1000.0)  # sleep 0.1 ms to avoid busy wait

        finally:
            # This MUST run even when exception occurs - finally doesn't suppress exceptions
            print('Cleaning up quest_teleop...')
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False

        if self.video_socket:
            try:
                self.video_socket.close()
            except:
                pass

        if self.pose_socket:
            try:
                self.pose_socket.close()
            except:
                pass

        print("Cleaned up, exiting...")


def QuestTeleopRun(env,
                   robot_ctrl_rate=30,
                   video_port=9500,
                   pose_port=9501,
                   frame_width=640,
                   frame_height=480,
                   jpeg_quality=80,
                   debug=False):
    """
    Run the Quest teleoperation loop.

    'env' is a gymnasium environment wrapped to receive quest pose data as action (to be converted to robot actions)
    and sets env.last_frame to the latest camera image, that will be sent to the Quest device for display.

    See tavis.wrappers for details on how to wrap the environment.
    """
    streamer = QuestTCPStreamer(env=env,
                                ctrl_rate=robot_ctrl_rate,
                                video_port=video_port,
                                pose_port=pose_port,
                                frame_width=frame_width,
                                frame_height=frame_height,
                                jpeg_quality=jpeg_quality,
                                debug=debug)
    streamer.start()
