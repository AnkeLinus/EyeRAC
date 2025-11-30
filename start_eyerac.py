import subprocess
import argparse
import os
#import sys
from enum import Enum
    
class CameraEnum(Enum):
    intel435 = "intel435"
    intel455 = "intel455"
    oak_d = "oak_d"
    no_camera = "no_camera"

    def __str__(self) -> str:
        return self.name

# Base directory
base_dir = os.path.expanduser("~/hirac/workspace/")

# Script to source before running each Python script
source_script = os.path.join(base_dir, "source_script.sh")

# List of Python scripts relative to the base directory
scripts = []


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Start camera scripts and optionally additional scripts. Example: python3 start_hirac.py intel no_robot")


parser.add_argument(
    "camera",
    type=CameraEnum,
    choices=CameraEnum,
    default=CameraEnum.intel435,
    help=f"Specify the camera to start (default: {CameraEnum.intel435}). Choose from: {' | '.join([choice.value for choice in CameraEnum])}. If {CameraEnum.no_camera} is choosen, no camera will be started."
)

args = parser.parse_args()


command = ["gnome-terminal"]
# Iterate over the scripts and open each one in a new terminal window
for script in scripts:
    # Full path to the Python script
    script_path = os.path.join(base_dir, script["name"])
    # Directory where the Python script is located
    script_dir = os.path.dirname(script_path)
    print(f"starting script {script_path}")

   
    # Open a new gnome-terminal window, source the script, and then run the Python script
    subprocess.Popen([
        "gnome-terminal",
        "--title", script["title"],        # Set terminal title
        "--",
        "bash", "-c", f"source {source_script};cd {script_dir}; python3 {script_path}; exec bash"
    ])


# start webserver seperately because it is now a module
print("Starting Eye tracker Communication")
subprocess.Popen([
        "gnome-terminal",
        "--title", "Eye tracker Communication",        # Set terminal title
        "--",
        "bash", "-c", f"source {source_script};cd {os.path.join(base_dir,'src')}; python3 -m eye_tracking_controller.eye_tracker_interface.eye_tracker_communicator; exec bash"
    ])

print("Starting task detection")
subprocess.Popen([
        "gnome-terminal",
        "--title", "Task Detection",        # Set terminal title
        "--",
        "bash", "-c", f"source {source_script};cd {os.path.join(base_dir,'src')}; python3 -m eye_tracking_controller.task_detection.task_detection_node.main; exec bash"
    ])

print("Starting Gaze Cursor")
subprocess.Popen([
        "gnome-terminal",
        "--title", "Gaze Cursor",        # Set terminal title
        "--",
        "bash", "-c", f"source {source_script};cd {os.path.join(base_dir,'src')}; python3 -m eye_tracking_controller.gaze_cursor.main; exec bash"
    ])

print("Starting feature matching")
subprocess.Popen([
        "gnome-terminal",
        "--title", "Feature Matching",        # Set terminal title
        "--",
        "bash", "-c", f"source {source_script};cd {os.path.join(base_dir,'src')}; python3 -m eye_tracking_controller.feature_matching.main; exec bash"
    ])

print("Starting object detection")
subprocess.Popen([
        "gnome-terminal",
        "--title", "Object Detection",        # Set terminal title
        "--",
        "bash", "-c", f"source {source_script};cd {os.path.join(base_dir,'src')}; python3 -m object_detection.object_detection_node.main; exec bash"
    ])

'''
# start webserver seperately because it is now a module
print("Starting webserver")
subprocess.Popen([
        "gnome-terminal",
        "--title", "Webserver",        # Set terminal title
        "--",
        "bash", "-c", f"source {source_script};cd {os.path.join(base_dir,'src')}; python3 -m webserver.run; exec bash"
    ])
'''