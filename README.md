# EyeRAC

# Eye-Tracking Driven Shared Control Framework


EyeRAC is an eye-tracking driven control for robotic arms based on a shared control approach. Shared control improves Human-Robot Interaction by reducing the user’s workload and increasing the robot’s autonomy. It enables robots to perform tasks under the user’s supervision.  

Current eye-tracking-driven approaches face challenges, such as the limited accuracy of eye trackers and the difficulty of interpreting gaze when multiple objects or abstract tasks are involved.  

This code is supplementary to the publication, in which we introduce an **eye-tracking-driven control framework** that:  
- Minimizes the necessity of precise calibration by transmitting user intent and task information via **pictogram-based fiducial markers**,  
- Bridges camera views through a **feature matching approach**, eliminating the need to know the user’s position relative to the robot or objects,  
- Integrates state-of-the-art **object detection models**, making it easily adaptable to new tasks and objects.  

In evaluation, the framework correctly interpreted object and task selections in up to **97.9% of trials**.  

## Hardware
The framework was implemented in **ROS2 Humble** on **Ubuntu 22.04**.

**Components used in EyeRAC:**  
- **Eye tracking:** Pupil Core glasses with *Pupil Capture v3.5.1* (Pupil Labs)    
- **Camera:** Intel RealSense D455 for recording the robot’s field of view  

The framework communicates via ROS2 topics, enabling flexible interaction between modules and compatibility with additional robots and sensors. Please note, that this is an interface which can be tailored to your robots control sequence.  

## Installation
Install the necessary software to interact with the hardware:
1. ROS2 Humble environment: Setup a ROS2 HUmble environment as stated by the ROS-community (https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).
2. Pupil Core: Install Pupil Core Capture as described on the manufacturing website (https://docs.pupil-labs.com/core/getting-started/).
3. Intel ReaslSense D455: Install Intel RealSense ROS wrapper as explained by the manufacturer (https://github.com/IntelRealSense/realsense-ros).
4. Download the git. It already contains the ROS2-workspace needed to interact with the system.
5. Install all necessary dependencies via requirements.txt.
6. Go to the workspace folder and build the workspace once: `colcon build`

## Usage
The repository contains the code of the framework and test files allowing testing certain functionalities of the system. 

Test files:
- Eye tracker interface:
   - Test_World_video.py: Streams the world video scene of the pupil core glasses and visualizes it.
   - Test_eyetracker_function.py: Streams fixation data from the pupil core glasses.
   - Both are used to evaluate if the eye tracker works correctly.
- Feature Matching:
   - Test_Feature_Matching_Evaluation.py: Used to compare different feature detectors and matching algorithms to find a suitable solution for your application.
- Gaze cursor:
   - Test_Sound_Output.py: Tests if the sound output works for the system you are currently working with.

Using the framework:
 1. Start up Pupil Core Capture and calibrate the camera.
 2. Start up the Intel RealSense Node with the following command: `ros2 launch realsense2_camera rs_launch.py` Don't forget to source before! `source /opt/ros/humble/setup.bash`
 3. Go via terminal into the cloned repository folder and start up EyeRAC with the following command: `python3 start_eyerac.py intel455`
All necessary nodes start up automatically.

---
## Cite this

If you use this repository for your own research, please cite: 
Anke Fischer-Janzen and Thomas M. Wendt and Kristof Van Laerhoven. Eye-Tracking-Driven Control in Daily Task Assistance for Assistive Robotic Arms, 	arXiv:2601.17404 [cs.RO], doi: 
https://doi.org/10.48550/arXiv.2601.17404, 2026.

```
@misc{fischerjanzen2026eyetrackingdrivencontroldailytask,
      title={Eye-Tracking-Driven Control in Daily Task Assistance for Assistive Robotic Arms}, 
      author={Anke Fischer-Janzen and Thomas M. Wendt and Kristof Van Laerhoven},
      year={2026},
      eprint={2601.17404},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.17404}, 
}
```
Thank you!

---

## License
This project is released under the [Apache 2.0 License](./LICENSE).
