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
5. Install all necessary dependencies.

## Usage



---

## Citation

If you use this code in your research, please cite the following paper:

> **Author(s)**, *"Title of Your Paper"*, Conference/Journal, Year.  
> DOI: [https://doi.org/xxx](https://doi.org/xxx)

A machine-readable citation file is included as [`CITATION.cff`](./CITATION.cff).

---

## License
This project is released under the [Apache 2.0 License](./LICENSE).
