import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import os

import logging
import logging.config

from eye_tracking_controller.eye_tracker_interface.eye_tracker_fixation import EyeTrackerFixation
from eye_tracking_controller.eye_tracker_interface.eye_tracker_video import EyeTrackerVideo

class EyeTrackerCommunicator(Node):

    def __init__(
            self,
            fixation_publisher: EyeTrackerFixation,
            video_publisher: EyeTrackerVideo
            ):
        
        self.node = Node("Handle_Fixation_Node")

        super().__init__('eye_tracker_communicator')
        self.video_publisher = video_publisher
        self.fixation_publisher = fixation_publisher

    def create_logger(config_name: str) -> logging.Logger:
        # configure logging
        current_dir = os.path.dirname(__file__)
        logging_config_path = os.path.join(current_dir, config_name)
        print("Trying to load logging config from %s", logging_config_path)
        logging.config.fileConfig(fname=logging_config_path, disable_existing_loggers=False)
        logger = logging.getLogger("Eye tracker Communicator")
        return logger

def main(args) -> None:
    logger = EyeTrackerCommunicator.create_logger("logging.conf")
    logger.debug("logger created")
    rclpy.init(args=None)

    video_publisher = EyeTrackerVideo()
    fixation_publisher = EyeTrackerFixation()
    eye_tracker_communicator = EyeTrackerCommunicator(
        video_publisher=video_publisher,
        fixation_publisher=fixation_publisher
    )
    executor = MultiThreadedExecutor()
    
    executor.add_node(node=eye_tracker_communicator)
    executor.add_node(node=video_publisher)
    executor.add_node(node=fixation_publisher)

    try:
        executor.spin()
    except KeyboardInterrupt:
        logger.debug("Keyboard interrupt, shutting down")
        video_publisher.destroy_node()
        fixation_publisher.destroy_node()

        logger.debug("Nodes destroyed")

    logger.debug("Shutting down complete, exiting")
    exit(1)
    
if __name__ == '__main__':
    print("Starting communication to Eye Tracker.")
    args = None
    main(args)
