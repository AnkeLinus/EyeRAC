import argparse
import json
import logging
import logging.config
import os

import rclpy
from rclpy.executors import MultiThreadedExecutor
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from general.world_video import WorldVideoSubscriber
from eye_tracking_controller.task_detection.task_detection_node.task_detection_publisher import TaskDetection

'''
Purpose of this node:
Detects the pictograms by using the Task detection model.
'''

# create logger
def create_logger(config_name: str) -> logging.Logger:
    # configure logging
    current_dir = os.path.dirname(__file__)
    logging_config_path = os.path.join(current_dir, config_name)
    print("Trying to load logging config from %s", logging_config_path)
    logging.config.fileConfig(fname=logging_config_path, disable_existing_loggers=False)
    logger = logging.getLogger(__name__)
    return logger

logger = create_logger("logging.conf")
logger.debug("logger created")

def main() -> None:
    logger = create_logger("logging.conf")
    logger.debug("Initializing rclpy")
    rclpy.init(args=None)

    logger.debug("Creating task detection model")
    current_dir = os.path.dirname(__file__)
    object_detection_model = YOLO(os.path.join(current_dir,"weights/best.pt"))

    logger.debug("Creating object tracker")
    object_tracker = DeepSort(max_age=5)

    logger.debug("Creating eye tracking subscriber")
    detector = TaskDetection(logger,object_detection_model=object_detection_model,object_tracker=object_tracker)
    
    #camera_subscriber = CameraSubscriber(
    camera_subscriber = WorldVideoSubscriber(
    
        callback_function=detector.et_camera_callback       
    )


    logger.debug("Spinning up nodes")

    if camera_subscriber.subscription:
        logger.debug("Created node successfully")
    else:
        logger.debut("No valid camera topic found, node not created")

    executor = MultiThreadedExecutor()
    executor.add_node(node=camera_subscriber)
    executor.add_node(node=detector)

    try:
        executor.spin()
    except KeyboardInterrupt:
        
        logger.debug("Keyboard interrupt, shutting down")
        executor.shutdown()
        camera_subscriber.destroy_node()
        detector.destroy_node()
        logger.debug("Nodes destroyed")

    logger.debug("Shutting down complete, exiting")
    exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Task Detection Node",
        description="EyeRAC node that detects and tracks task pictograms in the video stream. "
    )

    args=parser.parse_args()
    main()
