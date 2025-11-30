'''
This approach is based on a dwell time selection process with fixations between 300 and 600 ms. This is the minimum duration for a fixation with Pupil Core glasses. 
As soon as a fixation is detected within a bounding box, this bounding box is selected. The data is then 
'''

import logging
import logging.config
import os

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from general.world_video import WorldVideoSubscriber
from general.gaze_fixation import GazeFixationSubscriber
from typing import List
from eye_tracking_controller.gaze_webserver.webserver.ros_node.bounding_boxes import BoundingBoxSubscriber, BoundingBox
from hirac_msgs.msg import Features

from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np


import pygame 
#playsound("~/usr/share/sounds/sound-icons/prompt.wav")
import datetime
#ts = time.time()
#timestp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

class GazeCursor(Node):

    def __init__(
        self,
        logger: logging.Logger,
        world_video_subscriber: WorldVideoSubscriber,
        fixation_subscriber: GazeFixationSubscriber
        ):
        super().__init__("gaze_cursor_node")
        self.feature_eye_tracking_publisher = self.create_publisher(Features, '/feature_matching_eyetracking', 10)
        self.world_video_subscriber = world_video_subscriber
        self.fixation_subscriber = fixation_subscriber
        

        self.logger = logger
        self.bridge = CvBridge()

    def create_logger(config_name: str) -> logging.Logger:
        # configure logging
        current_dir = os.path.dirname(__file__)
        logging_config_path = os.path.join(current_dir, config_name)
        print("Trying to load logging config from %s", logging_config_path)
        logging.config.fileConfig(fname=logging_config_path, disable_existing_loggers=False)
        logger = logging.getLogger("Gaze Cursor")
        return logger
    
    def task_callback(self, update: List[BoundingBox], img) -> None:
        #self.logger.debug("New fixation")

        fixation = self.fixation_subscriber.get_latest_fixation()
        image = self.world_video_subscriber.get_latest_image()

        height = 480
        width = 640

        for bounding_box in update:
            range_x = range(int(bounding_box.x), int(bounding_box.x+bounding_box.width))
            range_y = range(int(bounding_box.y), int(bounding_box.y+bounding_box.height))
            fix_x = round(fixation.norm_pos_x,2)
            fix_y = round((1-fixation.norm_pos_y),2)
            
            range_x_norm = [round((x / width),2) for x in range_x]
            range_y_norm = [round((y / height),2) for y in range_y]
            
            if fix_x in range_x_norm:       
                if fix_y in range_y_norm:       # 1-fixation is necessary, since 0-coordinates are not alligning. 0-coordinate in fixation is lower left corner, 0-coordinate for bounidng box is upper left corner.
                    # For Measurement
                    pygame.init()
                    pygame.mixer.init()
                    plays = pygame.mixer.Sound("/usr/share/sounds/sound-icons/prompt.wav")
                    plays.play()
                    ts = time.time()
                    timestp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    self.logger.info(f"Bounding Box Selected: {timestp}")
                    self.logger.info(f"Fixation was detected within bounding box for bounding box {bounding_box.object_class_id}.")

                    selected_bb = self.cut_bb(image, range_x, range_y, bounding_box.width, bounding_box.height, False)

                    if bounding_box.object_class_id == 3:
                        time.sleep(2)                  
                        fixation2 = self.fixation_subscriber.get_latest_fixation()
                        self.secondary_object_exist =True

                        # For Measurement
                        pygame.init()
                        pygame.mixer.init()
                        plays = pygame.mixer.Sound("/usr/share/sounds/gnome/default/alerts/drip.ogg")
                        plays.play()
                        ts = time.time()
                        timestp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:3]
                        self.logger.info(f"Secondary object was selected: {timestp}")

                        fix_x2 = round(fixation2.norm_pos_x,2)
                        fix_y2 = round((1-fixation2.norm_pos_y),2)
                        secondary_object = self.cut_bb(image, fix_x2, fix_y2, bounding_box.width,bounding_box.height, True)
                    else:
                        self.secondary_object_exist =False
                        secondary_object = None

                    self.publish_msg(selected_bb, bounding_box.object_class_id, self.secondary_object_exist, secondary_object)
                    #time.sleep(2)
                    break

    def cut_bb(self, image, range_x, range_y, width, height, secondary_obj):
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        
        # Crop the image using numpy slicing
        if secondary_obj == False:
            cropped = cv_image[int(np.round((range_y[0]/480*720)-2*height)):int(np.round((range_y[-1]/480*720)+2*height)), 
                           int(np.round((range_x[0]/620*1280)-2*width)):int(np.round((range_x[-1]/620*1280)+2*width))] 
            cv2.imwrite("whole_image.png", cv_image)
            cv2.imwrite("cropped_image.png", cropped)
        else:
            cropped = cv_image[int(np.round((range_y*720)-2*height)):int(np.round((range_y*720)+2*height)), 
                           int(np.round((range_x*1280)-2*width)):int(np.round((range_x*1280)+2*width))] 
            cv2.imwrite("cropped_secondary_image.png", cropped)
         
        
        return cropped

    def publish_msg(self, selected_bb, box_id, secondary_object_exist, secondary_object):
        #TODO: Here normally a subtask routine has to be implemented, since pictograms on food items cannot be glued down directly.
        object_class_id = self.task_to_object(box_id)

        msg = Features()
        msg.height = selected_bb.shape[0]
        msg.width = selected_bb.shape[1]
        msg.encoding = "bgr8"
        msg.step = selected_bb.shape[1] * selected_bb.shape[2]
        msg.data = selected_bb.tobytes()
        msg.class_id = list(map(int, object_class_id))
        msg.task = box_id
        if secondary_object_exist == True:
            msg.secondary_object = secondary_object.tobytes()
        else:
            msg.secondary_object = [0]
        self.feature_eye_tracking_publisher.publish(msg)
        
    def task_to_object(self, box_id):
        if box_id == 0:     # brush
            self.logger.info(f"Gaze of Task Brush. Return object_class_id 4")
            class_id = [4] 
            return class_id
        elif box_id == 1:   # drink
            self.logger.info(f"Gaze of Task Drink. Return object_class_ids 1")
            class_id = [1]
            return class_id
        elif box_id == 2:   # eat
            self.logger.info(f"Gaze of Task Eat. Return object_class_ids 2 to 4")
            class_id = [2,4]
            return class_id
        elif box_id == 3:   # fill cup
            self.logger.info(f"Gaze of Task Fill Cup. Return object_class_ids 0")
            class_id = [0]
            return class_id
        elif box_id == 4:   # pick
            self.logger.info(f"Gaze of Task Pick. Return object_class_ids between 0 and 4")
            class_id = [0,1,2,3,4]
            #class_id = [0] # bottle
            return class_id
        elif box_id == 5:   # place
            self.logger.info(f"Gaze of Task Place. Return object_class_ids 5 for test cases")
            class_id = [0,1,2,3,4]
            #class_id = [2] # Fork
            return class_id
        elif box_id == 6:   # scratch
            self.logger.info(f"Gaze of Task Scratch. Return object_class_ids 4")
            class_id = [4] 
            return class_id
        elif box_id == 7:   # switch
            self.logger.info(f"Gaze of Task Switch. Return object_class_ids 4")
            class_id = [4]
            return class_id
        else:
            self.logger.info("Case is not a class. Doing nothing!")
        # Implementation for MS COCO
        '''
        if box_id == 0:     # brush
            self.logger.info(f"Gaze of Task Brush. Return object_class_id Brush")
            # Remark: MS Coco does not contain the class of brush. Returned is class toothbrush
            ms_coco_class_id = [80]
            return ms_coco_class_id
        elif box_id == 1:   # drink
            self.logger.info(f"Gaze of Task Drink. Return object_class_ids 41 and 42")
            ms_coco_class_id = [41, 42]
            return ms_coco_class_id
        elif box_id == 2:   # eat
            self.logger.info(f"Gaze of Task Eat. Return object_class_ids 43 to 56")
            ms_coco_class_id = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
            return ms_coco_class_id
        elif box_id == 3:   # fill cup
            self.logger.info(f"Gaze of Task Fill Cup. Return object_class_ids 40 to 42")
            # Remark: Has only to be fixed on bottles.
            ms_coco_class_id = [40]
            return ms_coco_class_id
        elif box_id == 4:   # pick
            self.logger.info(f"Gaze of Task Pick. Return object_class_ids between 25 and 80, including only graspable objects")
            ms_coco_class_id = [25, 26,27,28,29,30,35,36,37,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,64,65,6667,68,7475,76,77,78,79,80]
            return ms_coco_class_id
        elif box_id == 5:   # place
            self.logger.info(f"Gaze of Task Place. Return object_class_ids 0")
            ms_coco_class_id = [0]
            return ms_coco_class_id
        elif box_id == 6:   # scratch
            self.logger.info(f"Gaze of Task Scratch. Return object_class_ids 1")
            # This indicates detection of human. no subroutine to reduce the possibilities is given.
            ms_coco_class_id = [1]
            return ms_coco_class_id
        elif box_id == 7:   # switch
            self.logger.info(f"Gaze of Task Switch. Return object_class_ids 66")
            ms_coco_class_id = [66]
            return ms_coco_class_id
        else:
            self.logger.info("Case is not a class. Doing nothing!")
        '''

        
def main(args) -> None: 

    logger = GazeCursor.create_logger("logging.conf")
    logger.debug("logger created")
    rclpy.init(args=None)

    world_video_subscriber = WorldVideoSubscriber(callback_function=None)
    fixation_subscriber = GazeFixationSubscriber(callback_function=None)
    
    gaze_cursor = GazeCursor(
        world_video_subscriber=world_video_subscriber,
        fixation_subscriber=fixation_subscriber,
        logger = logger
    )
    bounding_box_subscriber = BoundingBoxSubscriber(gaze_cursor.task_callback, 1280, 720)
    executor = MultiThreadedExecutor()
    
    #executor.add_node(node=gaze_cursor)
    executor.add_node(node=bounding_box_subscriber)
    executor.add_node(node=world_video_subscriber)
    executor.add_node(node=fixation_subscriber)    

    try:
        executor.spin()
    except KeyboardInterrupt:
        logger.debug("Keyboard interrupt, shutting down")
        #gaze_cursor.destroy_node()
        world_video_subscriber.destroy_node()
        fixation_subscriber.destroy_node()
        bounding_box_subscriber.destroy_node()

        logger.debug("Nodes destroyed")

    logger.debug("Shutting down complete, exiting")
    exit(1)


if __name__ == "__main__":
    

    main(args = None)
