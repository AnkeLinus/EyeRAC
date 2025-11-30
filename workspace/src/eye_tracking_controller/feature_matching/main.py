'''
This package uses bounding boxes from the eye tracker and from the robot mounted camera to find the best 
match between objects and therefore can differenciate and tell the robot which object was selected.
Since it was found, that this approach might have a high number of incorrect predictions, a logic was 
implemented reducing the risk of wrong predictions: 

Logic:
    The bounding box cutouts from the eye tracker are compared to the bouning box cutouts of the robot, when
    the selected class is predicted at least once in the robot's field of view. (Matrix-Cross chekc of feature 
    matching predicted the same class as YOLO)
    The bounding box cutouts from the eye tracker are compared to the complete robot field of view, in the case 
    the class is not predicted once. The highest cluster is used to define the nearest found bouning box. (Case 
    wrongfully labeled class: Most features matched in bb wins. Case not-labeld boundingbox: ...(no solution till 
    now. Check after implementation, if this is an actual problem or not.))

    Last and new case: What happens, when the bounding box is not predicted in the robot field of view and there
    is empty space?

The result should be that wrongfully labeled bounding boxes can be used, wrongfully predictions of the feature 
matching can be detected, and non-detections of objects can be improved by reducing the searched space.

Inputs:
    - Video Stream Robot
    - bounding boxes robot
    - bounding boxes eye tracker

Output:
    - Selected Bounding box robot
    - Best for complete data transfer to HIRAC Interface: Selected Task
'''

'''
Idea: Using only pictograms to find the most correct match. Quite easy to implement. As a result a matrix should spawn in which the highest number of features matched is hinting into the correct direction.
'''
import logging
import logging.config
import os

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from general.camera import CameraSubscriber
from brain.brain_node.camera_info_subscriber import CameraInfoSubscriber
from general.selected_task_subscriber import SelectedTaskSubscriber
from general.robot_bb_subscriber import BoundingBoxSubscriberBB, BoundingBox
from eye_tracking_controller.feature_matching.hirac_publisher import SelectedBoundingBoxPublisher
from eye_tracking_controller.feature_matching.hirac_publisher import SelectedTaskPublisher
from eye_tracking_controller.feature_matching.hirac_publisher import SelectedSecondaryObjectPublisher
from typing import List

from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from matplotlib import pyplot
from collections import Counter

import cv2 as cv
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import time
import numpy as np

import pygame
import datetime
#ts = time.time()
#timestp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

class FeatureMatchingAKAZE(Node):

    def __init__( # Todo: this is copied from task action server. Correct and fit to necessary packages.
        self,
        logger:logging.Logger,
        camera_subscriber: CameraSubscriber,
        camera_info_subscriber: CameraInfoSubscriber,
        robot_bb_subscriber = BoundingBoxSubscriberBB,
        selected_bb_publisher = SelectedBoundingBoxPublisher,
        selected_task_publisher = SelectedTaskPublisher,
        selected_sec_obj_publisher = SelectedSecondaryObjectPublisher,
        akaze = cv.AKAZE_create()
    ):
        super().__init__("feature_matching_node")
        self.camera_subscriber = camera_subscriber
        self.camera_info_subscriber = camera_info_subscriber
        self.robot_bb_subscriber = robot_bb_subscriber
        self.selected_bb_publisher = selected_bb_publisher
        self.selected_task_publisher = selected_task_publisher
        self.selected_sec_obj_publisher = selected_sec_obj_publisher
        self.akaze = akaze

        #Fine-tuning parameters
        self.ratio_cross_comparison = 0.75
        self.ratio_image_search = 0.75
        self.threshold_min_features_matched = 5

        self.logger = logger
        self.bridge = CvBridge()

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = FLANN_INDEX_LSH,  
                            table_number=6,      # number of hash tables (e.g., 6–12)
                            key_size=12,         # hash key size in bits (e.g., 12–20)
                            multi_probe_level=1) # how many nearby buckets to check (higher=more recall)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv.FlannBasedMatcher(index_params,search_params)

    def create_logger(config_name: str) -> logging.Logger:
        # configure logging
        current_dir = os.path.dirname(__file__)
        logging_config_path = os.path.join(current_dir, config_name)
        print("Trying to load logging config from %s", logging_config_path)
        logging.config.fileConfig(fname=logging_config_path, disable_existing_loggers=False)
        logger = logging.getLogger(" Feature Matcher")
        return logger


    def ratio_test_flann(self, matches):
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        good = []
        # ratio test as per Lowe's paper
        for i,m_n in enumerate(matches):
            if len(m_n) < 2:  # Skip if less than 2 neighbors
                continue
            m, n = m_n
            if m.distance < self.ratio_cross_comparison*n.distance:
                matchesMask[i]=[1,0]
                good.append([m])
        return good, matchesMask
    
    def ratio_test_flann_image_search(self, matches):
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        good = []
        # ratio test as per Lowe's paper
        for i,m_n in enumerate(matches):
            if len(m_n) < 2:  # Skip if less than 2 neighbors
                continue
            m, n = m_n
            if m.distance < self.ratio_image_search*n.distance:
                matchesMask[i]=[1,0]
                good.append([m])
        return good, matchesMask

    def plot_flann(self, matches, matchesMask, pic1, kp1, pic2, kp2):
        draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
        img2 = cv.drawMatchesKnn(pic1, kp1, pic2, kp2,matches,None,**draw_params)
        cv.imwrite("plot_FLANN.png", img2)
        #plt.imshow(img2),plt.show()

    def plot_flann_no_match_bb(self, matches, matchesMask, pic1, kp1, pic2, kp2):
        draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
        img2 = cv.drawMatchesKnn(pic1, kp1, pic2, kp2,matches,None,**draw_params)
        cv.imwrite("plot_FLANN_no_match.png", img2)
        #plt.imshow(img2),plt.show()

    def cut_bb(self, image, bb):
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        
        # Crop the image using numpy slicing
        cropped = cv_image[int(np.round((bb.y/480*720))):int(np.round(((bb.y+bb.height)/480*720))), 
                           int(np.round((bb.x/620*1280))):int(np.round(((bb.x+bb.width)/620*1280)))] 
        return cropped
    
    def publish_info(self, selected_bb, task, secondary_object):
        #self.logger.info("sending data to hirac_interface")
        self.logger.info(f"Bounding Box Class ID: {selected_bb.object_class_id}")
        self.logger.info(f"Task information ID {task}")
        self.logger.info(f"Secondary Object Information {secondary_object}")

        self.selected_bb_publisher.bb_selected(selected_bb)

        hirac_task = self.task_translator(task)
        self.selected_task_publisher.task_selected(hirac_task)
        
        if task == 3:
            self.selected_sec_obj_publisher.bb_secondary_object(secondary_object)

        # Task selection duration Measurement
        #ts = time.time()
        #timestp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:3]
        #self.logger.info(f"Task selection was successful published: {timestp}")

        pygame.init()
        pygame.mixer.init()
        plays = pygame.mixer.Sound("/usr/share/sounds/gnome/default/alerts/glass.ogg")
        plays.play()
        

        
    def task_translator(self, task):
        #0: "Task-Brush", 1:	"Task-Drink", 2: "Task-Eat", 3:	"Task-Fill-Cup", 4: "Task-Pick", 5: "Task-Place", 6: "Task-Scratch", 7: "Task-Switch"
        if task == 0:
            return "brush"
        elif task == 1:
            return "drink"
        elif task == 2:
            return "eat"
        elif task == 3:
            return "grab to pour"
        elif task == 4:
            return "pick"
        elif task == 5:
            return "place selected"
        elif task == 6:
            return "scratch"
        elif task == 7:
            return "switch"
        else:
             self.logger.info("Task classes are initiated incorrectly.")

    def find_secodary_object(self, image, cropped_et_secondary_object, bounding_box_array_robot):
        self.logger.info("Secondary Object needed. Starting finding it in robot scene.")
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        
        # Feature detection
        kp_et_sec, descriptors_et_sec = self.akaze.detectAndCompute(cropped_et_secondary_object, None)
        kp_robot_sec, descriptors_robot_sec = self.akaze.detectAndCompute(cv_image, None)

        # Feature matching
        matches = []
        matches.append(self.flann.knnMatch(descriptors_et_sec,descriptors_robot_sec,k=2))

        for i in range(len(matches)):
            good, matches_mask =  self.ratio_test_flann(matches[i])

        # Finding the closest bounding box to the biggest cluster.
        # define dataset
        X = []
        for match in good:
            for m in match:
                pt = kp_robot_sec[m.trainIdx].pt  # (x, y) tuple
                X.append([pt[0], pt[1]])

        X = np.array(X)
        if len(X) < self.threshold_min_features_matched:
            self.logger.info("Not enough matches for clustering")
            return
        # define the model
        model = KMeans(n_clusters=(len(bounding_box_array_robot)+1))
        # fit the model
        model.fit(X)
        # assign a cluster to each example
        yhat = model.predict(X)
        # retrieve unique clusters
        clusters = unique(yhat)
        # create scatter plot for samples from each cluster
        for cluster in clusters:
            # get row indexes for samples with this cluster
            row_ix = where(yhat == cluster)
            # create scatter of these samples
            pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
        # show the plot
        pyplot.show()

        cluster_counts = Counter(yhat)
        biggest_cluster_label = cluster_counts.most_common(1)[0][0]
        points_in_biggest_cluster = X[yhat == biggest_cluster_label]

        # Find nearest bounding box over mean center
        mean_biggest_cluster = np.mean(points_in_biggest_cluster, axis=0)

        mean_bb = []
        for bb in bounding_box_array_robot:
            center = [bb.x + bb.width / 2, bb.y + bb.height]
            mean_bb.append(center)
        mean_bb = np.array(mean_bb)  # shape: (N, 2)

        # 3. Compute Euclidean distances to all bounding boxes
        distances = np.linalg.norm(mean_bb - mean_biggest_cluster, axis=1)

        # 4. Get index and value of closest bounding box
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        nearest_bb = bounding_box_array_robot[min_index]  # This is the bounding box you want

        return nearest_bb


    ####  AKAZE approach   ####
    #### Feature detection ####
        
    def match_selected_object(self, update: List[BoundingBox]) -> None:
        bounding_box_array_robot = self.robot_bb_subscriber.get_latest_robot_bb()
        bounding_box_selected_obj = update
        self.task = update.task
        self.secondary_object = update.secondary_object
        image = self.camera_subscriber.get_latest_image()
        cropped_et_bounding_box = np.frombuffer(update.data, dtype=np.uint8).reshape((update.height, update.width, 3))
        if update.secondary_object ==[0]:
            cropped_et_secondary_object = np.frombuffer(update.secondary_object, dtype=np.uint8).reshape((update.height, update.width, 3))

        bb_classes = []
        for bb in bounding_box_array_robot:
            bb_classes.append(bb.object_class_id)

        ## Case 1: bounding box classes exist in both images.
        if set(bb_classes) & set(bounding_box_selected_obj.class_id):
            self.logger.info("There is a common object! Using cross comparison.")
            descriptors_robot = []
            kp_robot = []

            # cutout bounding boxes when not already done
            it = 0
            cropped_robot_bb_array = []
            for bounding_box in bounding_box_array_robot:
                cropped_robot_bounding_box = self.cut_bb(image, bounding_box)
                cropped_robot_bb_array.append(cropped_robot_bounding_box)
                # Image output as needed
                #safe_name = "robot_image"+str(it)+".png"
                #cv.imwrite(safe_name, cropped_robot_bounding_box)
                #it = it + 1

            # Feature detection
            kp_et, descriptors_et = self.akaze.detectAndCompute(cropped_et_bounding_box,None)

            for cropped_img in cropped_robot_bb_array:
                keypoint, descriptor = self.akaze.detectAndCompute(cropped_img,None)
                kp_robot.append(keypoint)
                descriptors_robot.append(descriptor) 

            # Feature matching
            matches_all = []
            self.logger.info(f"Features in ET: {len(descriptors_et)}")
            for n in descriptors_robot:
                if n is not None and len(n) >= 2:
                    self.logger.info(f"Features in Robot: {len(n)}")
                    matches = self.flann.knnMatch(descriptors_et,n,k=2)
                    matches_all.append(matches)
                else:
                    matches = []
                    matches_all.append(matches)

            good_all = []
            matches_mask = []

            for match in matches_all:
                good_ratio, matches_mask_ratio =  self.ratio_test_flann(match)
                good_all.append(good_ratio)
                matches_mask.append(matches_mask_ratio)

            # Finding the highest number of matches:
            max_val = []
            for good in good_all:
                max_val.append(len(good))
            max_index = max_val.index(max(max_val))
            potential_robot_object = bounding_box_array_robot[max_index]

            #Plotting data
            self.plot_flann(matches_all[max_index], matches_mask[max_index], cropped_et_bounding_box, kp_et, cropped_robot_bb_array[max_index], kp_robot[max_index])

            if (potential_robot_object.object_class_id in bounding_box_selected_obj.class_id) and len(good_all[max_index]) >= self.threshold_min_features_matched:
                self.logger.info("Objects have the same class. High confidence that it's the same object.")
                if update.task == 3:
                    self.secondary_object = self.find_secodary_object(image, cropped_et_secondary_object, bounding_box_array_robot)

                self.publish_info(potential_robot_object, self.task, self.secondary_object)
            else:
                self.logger.info("Objects don't have the same class. Low confidence that it's the same object. Repeat the measurement.")

        # Case 2: Bounding Box classes don't fit.
        else:
            self.logger.info("There is no common object! Comparing the robot's image with the received bounding box. Calculating the heaviest cluster and matching it to the nearest bounding box.")
            cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
            
            # Feature detection
            kp_et, descriptors_et = self.akaze.detectAndCompute(cropped_et_bounding_box, None)
            kp_robot, descriptors_robot = self.akaze.detectAndCompute(cv_image, None)

            # Feature matching
            match = self.flann.knnMatch(descriptors_et,descriptors_robot,k=2)
            self.logger.info(f"Match is {len(match)} long.")
            good, match_mask =  self.ratio_test_flann_image_search(match)
            self.logger.info(f"Good is {len(good)} long.")

            #self.plot_flann_no_match_bb(match, match_mask, cropped_et_bounding_box, kp_et, cv_image, kp_robot)


            # Finding the closest bounding box to the biggest cluster.
            # define dataset
            X = []
            for match_i in good:
                #self.logger.info(f"Match is {match}.")
                pt = kp_robot[match_i[0].trainIdx].pt  # (x, y) tuple
                X.append([pt[0], pt[1]])

            self.logger.info(f"X is {X}.")
            X = np.array(X)
            if len(X) < (len(bounding_box_array_robot)+1):
                self.logger.info("Not enough matches for clustering")
                return
            else:
                self.logger.info(f"X is {len(X)} long.")
                # define the model
                model = KMeans(n_clusters=(len(bounding_box_array_robot)+1))
                # fit the model
                model.fit(X)
                # assign a cluster to each example
                yhat = model.predict(X)
                # retrieve unique clusters
                clusters = unique(yhat)
                # create scatter plot for samples from each cluster
                for cluster in clusters:
                    # get row indexes for samples with this cluster
                    row_ix = where(yhat == cluster)
                    # create scatter of these samples
                    #pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
                # show the plot
                #pyplot.show()

                cluster_counts = Counter(yhat)
                biggest_cluster_label = cluster_counts.most_common(1)[0][0]
                points_in_biggest_cluster = X[yhat == biggest_cluster_label]

                # Find nearest bounding box over mean center
                mean_biggest_cluster = np.mean(points_in_biggest_cluster, axis=0)

                mean_bb = []
                for bb in bounding_box_array_robot:
                    center = [bb.x + bb.width / 2, bb.y + bb.height]
                    mean_bb.append(center)
                mean_bb = np.array(mean_bb)  # shape: (N, 2)

                # 3. Compute Euclidean distances to all bounding boxes
                distances = np.linalg.norm(mean_bb - mean_biggest_cluster, axis=1)

                # 4. Get index and value of closest bounding box
                min_index = np.argmin(distances)
                min_distance = distances[min_index]
                nearest_bb = bounding_box_array_robot[min_index]  # This is the bounding box you want

                # Optional debug
                self.logger.info(f"Nearest bounding box is #{min_index} at distance {min_distance}")
                if update.task == 3:
                    self.secondary_object = self.find_secodary_object(image, cropped_et_secondary_object, bounding_box_array_robot)

                self.publish_info(nearest_bb, self.task, self.secondary_object)


        
        
def main(args) -> None: # Todo: Copied from task action server. Fix and fit to needs.

    logger = FeatureMatchingAKAZE.create_logger("logging.conf")
    logger.debug("logger created")
    rclpy.init(args=None)

    camera_subscriber = CameraSubscriber(callback_function=None)
    camera_info_subscriber = CameraInfoSubscriber()
    robot_bb_subscriber = BoundingBoxSubscriberBB()
    selected_bb_publisher = SelectedBoundingBoxPublisher()
    selected_task_publisher = SelectedTaskPublisher()
    selected_sec_obj_publisher = SelectedSecondaryObjectPublisher()
    
    feature_matcher = FeatureMatchingAKAZE(
        robot_bb_subscriber = robot_bb_subscriber,
        camera_subscriber = camera_subscriber,
        camera_info_subscriber = camera_info_subscriber,
        selected_bb_publisher = selected_bb_publisher,
        selected_task_publisher = selected_task_publisher,
        selected_sec_obj_publisher = selected_sec_obj_publisher,
        logger = logger
    )
    selected_task = SelectedTaskSubscriber(feature_matcher.match_selected_object)
    executor = MultiThreadedExecutor()
    
    executor.add_node(node=selected_task)
    executor.add_node(node=robot_bb_subscriber)
    executor.add_node(node=camera_subscriber)
    executor.add_node(node=camera_info_subscriber)    
    executor.add_node(node=selected_bb_publisher)
    executor.add_node(node=selected_task_publisher)
    executor.add_node(node=selected_sec_obj_publisher)

    try:
        executor.spin()
    except KeyboardInterrupt:
        logger.debug("Keyboard interrupt, shutting down")
        #gaze_cursor.destroy_node()
        camera_subscriber.destroy_node()
        camera_info_subscriber.destroy_node()
        robot_bb_subscriber.destroy_node()
        selected_task.destroy_node()

        logger.debug("Nodes destroyed")

    logger.debug("Shutting down complete, exiting")
    exit(1)

if __name__ == "__main__":
    

    main(args = None)
