import logging

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from cv_bridge import CvBridge
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image

from hirac_msgs.msg import BoundingBox
from hirac_msgs.msg import BoundingBoxArray

# configure
CONFIDENCE_THRESHOLD = 0.5

bridge = CvBridge()

class ObjectDetection(Node):
    def __init__(
        self,
        basic_logger: logging.Logger,
        object_detection_model: YOLO,
        object_tracker: DeepSort,
    ):
        super().__init__("object_detection_publisher")
        self._basic_logger = basic_logger
        
        self._model = object_detection_model
        self._tracker = object_tracker

        self.bounding_box_array_msg = BoundingBoxArray()

        self.bounding_box_publisher = self.create_publisher(
            BoundingBoxArray, "/hirac_object_detection", 1
        )

    def camera_callback(self, image: Image) -> None:
        if image is None:
            self._basic_logger.debug("Received None as image")
            return

        self._basic_logger.debug("Recieved image")
        cv2_image = bridge.imgmsg_to_cv2(image, "bgr8")
        self._basic_logger.debug("Converted image")
        detected_objects = self.detect_objects(cv2_image)
        self._basic_logger.debug("Detected %d objects", len(detected_objects))

        tracked_objects = self.track_objects(cv2_image, detected_objects)
        self._basic_logger.debug("Found %d tracked objects", len(tracked_objects))
        self._basic_logger.debug(tracked_objects)

        # create boundingBoxMessage
        self.bounding_box_array_msg.image = image
        for tracked_object in tracked_objects:
            bounding_box = BoundingBox()
            bounding_box.class_id = tracked_object[0]
            bounding_box.object_id = int(tracked_object[1])
            bounding_box.x = tracked_object[2][0]
            bounding_box.y = tracked_object[2][1]
            bounding_box.width = tracked_object[2][2]
            bounding_box.height = tracked_object[2][3]
            self.bounding_box_array_msg.bounding_boxes.append(bounding_box)

        self.bounding_box_publisher.publish(self.bounding_box_array_msg)
        self.bounding_box_array_msg.bounding_boxes.clear()



    def detect_objects(self, image: np.ndarray) -> list[dict]:

        resized = cv2.resize(image, (640, 480))
        detections = self._model.predict(
            source=resized, save=False, imgsz=640, conf=0.75
        )[0]

       
        # initialize the list of bounding boxes and confidences
        results: list[dict]
        results = []

        # loop over the detections

        for image in detections.boxes.data.tolist():
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = image[4]

            # filter out weak detections by ensuring the
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # if the confidence is greater than the minimum confidence,
            # get the bounding box and the class id
            xmin, ymin, xmax, ymax = (
                int(image[0]),
                int(image[1]),
                int(image[2]),
                int(image[3]),
            )
            class_id = int(image[5])
            # add the bounding box (x, y, w, h), confidence and class id to the results list
            results.append(
                [[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id]
            )
        return results

    def track_objects(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        results: list[dict]
        results = []
        tracks = self._tracker.update_tracks(detections, frame=frame)

        # loop over the tracks
        for track in tracks:
            # if the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue

            # get the track id and the bounding box
            track_id = int(track.track_id)
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()

            xmin, ymin, xmax, ymax = (
                float(ltrb[0]),
                float(ltrb[1]),
                float(ltrb[2]),
                float(ltrb[3]),
            )
            results.append([class_id, track_id, [xmin, ymin, xmax - xmin, ymax - ymin]])
        return results