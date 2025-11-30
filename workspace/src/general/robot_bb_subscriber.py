from typing import Callable
import rclpy
from rclpy.node import Node
from hirac_msgs.msg import BoundingBox as BoundingBoxMsg, BoundingBoxArray as BoundingBoxArrayMsg


class BoundingBox:
    """Represents a bounding box"""

    def __init__(
        self,
        id: int,
        x: int,
        y: int,
        width: int,
        height: int,
        object_class_id: int,
        image_width: int,
        image_height: int,
    ) -> None:
        self.id = id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.object_class_id = object_class_id
        self.image_width = image_width
        self.image_height = image_height

    def get_dto(self) -> dict:
        return {
            "id": self.id,
            "xInPercent": self.x / self.image_width * 100,
            "yInPercent": self.y / self.image_height * 100,
            "widthInPercent": self.width / self.image_width * 100,
            "heightInPercent": self.height / self.image_height * 100,
            "boxClass": self.object_class_id,
        }
    
    
    def get_bounding_box_msg(self) -> BoundingBoxMsg:
        bounding_box = BoundingBoxMsg()        
        bounding_box.class_id = self.id
        bounding_box.object_id = self.object_class_id
        bounding_box.x = self.x
        bounding_box.y = self.y
        bounding_box.width = self.width
        bounding_box.height = self.height

        return bounding_box
    
class BoundingBoxSubscriberBB(Node):
    def __init__(self) -> None:
        super().__init__("object_detection_subscriber")
        self.image_width = 640
        self.image_height = 480
        self.subscription = self.create_subscription(
            BoundingBoxArrayMsg,
            "/hirac_object_detection",
            self.bounding_box_array_callback,
            1,  # Set the queue size as needed
        )
        self.subscription  # Prevent unused variable warning

    def bounding_box_array_callback(self, msg: BoundingBoxArrayMsg) -> None:
        # Process the received bounding box message here

        # self.get_logger().info("Got  %d bounding boxes" % (len(msg.bounding_boxes)))

        self.bounding_boxes = self.convert_bounding_boxes_to_dictionary(
            msg.bounding_boxes
        )
        
        self.latest_robot_bb = self.bounding_boxes

    def convert_bounding_boxes_to_dictionary(
        self, bounding_boxes: list[BoundingBoxMsg]
    ) -> list[BoundingBox]:
        # self.get_logger().info("converting %d" % len(bounding_boxes))

        boxes: list[BoundingBox]
        boxes = []

        for box in bounding_boxes:
            # self.get_logger().info(f"class_id: {box.class_id} object_id:{box.object_id}")

            boxes.append(
                BoundingBox(
                    box.object_id,
                    box.x,
                    box.y,
                    box.width,
                    box.height,
                    box.class_id,
                    self.image_width,
                    self.image_height,
                )
            )
        return boxes
    
    def get_latest_robot_bb(self) -> BoundingBoxArrayMsg:
        print("someone asked for camerainfo")
        return self.latest_robot_bb
'''
    def start(self) -> None:
        self.get_logger().debug("Starting")
        rclpy.spin(self)
        self.get_logger().debug("Started")

    def stop(self) -> None:
        self.get_logger().debug("Stopping")
        self.destroy_node()
        self.context.try_shutdown()
        self.get_logger().debug("Stopped")
'''