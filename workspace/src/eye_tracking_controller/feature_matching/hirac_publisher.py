from typing import Callable
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from hirac_msgs.msg import BoundingBox as BoundingBoxMsg, BoundingBoxArray as BoundingBoxArrayMsg, Task as TaskMsg

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

class SelectedBoundingBoxPublisher(Node):
    def __init__(self)-> None:
        super().__init__('selected_bb_publisher')
        self.selected_bb_publisher = self.create_publisher(BoundingBoxMsg, '/selected_bb', 10)

    def bb_selected(self, bb:BoundingBox)-> None:
        
        bounding_box = bb.get_bounding_box_msg()        
        self.selected_bb_publisher.publish(bounding_box)

class SelectedTaskPublisher(Node):
    def __init__(self)-> None:
        super().__init__('selected_task_publisher')
        self.selected_task_publisher = self.create_publisher(TaskMsg, '/selected_task', 10)

    def task_selected(self, taskmsg:String)-> None:
        
        task = TaskMsg()
        task.task = taskmsg        
        self.selected_task_publisher.publish(task)

class SelectedSecondaryObjectPublisher(Node):
    def __init__(self)-> None:
        super().__init__('selected_secondary_object_publisher')
        self.selected_bb_publisher = self.create_publisher(BoundingBoxMsg, '/selected_secondary_object', 10)

    def bb_secondary_object(self, bb:BoundingBox)-> None:
        bounding_box = bb.get_bounding_box_msg()        
        self.selected_bb_publisher.publish(bounding_box)