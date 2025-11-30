from typing import Callable
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import copy

class WorldVideoSubscriber(Node):
    def __init__(
        self, 
        callback_function: Callable,
    ):
        super().__init__("world_video_subscriber")
        self.camera_callback = callback_function

        subscription_topic = '/world_video'

        self._logger.info("Subscription to Eye tracker.")
        self.subscription = self.create_subscription(
            Image, 
            subscription_topic, 
            self._internal_camera_callback, 
            1
        )

        self.subscription
        self.latest_image = None


    def _internal_camera_callback(self,image: Image) -> None:
        if image is not None:
            if self.latest_image is None:
                print("got world video image")
            self.latest_image = copy.deepcopy(image)
        if self.camera_callback is not None:
            self.camera_callback(image)

    def get_latest_image(self) -> Image:
        #print("Someone asked for world video image.")
        return copy.deepcopy(self.latest_image)



if __name__ == "__main__":
    rclpy.init()
    node = WorldVideoSubscriber()
    print("Created node successfully")
    rclpy.shutdown()