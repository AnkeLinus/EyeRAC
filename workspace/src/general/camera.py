from typing import Callable
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from ros_hirac_camera_topic import script

class CameraSubscriber(Node):
    def __init__(
        self, 
        callback_function: Callable,
    ):
        super().__init__("camera_subscriber")
        self.camera_callback = callback_function

        subscription_topic = script.get_image_topic()

        if "/camera/color/image_raw" in subscription_topic:   
            self._logger.info("Subscription to Realsense Image.")
            self.subscription = self.create_subscription(
                Image, 
                subscription_topic, 
                self._internal_camera_callback, 
                10
            )

        elif subscription_topic == '/oak/rgb/image_raw':
            self._logger.info("Subscription to Oak-D Image.")
            self.subscription = self.create_subscription(
                Image, 
                subscription_topic, 
                self._internal_camera_callback, 
                10)
        
        elif subscription_topic == '/right/image_rect':
            self._logger.info("Subscription to Oak-D Image.")
            self.subscription = self.create_subscription(
                Image, 
                subscription_topic, 
                self._internal_camera_callback, 
                10)
    
        else:
            self.get_logger().info('Could not receive Image. Topic Issues.')

        self.subscription
        self.latest_image = None


    def _internal_camera_callback(self,image: Image) -> None:
        if image is not None:
            if self.latest_image is None:
                print("got image")
            self.latest_image = image
        if self.camera_callback is not None:
            self.camera_callback(image)

    def get_latest_image(self) -> Image:
        return self.latest_image



if __name__ == "__main__":
    rclpy.init()
    node = CameraSubscriber()
    print("Created node successfully")
    rclpy.shutdown()