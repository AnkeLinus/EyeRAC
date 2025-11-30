from typing import Callable
import rclpy
from rclpy.node import Node
from hirac_msgs.msg import Features

class SelectedTaskSubscriber(Node):
    def __init__(
        self, 
        callback_function: Callable,
    ):
        super().__init__("selected_task_subscriber")
        self.camera_callback = callback_function

        subscription_topic = '/feature_matching_eyetracking'

        self._logger.info("Subscription to Eye tracking.")
        self.subscription = self.create_subscription(
            Features, 
            subscription_topic, 
            self._internal_feature_callback, 
            1
        )

        self.subscription
        self.latest_image = None


    def _internal_feature_callback(self,image: Features) -> None:
        if image is not None:
            if self.latest_image is None:
                print("got image")
            self.latest_image = image
        if self.camera_callback is not None:
            self.camera_callback(image)

    def get_latest_feature(self) -> Features:
        return self.latest_image



if __name__ == "__main__":
    rclpy.init()
    node = SelectedTaskSubscriber()
    print("Created node successfully")
    rclpy.shutdown()