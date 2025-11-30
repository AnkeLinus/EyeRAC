from typing import Callable
import rclpy
from rclpy.node import Node
from hirac_msgs.msg import Fixation
import copy

class GazeFixationSubscriber(Node):
    def __init__(
        self, 
        callback_function: Callable,
    ):
        super().__init__("fixation_subscriber")
        self.camera_callback = callback_function

        subscription_topic = '/gaze_fixation'

        self._logger.info("Subscription to Eye tracking.")
        self.subscription = self.create_subscription(
            Fixation, 
            subscription_topic, 
            self._internal_camera_callback, 
            1
        )

        self.subscription
        self.latest_image = None


    def _internal_camera_callback(self,image: Fixation) -> None:
        if image is not None:
            if self.latest_image is None:
                print("got fixation")
            self.latest_image = copy.deepcopy(image)

    def get_latest_fixation(self) -> Fixation:
        #print("Someone asked for fixation data")
        return copy.deepcopy(self.latest_image)



if __name__ == "__main__":
    rclpy.init()
    node = GazeFixationSubscriber()
    print("Created node successfully")
    rclpy.shutdown()