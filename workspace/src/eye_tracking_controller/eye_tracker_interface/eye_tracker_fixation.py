import rclpy
from rclpy.node import Node
from hirac_msgs.msg import Fixation

import msgpack
import zmq

class EyeTrackerFixation(Node):
    def __init__(self) -> None:
        super().__init__("fixation_publisher")
        self.get_logger().info('EyeTrackerFixation class.')
        # Initializing eye tracker connection
        self.get_logger().info('Initializing zmq connection with eye tracker.')
        self.PUPIL_HOST = '127.0.0.1'
        self.PORT = '50020'  # ZMQ PUB socket port
        self.SUB_PORT = '50021'  # ZMQ SUB socket port
        self.TOPIC = 'fixation'  # You can use 'eye0', 'eye1', or 'world'

        ctx = zmq.Context()
        pupil_remote = zmq.Socket(ctx, zmq.REQ)

        pupil_remote.connect(f'tcp://{self.PUPIL_HOST}:{self.PORT}')
        #print("Eyetracker connected")

        pupil_remote.send_string('SUB_PORT')
        sub_port = pupil_remote.recv_string()

        self.subscriber = ctx.socket(zmq.SUB)
        self.subscriber.connect(f'tcp://{self.PUPIL_HOST}:{sub_port}')
        self.subscriber.subscribe(self.TOPIC) 

        self.fixation_publisher = self.create_publisher(Fixation, '/gaze_fixation', 10)
        self.get_logger().info('Gaze Fixation Publisher has started')
        timer_period = 0.01
        self.timer =self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('EyeTrackerFixation Publisher has started')
        
    def timer_callback(self):
        try:
            topic, payload = self.subscriber.recv_multipart()
            message = msgpack.loads(payload)
            norm_pos = message['norm_pos']
            msg = Fixation()
            msg.norm_pos_x = norm_pos[0]
            msg.norm_pos_y = norm_pos[1]
            self.fixation_publisher.publish(msg)
        
        except Exception as e:
            print(f"Error receiving fixation: {e}")
        