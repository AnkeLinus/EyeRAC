import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import msgpack
import zmq
import msgpack_numpy as m
import numpy as np
import cv2

class EyeTrackerVideo(Node):
    def __init__(self) -> None:
        super().__init__("world_video_publisher")
        self.get_logger().info('Initializing EyeTrackerVideo class.')
        #Initializing publisher
        self.fixation_publisher = self.create_publisher(Image, '/world_video', 10)

        # Initializing eye tracker connection
        self.get_logger().info('Initializing zmq connection with eye tracker.')
        self.PUPIL_HOST = '127.0.0.1'
        self.PORT = '50020'  # ZMQ PUB socket port
        self.SUB_PORT = '50021'  # ZMQ SUB socket port
        self.TOPIC = 'frame.world'  # You can use 'eye0', 'eye1', or 'world'

        self.ctx = zmq.Context()
        pupil_remote = zmq.Socket(self.ctx, zmq.REQ)

        pupil_remote.connect(f'tcp://{self.PUPIL_HOST}:{self.PORT}')
        print("Eyetracker connected")

        pupil_remote.send_string('SUB_PORT')
        self.sub_port = pupil_remote.recv_string()

        self.subscriber = self.ctx.socket(zmq.SUB)
        self.subscriber.connect(f'tcp://{self.PUPIL_HOST}:{self.sub_port}')
        self.subscriber.subscribe(self.TOPIC)  

        timer_period = 0.001 # less than 30 FPS
        self.timer =self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('World Video Publisher started and connected to Pupil Core.')


    def timer_callback(self):
        try:
            # Drain the ZMQ socket and keep only the latest frame
            latest_parts = None
            while self.subscriber.poll(timeout=0):  # Non-blocking poll
                latest_parts = self.subscriber.recv_multipart(flags=zmq.NOBLOCK)
            
            if latest_parts is None:
                return
            
            #    parts = self.subscriber.recv_multipart()
            img_bytes = latest_parts[2]
            nparr = np.frombuffer(img_bytes, np.uint8)

            # Decode the JPEG image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
            if frame is not None:
                # Publish msg to framework
                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "world_camera_frame"
                msg.height = frame.shape[0]
                msg.width = frame.shape[1]
                msg.encoding = "bgr8"
                msg.step = frame.shape[1]*frame.shape[2]
                msg.data = frame.tobytes()
                self.fixation_publisher.publish(msg)

                    # Show the imageq
                    #cv2.imshow("Live World Frame", frame)
            else:
                print('Failed to decode frame.')
               
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting stream.")
            #    break
                
        except Exception as e:
            print(f"Error receiving frame: {e}")
            
        