import zmq
import msgpack_numpy as m
import numpy as np
import cv2


# Enable numpy support in msgpack
m.patch()


# Settings
PUPIL_HOST = '127.0.0.1'
PORT = '50020'  # ZMQ PUB socket port
SUB_PORT = '50021'  # ZMQ SUB socket port
TOPIC = 'frame.world'  # You can use 'eye0', 'eye1', or 'world'


def receive_frames():
    ctx = zmq.Context()
    # The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
    pupil_remote = zmq.Socket(ctx, zmq.REQ)


    pupil_remote.connect(f'tcp://{PUPIL_HOST}:{PORT}')
    print("Eyetracker connected")


    pupil_remote.send_string('SUB_PORT')
    sub_port = pupil_remote.recv_string()


    #...continued from above
    # Assumes `sub_port` to be set to the current subscription port
    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(f'tcp://{PUPIL_HOST}:{sub_port}')
    subscriber.subscribe(TOPIC)  # receive all gaze messages


    while True:
        try:
            while True:
                parts = subscriber.recv_multipart()
                img_bytes = parts[2]
                nparr = np.frombuffer(img_bytes, np.uint8)


                # Decode the JPEG image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # use IMREAD_GRAYSCALE if you expect gray
                if frame is not None:
                    # Show the imageq
                    cv2.imshow("Live World Frame", frame)
                else:
                    print('Failed to decode frame.')
               
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting stream.")
                    break
               
        except Exception as e:
            print(f"Error receiving frame: {e}")
            break


if __name__ == '__main__':
    print(f"Subscribing to Pupil {TOPIC} video stream...")
    receive_frames()