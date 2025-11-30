import zmq
import time as t

''' 
Important: Pupil Core Capture has to be started before starting this program.
'''

run_test =2
subscription_topic = 'fixation'

ctx = zmq.Context()
# The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
pupil_remote = zmq.Socket(ctx, zmq.REQ)

#ip = 'localhost'  # If you talk to a different machine use its IP.
ip = "120.0.0.1"
port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.

pupil_remote.connect(f'tcp://{ip}:{port}')
print("Eyetracker connected")


if run_test == 1:

    ###Test 1: Recording
    # start recording
    pupil_remote.send_string('R')
    print("Asked to subscribe")
    print(pupil_remote.recv_string())

    t.sleep(5)
    pupil_remote.send_string('r')
    print(pupil_remote.recv_string())

else:
    ### Test 2: Using IPC Backbone
    # Request 'SUB_PORT' for reading data
    pupil_remote.send_string('SUB_PORT')
    sub_port = pupil_remote.recv_string()
    print("successful subscription")

    # Request 'PUB_PORT' for writing data
    #pupil_remote.send_string('PUB_PORT')
    #pub_port = pupil_remote.recv_string()


    #...continued from above
    # Assumes `sub_port` to be set to the current subscription port
    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(f'tcp://{ip}:{sub_port}')
    subscriber.subscribe(subscription_topic)  # receive all gaze messages
    print("Subscribed")
    # we need a serializer
    import msgpack

    while True:
        topic, payload = subscriber.recv_multipart()
        message = msgpack.loads(payload)
        print(f"{topic}: {message}")
        norm_pos = message[b'norm_pos']
        print(f"{norm_pos[0], norm_pos[1]}")
    