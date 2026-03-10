import numpy as np

test_environment = {
    "id": "7656679b-25fe-4af5-a19d-73e7df16f384",
    "name": "Home Setup",
    "robots": [
        {
            "robot": {
                "id": "c3f3f886-8813-4b3b-ba48-165cdaa39995",
                "name": "Khaos",
                "connection_string": "",
                "serial_number": "5AA9017083",
                "type": "SO101_Follower",
            },
            "tele_operator": {"type": "none"},
        }
    ],
    "cameras": [
        {
            "id": "3ed60255-04ae-407b-8e2c-c3281847a4e0",
            "driver": "usb_camera",
            "name": "grabber",
            "fingerprint": "/dev/video0:0",
            "hardware_name": None,
            "payload": {"width": 640, "height": 480, "fps": 30},
        },
        {
            "id": "4629e172-2aa7-4fde-86b1-e19eb1d210ff",
            "driver": "usb_camera",
            "name": "front",
            "fingerprint": "/dev/video6:6",
            "hardware_name": None,
            "payload": {"width": 640, "height": 480, "fps": 30},
        },
    ],
}

test_observation = {
    'shoulder_pan.pos': -11.076923076923077,
    'shoulder_lift.pos': 56.043956043956044,
    'elbow_flex.pos': -10.197802197802197,
    'wrist_flex.pos': 69.45054945054945,
    'wrist_roll.pos': -24.791208791208792,
    'gripper.pos': 12.364425162689804,
    '3ed60255-04ae-407b-8e2c-c3281847a4e0': np.zeros([480, 640, 3], dtype=np.uint8),
    '4629e172-2aa7-4fde-86b1-e19eb1d210ff': np.zeros([480, 640, 3], dtype=np.uint8)
}
