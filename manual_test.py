# manual_test.py

from drone_controller import DroneController
import time

DESCEND_THROTTLE  = -45
FORWARD_PITCH     = 40
TURN_YAW          = 90
SHORT_MOVE        = 0.3
TURN_MOVE         = 0.85
TINY_MOVE         = 0.0000000000001   # very little forward between turns
SMALL_MOVE        = 0.1   # a bit forward between turns

controller = DroneController()

try:
    print("connecting...")
    controller.connect()

    print("taking off...")
    controller.takeoff()
    time.sleep(1.5)

    # descend
    print("descending...")
    controller.send_command({"roll": 0, "pitch": 0, "yaw": 0, "throttle": DESCEND_THROTTLE})
    time.sleep(1.8)

    # forward
    print("forward...")
    controller.send_command({"roll": 0, "pitch": FORWARD_PITCH, "yaw": 0, "throttle": 0})
    time.sleep(SHORT_MOVE)

    # forward
    print("forward...")
    controller.send_command({"roll": 0, "pitch": FORWARD_PITCH, "yaw": 0, "throttle": 0})
    time.sleep(SHORT_MOVE)

# curve sequence - more steps, tighter turns
    print("look left...")
    controller.send_command({"roll": 0, "pitch": 0, "yaw": TURN_YAW, "throttle": 0})
    time.sleep(TURN_MOVE)

    print("tiny forward...")
    controller.send_command({"roll": 0, "pitch": FORWARD_PITCH, "yaw": 0, "throttle": 0})
    time.sleep(TINY_MOVE)

    print("look left...")
    controller.send_command({"roll": 0, "pitch": 0, "yaw": TURN_YAW, "throttle": 0})
    time.sleep(TURN_MOVE)



    # past the curve now, nudge toward the pad
    print("tiny forward to pad...")
    controller.send_command({"roll": 0, "pitch": FORWARD_PITCH, "yaw": 0, "throttle": 0})
    time.sleep(0.1)

    print("landing...")
    controller.land()

except BaseException as e:
    print("something went wrong:", e)
finally:
    try:
        controller.close()
    except BaseException:
        pass