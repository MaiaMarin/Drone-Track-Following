from codrone_edu.drone import Drone
import time

FORWARD_PITCH = 8
ROLL_SCALE = 22
MAX_ROLL = 18
MOVE_DURATION = 0.10
PAUSE_DURATION = 0.02

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def direction_to_command(direction):
    if direction is None:
        return {
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
            "throttle": 0
        }

    dy = direction["dy"]

    roll = int(dy * ROLL_SCALE)
    roll = clamp(roll, -MAX_ROLL, MAX_ROLL)

    return {
        "roll": roll,
        "pitch": FORWARD_PITCH,
        "yaw": 0,
        "throttle": 0
    }

class DroneController:
    def __init__(self):
        self.drone = Drone()
        self.is_flying = False

    def connect(self):
        self.drone.pair()
        time.sleep(2)

    def takeoff(self):
        self.drone.takeoff()
        time.sleep(3)
        self.is_flying = True

    def send_command(self, command):
        self.drone.set_roll(command["roll"])
        self.drone.set_pitch(command["pitch"])
        self.drone.set_yaw(command["yaw"])
        self.drone.set_throttle(command["throttle"])
        self.drone.move(MOVE_DURATION)

        self.drone.set_roll(0)
        self.drone.set_pitch(0)
        self.drone.set_yaw(0)
        self.drone.set_throttle(0)
        self.drone.move(PAUSE_DURATION)

    def send_direction(self, direction):
        command = direction_to_command(direction)
        self.send_command(command)

    def hover(self):
        self.drone.set_roll(0)
        self.drone.set_pitch(0)
        self.drone.set_yaw(0)
        self.drone.set_throttle(0)
        self.drone.move(0.10)

    def land(self):
        if not self.is_flying:
            return

        self.hover()
        self.drone.land()
        time.sleep(2)
        self.is_flying = False

    def close(self):
        try:
            if self.is_flying:
                self.land()
            self.drone.close()
        except BaseException:
            pass