from codrone_edu.drone import Drone
import time

drone = Drone()
drone.pair()

drone.takeoff()
time.sleep(2)
drone.land()

drone.close()