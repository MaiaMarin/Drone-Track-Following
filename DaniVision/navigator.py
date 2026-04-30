from codrone_edu.drone import *

class Navigator:
    def __init__(self):
        self.drone = Drone()
        # Proportional controller constant. 
        # If the drone overcorrects and wiggles, make this smaller!
        self.kp = 0.5 

    def connect_and_takeoff(self):
        self.drone.pair()
        self.drone.takeoff()

    def follow_track(self, drone_pos, target_pos):
        if not drone_pos or not target_pos:
            self.drone.hover()
            return
            
        drone_x, drone_y = drone_pos
        target_x, target_y = target_pos
        
        # Calculate Cross-Track Error
        error_x = target_x - drone_x
        
        # Apply Proportional Control
        roll_command = int(error_x * self.kp)
        
        # Clamp values to safe limits (-50 to 50) so it doesn't crash instantly
        roll_command = max(-50, min(50, roll_command))
        
        # Send command (Roll handles left/right, Pitch handles forward speed)
        self.drone.set_pitch(20) # Move forward constantly at 20% speed
        self.drone.set_roll(roll_command)
        self.drone.move()

    def land(self):
        self.drone.land()
        self.drone.close()