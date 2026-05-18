import time
import math

class FlightController:
    def __init__(self, hover_height_offset=0.45):
        self.hover_height_offset = hover_height_offset 
        
        # Lower, smoother gains to eliminate the high-frequency flinching
        self.Kp_roll = 0.35  
        self.Kd_roll = 0.08
        
        self.Kp_pitch = 0.35 
        self.Kd_pitch = 0.08 

        self.Kp_throttle = 0.50

        self.last_error_forward = 0.0
        self.last_error_right = 0.0 
        self.last_time = time.time()
        
        self.smooth_roll = 0.0
        self.smooth_pitch = 0.0
        self.smooth_throttle = 0.0
        
        # Track heading angle (calculated automatically at launch)
        self.track_angle = 0.0 

    def set_track_alignment(self, start_wp, lookahead_wp):
        """Calculates the exact direction the drone is facing along the track"""
        dx = lookahead_wp[0] - start_wp[0]
        dz = lookahead_wp[2] - start_wp[2]
        self.track_angle = math.atan2(dz, dx)
        print(f"📐 Track Alignment Locked: {math.degrees(self.track_angle):.1f}° relative to camera.")

    def calculate_velocities(self, drone_3d, target_wp):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.01

        # 1. Raw errors in Camera Frame
        error_x = target_wp[0] - drone_3d[0]
        error_z = target_wp[2] - drone_3d[2]

        target_y = target_wp[1] - self.hover_height_offset
        error_y = drone_3d[1] - target_y 

        # 2. ROTATION MATRIX: Translate Camera Frame into Drone Body Frame
        # Aligns pitch with the track direction and roll with lateral corrections
        error_forward = error_x * math.cos(self.track_angle) + error_z * math.sin(self.track_angle)
        error_right = error_x * math.sin(self.track_angle) - error_z * math.cos(self.track_angle)

        # 3. Derivatives based on drone frame errors
        derivative_forward = (error_forward - self.last_error_forward) / dt
        derivative_right = (error_right - self.last_error_right) / dt

        # PID Outputs
        raw_pitch = (self.Kp_pitch * error_forward) + (self.Kd_pitch * derivative_forward)
        raw_roll = (self.Kp_roll * error_right) + (self.Kd_roll * derivative_right)
        raw_throttle = (self.Kp_throttle * error_y) 

        self.last_error_forward = error_forward
        self.last_error_right = error_right
        self.last_time = current_time
        
        # High smoothing blend to ensure natural, fluid transitions
        self.smooth_roll = (0.6 * self.smooth_roll) + (0.4 * raw_roll)
        self.smooth_pitch = (0.6 * self.smooth_pitch) + (0.4 * raw_pitch)
        self.smooth_throttle = (0.8 * self.smooth_throttle) + (0.2 * raw_throttle)

        return self.smooth_roll, self.smooth_pitch, self.smooth_throttle