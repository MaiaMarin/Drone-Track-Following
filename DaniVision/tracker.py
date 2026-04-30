import cv2
import numpy as np

class Tracker:
    def __init__(self):
        # TODO: Set this to the distance from your camera to the floor in millimeters.
        # Example: If the camera is 2 meters up, the floor is 2000mm. 
        # We look for things closer than 1800mm.
        self.floor_depth_mm = 1800 

    def get_drone_position(self, depth_image):
        # Create a mask where depth is closer than the floor, but greater than 0 (valid data)
        mask = cv2.inRange(depth_image, 1, self.floor_depth_mm)
        
        # Find objects floating above the floor
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # The drone should be the biggest thing floating
            drone_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(drone_contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), mask
                
        return None, mask