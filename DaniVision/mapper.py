
import cv2
import numpy as np

class Mapper:
    def __init__(self):
        # TODO: You will need to tune these HSV values for the wooden track in your room!
        self.lower_wood = np.array([10, 50, 50]) 
        self.upper_wood = np.array([30, 255, 255])

    def get_target_point(self, color_image):
        # Convert to HSV for better color masking
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_wood, self.upper_wood)
        
        # Find the shapes that match the wood
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get the biggest piece of track
            biggest_track = max(contours, key=cv2.contourArea)
            M = cv2.moments(biggest_track)
            
            if M["m00"] != 0:
                # Calculate center X, Y
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), mask
                
        return None, mask