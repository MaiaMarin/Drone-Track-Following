import cv2
import numpy as np

class Tracker:
    def __init__(
        self,
        floor_depth_mm=2500,
        min_height_above_floor_mm=250,
        min_depth_mm=200,
        min_area=400,
        max_area=50000
    ):
        self.floor_depth_mm = floor_depth_mm
        self.min_height_above_floor_mm = min_height_above_floor_mm
        self.min_depth_mm = min_depth_mm
        self.min_area = min_area
        self.max_area = max_area

    def get_drone_position(self, depth_image):
        max_drone_depth = self.floor_depth_mm - self.min_height_above_floor_mm

        # Mask objects above the floor (depth smaller = closer to camera)
        mask = cv2.inRange(depth_image, self.min_depth_mm, max_drone_depth)

        # Stronger smoothing (helps with angled noisy depth)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                candidates.append(contour)

        if not candidates:
            return None, mask

        # Pick the largest valid blob (assumed drone)
        best = max(candidates, key=cv2.contourArea)
        moments = cv2.moments(best)

        if moments["m00"] == 0:
            return None, mask

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        return (cx, cy), mask