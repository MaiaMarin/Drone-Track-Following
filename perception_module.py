import cv2
import numpy as np

class DepthTrackPlanner:
    def __init__(self, intrinsic_matrix=None):
        self.window_name = "RealSense D455 - Depth Track Planner"
        cv2.namedWindow(self.window_name)

        self.is_drawing = False
        self.current_mouse_pos = (320, 240)
        self.pixel_path = []      
        self.waypoint_path = []  
        
        # --- Pure Motion Tracking States ---
        self.tracking_initialized = False
        self.last_drone_pixel = None  
        self.search_window_radius = 60  # Box size that follows the motion
        self.prev_gray = None           # Memory frame to compute differences

        # Set up mouse callback ONLY for drawing the path
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        if intrinsic_matrix is not None:
            self.fx, self.fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
            self.cx, self.cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        else:
            self.fx, self.fy = 385.0, 385.0  
            self.cx, self.cy = 320.0, 240.0

    def mouse_callback(self, event, x, y, flags, param):
        self.current_mouse_pos = (x, y)

    def toggle_drawing(self):
        if not self.is_drawing:
            self.is_drawing = True
            self.pixel_path = []
            self.waypoint_path = []
            self.tracking_initialized = False
            self.last_drone_pixel = None
            print("✏️ Started drawing... Trace your track layout.")
        else:
            self.is_drawing = False
            print("📍 Path locked! System armed. Preparing for auto-takeoff...")

    def deproject_pixel_to_3d(self, x, y, depth_meters):
        X = (x - self.cx) * depth_meters / self.fx
        Y = (y - self.cy) * depth_meters / self.fy
        Z = depth_meters
        return (X, Y, Z)

    def process_frames(self, color_image, depth_frame):
        if color_image is None or depth_frame is None:
            return {"path_ready": False, "waypoints": [], "drone_3d": None}, color_image

        debug_img = color_image.copy()
        h, w, _ = color_image.shape
        self.cx, self.cy = w / 2.0, h / 2.0

        current_waypoints = []
        drone_3d_pos = None

        # Convert frame to grayscale and blur out camera sensor static noise
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return {"path_ready": False, "waypoints": [], "drone_3d": None}, color_image

        # --- TEMPORAL DIFFERENCING ENGINE ---
        frame_diff = cv2.absdiff(gray, self.prev_gray)
        _, motion_mask = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)

        window_motion_mask = np.zeros_like(motion_mask)

        # --- AUTOMATIC DRONE INITIALIZATION ON DECOLLATE ---
        if not self.tracking_initialized:
            # If path is locked, we scan the WHOLE frame for the burst of movement from takeoff
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                # Look for an initial takeoff movement footprint (50 to 800 pixel cluster)
                if 50 < area < 800:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Verify it has a valid depth to prevent false locking on screen edge glitches
                        if hasattr(depth_frame, 'get_distance'):
                            d_check = depth_frame.get_distance(cx, cy)
                        else:
                            d_check = depth_frame[cy, cx] * 0.001

                        if 0.2 < d_check < 4.0:
                            self.last_drone_pixel = (cx, cy)
                            self.tracking_initialized = True
                            print(f"🎯 AUTOMATIC LOCK ENGAGED! Drone detected at takeoff spot: ({cx}, {cy})")
                            break

        # --- CONTINUOUS TRACKING LOOP ---
        if self.tracking_initialized and self.last_drone_pixel is not None:
            lx, ly = self.last_drone_pixel
            r = self.search_window_radius
            
            x_min, x_max = max(0, lx - r), min(w, lx + r)
            y_min, y_max = max(0, ly - r), min(h, ly + r)
            
            # Isolate movement strictly inside our adaptive tracking window
            window_motion_mask[y_min:y_max, x_min:x_max] = motion_mask[y_min:y_max, x_min:x_max]
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            window_motion_mask = cv2.morphologyEx(window_motion_mask, cv2.MORPH_CLOSE, kernel)

            cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

            contours, _ = cv2.findContours(window_motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cx, cy = self.last_drone_pixel
            
            best_contour = None
            largest_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 15 < area < 2000: 
                    if area > largest_area:
                        largest_area = area
                        best_contour = contour

            if best_contour is not None:
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

            self.last_drone_pixel = (cx, cy)
            
            if hasattr(depth_frame, 'get_distance'):
                d_depth = depth_frame.get_distance(cx, cy)
            else:
                d_depth = depth_frame[cy, cx] * 0.001 

            if d_depth <= 0.1:
                y_safe = max(0, min(cy, h-1))
                x_safe = max(0, min(cx, w-1))
                d_depth = depth_frame[y_safe, x_safe] * 0.001 if not hasattr(depth_frame, 'get_distance') else depth_frame.get_distance(x_safe, y_safe)

            if 0.2 < d_depth < 5.0:
                drone_3d_pos = self.deproject_pixel_to_3d(cx, cy, d_depth)
                cv2.circle(debug_img, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(debug_img, f"DRONE Z:{d_depth:.2f}m", (cx - 30, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        self.prev_gray = gray.copy()

        # --- PATH DRAWING SYSTEM ---
        if self.is_drawing:
            x, y = self.current_mouse_pos
            x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
            if not self.pixel_path or self.pixel_path[-1] != (x, y):
                self.pixel_path.append((x, y))

        if len(self.pixel_path) > 1:
            for i in range(len(self.pixel_path)):
                x, y = self.pixel_path[i]
                if hasattr(depth_frame, 'get_distance'):
                    depth_meters = depth_frame.get_distance(x, y)
                else:
                    depth_meters = depth_frame[y, x] * 0.001 

                current_waypoints.append(self.deproject_pixel_to_3d(x, y, depth_meters if depth_meters > 0.1 else 1.5))
                color = (0, 255, 0) if depth_meters > 0.1 else (128, 128, 128)
                cv2.circle(debug_img, (x, y), 2, color, -1)
                if i > 0:
                    px, py = self.pixel_path[i-1]
                    cv2.line(debug_img, (px, py), (x, y), color, 2)

            if not self.is_drawing:
                self.waypoint_path = current_waypoints

        # --- HUD OVERLAYS ---
        if self.is_drawing:
            cv2.putText(debug_img, f"DRAWING WAYPOINTS ({len(self.pixel_path)}) - Space to Stop", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        elif len(self.waypoint_path) > 0 and not self.tracking_initialized:
            cv2.putText(debug_img, "⚡ PATH ARMED. WAITING FOR LIFT-OFF MOVEMENT...", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        elif self.tracking_initialized:
            cv2.putText(debug_img, "🔒 AUTO-MOTION LOCK ACTIVE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imshow(self.window_name, debug_img)
        
        payload = {
            "path_ready": not self.is_drawing and len(self.waypoint_path) > 0,
            "tracking_active": self.tracking_initialized,
            "waypoints": self.waypoint_path,
            "drone_3d": drone_3d_pos 
        }
        return payload, debug_img