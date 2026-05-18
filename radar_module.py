import cv2
import numpy as np

class FlightRadar:
    def __init__(self, width=400, height=400):
        self.w = width
        self.h = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Scaling factor: converts physical meters into radar pixels
        # E.g., 80 pixels per meter means a point 2 meters away sits 160 pixels out
        self.pixels_per_meter = 80 

    def draw_dashboard(self, perception_data, current_state, altitude):
        """
        Generates a 2D radar/plan view mapping the manually drawn 
        3D spatial waypoints relative to the drone's position.
        """
        # Create a clean dark radar grid
        grid = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        grid[:] = (15, 15, 15) # Dark charcoal background
        
        # Draw concentric radar tracking circles (representing distance thresholds)
        for r in [50, 100, 150]:
            cv2.circle(grid, (self.center_x, self.center_y), r, (40, 40, 40), 1)
        # Radar crosshairs
        cv2.line(grid, (self.center_x, 0), (self.center_x, self.h), (30, 30, 30), 1)
        cv2.line(grid, (0, self.center_y), (self.w, self.center_y), (30, 30, 30), 1)

        # Extract waypoints safely from our new dictionary structure
        waypoints = perception_data.get("waypoints", [])
        has_path = len(waypoints) > 0

        # 1. DRAW THE REAL 3D WAYPOINT PATH ON THE RADAR
        if has_path:
            # We map the 3D points from camera coordinates onto a 2D Top-Down View
            # Camera Frame: X = Left/Right, Z = Forward Distance (Depth)
            for i in range(len(waypoints)):
                X, Y, Z = waypoints[i]
                
                # Transform meters to radar pixel offsets
                # Z (Depth) drives the vertical position, X (Horizontal) drives lateral position
                w_x = int(self.center_x + (X * self.pixels_per_meter))
                w_y = int(self.center_y - (Z * self.pixels_per_meter)) # Subtract because -Y is 'up' on a screen
                
                # Keep coordinates drawn neatly inside dashboard walls
                w_x = max(0, min(w_x, self.w - 1))
                w_y = max(0, min(w_y, self.h - 1))
                
                # Draw waypoint nodes (Cyan targets)
                cv2.circle(grid, (w_x, w_y), 4, (255, 255, 0), -1)
                
                # Interconnect the waypoints into a continuous flight track corridor
                if i > 0:
                    prev_X, prev_Y, prev_Z = waypoints[i-1]
                    pw_x = int(self.center_x + (prev_X * self.pixels_per_meter))
                    pw_y = int(self.center_y - (prev_Z * self.pixels_per_meter))
                    pw_x = max(0, min(pw_x, self.w - 1))
                    pw_y = max(0, min(pw_y, self.h - 1))
                    
                    cv2.line(grid, (pw_x, pw_y), (w_x, w_y), (0, 180, 255), 2)
                    
            # Highlight the current active target waypoint (the very first node in line)
            next_wp = waypoints[0]
            tgt_x = int(self.center_x + (next_wp[0] * self.pixels_per_meter))
            tgt_y = int(self.center_y - (next_wp[2] * self.pixels_per_meter))
            tgt_x = max(0, min(tgt_x, self.w - 1))
            tgt_y = max(0, min(tgt_y, self.h - 1))
            cv2.circle(grid, (tgt_x, tgt_y), 7, (0, 255, 255), 2) # Yellow accent tracking circle
            
        else:
            # Informative message while the pilot is still setting up/drawing
            cv2.putText(grid, "AWAITING PATH DRAWING...", (self.center_x - 110, self.center_y - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        # 2. PLOT THE DRONE (Always fixed at the center of its own radar sweep)
        drone_color = (0, 255, 0) if has_path else (0, 0, 255)
        cv2.circle(grid, (self.center_x, self.center_y), 10, drone_color, 2)
        cv2.line(grid, (self.center_x - 15, self.center_y), (self.center_x + 15, self.center_y), drone_color, 2)
        cv2.line(grid, (self.center_x, self.center_y - 15), (self.center_x, self.center_y + 15), drone_color, 2)

        # 3. OVERLAY TELEMETRY STATUS TEXT
        cv2.putText(grid, f"STATE: {current_state}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Handle depth/altitude readout safely
        alt_text = f"ALT: {altitude:.2f}m" if altitude > 0.05 else "ALT: < 0.40m (BLIND)"
        cv2.putText(grid, alt_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display direct 3D guidance data instead of raw 2D pixel error offsets
        if has_path:
            closest_wp = waypoints[0]
            err_text = f"NEXT WP: X:{closest_wp[0]:.2f}m Z:{closest_wp[2]:.2f}m"
        else:
            err_text = "NEXT WP: NO TARGET"
            
        cv2.putText(grid, err_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return grid