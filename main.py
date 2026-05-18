import time
import cv2
import math
from camera_module import RealsenseCamera
from perception_module import DepthTrackPlanner 
from control_module import FlightController
from radar_module import FlightRadar  
from codrone_edu.drone import *

def main():
    print("🔌 Connecting to Robolink Drone...")
    drone = Drone()
    drone.pair() 
    print("✅ Drone Paired!")

    cam = RealsenseCamera()
    perception = DepthTrackPlanner()
    controller = FlightController(hover_height_offset=0.45) 
    radar = FlightRadar()             
    
    state = "WAITING" 
    takeoff_start_time = 0.0
    print("System armed. State: WAITING. Press space to draw path.")

    try:
        while True:
            color_img, depth_frame = cam.get_frames()
            if color_img is None or depth_frame is None: continue
                
            data, debug_img = perception.process_frames(color_img, depth_frame)
            center_depth = depth_frame.get_distance(320, 240) 
            
            # --- STATE MACHINE MANAGEMENT ---
            if state == "WAITING":
                # Step 1: Once path drawing is finished, take off and wait safely
                if data["path_ready"]:
                    print("🚀 Path locked. Performing initial safe liftoff...")
                    drone.takeoff() 
                    takeoff_start_time = time.time()
                    state = "TAKEOFF_HOLD"
                    
            elif state == "TAKEOFF_HOLD":
                # Keep drone strictly locked in position while it climbs
                drone.set_roll(0)
                drone.set_pitch(0)
                drone.set_throttle(0)
                drone.move()
                
                elapsed = time.time() - takeoff_start_time
                print(f"⏱️ Establishing safe baseline hover... ({elapsed:.1f}s / 2.5s)", end="\r")
                
                # After 2.5 seconds, lock it into a permanent hover state until you click it
                if elapsed >= 2.5:
                    print("\n🛑 Hover locked! WAITING FOR SELECTION. Click the drone on screen to begin track sequence.")
                    state = "WAITING_FOR_CLICK"
                    
            elif state == "WAITING_FOR_CLICK":
                # Absolutely no horizontal or vertical movement allowed here. Just hover in place.
                drone.set_roll(0)
                drone.set_pitch(0)
                drone.set_throttle(0)
                drone.move()
                
                # The exact moment you manual left-click the drone on the screen, transition to flight!
                if data["tracking_active"] and data["drone_3d"] is not None:
                    print("\n🔒 Selection detected! Tracking active. Aligning with track heading...")
                    
                    waypoints = data["waypoints"]
                    if len(waypoints) > 5:
                        controller.set_track_alignment(waypoints[0], waypoints[5])
                    else:
                        controller.set_track_alignment(waypoints[0], waypoints[-1])
                        
                    state = "FLYING"
                    
            elif state == "FLYING":
                waypoints = data["waypoints"]
                drone_pos = data["drone_3d"]
                
                if len(waypoints) > 0:
                    if drone_pos is not None:
                        
                        # Lookahead queue trimmer
                        while len(waypoints) > 0:
                            dist = math.sqrt((waypoints[0][0] - drone_pos[0])**2 + (waypoints[0][2] - drone_pos[2])**2)
                            if dist < 0.22:
                                waypoints.pop(0)
                            else:
                                break
                        
                        if len(waypoints) > 0:
                            target_wp = waypoints[0]
                            roll, pitch, throttle = controller.calculate_velocities(drone_pos, target_wp)
                            
                            # Safe, smooth speed constraints
                            rb_roll = int(max(-18, min(roll * 100, 18)))
                            rb_pitch = int(max(-18, min(pitch * 100, 18)))
                            rb_throttle = int(max(-30, min(throttle * 100, 30)))
                            
                            drone.set_roll(rb_roll)
                            drone.set_pitch(rb_pitch)
                            drone.set_throttle(rb_throttle)
                            drone.move() 
                        else:
                            state = "LANDING"
                    else:
                        # Hold position safely if tracking drops out
                        drone.set_roll(0)
                        drone.set_pitch(0)
                        drone.set_throttle(0)
                        drone.move()
                else:
                    state = "LANDING"
                    
            elif state == "LANDING":
                print("🏁 Track complete. Executing landing sequence...")
                drone.set_roll(0)
                drone.set_pitch(0)
                drone.set_throttle(0)
                drone.land()
                state = "DONE"
                    
            elif state == "DONE":
                break
            
            radar_img = radar.draw_dashboard(data, state, center_depth)
            cv2.imshow("Flight Control Dashboard", radar_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                print("🛑 EMERGENCY STOP ACTIVATED")
                drone.emergency_stop() 
                break
            elif key == ord(' '): 
                if state == "WAITING": 
                    perception.toggle_drawing()
            
            time.sleep(0.03) 
            
    except KeyboardInterrupt:
        print("🛑 Manual override triggered. Landing...")
        drone.land()
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        drone.close() 

if __name__ == "__main__":
    main()