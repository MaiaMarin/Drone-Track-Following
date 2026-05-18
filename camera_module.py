import pyrealsense2 as rs
import numpy as np
import sys

class RealsenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable both color and depth streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        try:
            # Start streaming
            self.profile = self.pipeline.start(self.config)
            print("✅ RealSense D455 connected and streaming successfully.")
        except RuntimeError as e:
            print("\n❌ ERROR: Intel RealSense device could not be found!")
            print("👉 Check that the camera is plugged into a USB 3.0 port using a high-speed data cable.")
            print("👉 If using a Mac dongle, ensure it supports high-bandwidth USB 3.0 data transfer.\n")
            sys.exit(1) # Exit the program cleanly
        
        # Align depth frame to color frame
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def get_frames(self):
        """Returns aligned color image and depth frame."""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
                
            color_image = np.asanyarray(color_frame.get_data())
            return color_image, depth_frame
        except Exception as e:
            print(f"Error retrieving frames: {e}")
            return None, None

    def stop(self):
        try:
            self.pipeline.stop()
        except:
            pass