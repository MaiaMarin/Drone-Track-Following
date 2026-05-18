import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs

class PureRealSenseSaver:
    def __init__(self, width=1280, height=720):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        
        self.align = rs.align(rs.stream.color)
        self.is_running = False

    def start(self):
        """Starts the camera pipeline with an initial hardware reset safety check."""
        # --- NEW RESET LOGIC ---
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            print("Sending hardware reset signal to RealSense device...")
            devices[0].hardware_reset()
            time.sleep(2.5)  # Crucial: Give the OS time to re-enumerate the USB device
        # ------------------------

        profile = self.pipeline.start(self.config)
        
        # Fetch the depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.is_running = True
        
        # Warm-up
        for _ in range(10):
            self.pipeline.wait_for_frames()

    def capture_pure_data(self, output_dir="pure_data", prefix="frame"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        # 1. Get the raw 16-bit integer matrix from the sensor
        raw_depth = np.asanyarray(depth_frame.get_data())
        
        # 2. Convert to PURE meters using the camera's exact hardware depth scale
        # This creates a float32 matrix where every pixel value IS the distance in meters
        depth_in_meters = raw_depth.astype(np.float32) * self.depth_scale
        
        # Get RGB matrix
        color_matrix = np.asanyarray(color_frame.get_data())

        # Timestamps for unique files
        ts = int(time.time())
        npy_path = os.path.join(output_dir, f"{prefix}_{ts}_depth.npy")
        rgb_path = os.path.join(output_dir, f"{prefix}_{ts}_rgb.png")

        # Save the pure numpy array (uncompressed, exact float32 values)
        np.save(npy_path, depth_in_meters)
        # Save the companion RGB frame
        cv2.imwrite(rgb_path, color_matrix)

        print(f"Saved pure depth matrix to: {npy_path}")
        print(f"Saved matching color frame to: {rgb_path}")

    def stop(self):
        if self.is_running:
            self.pipeline.stop()

if __name__ == "__main__":
    saver = PureRealSenseSaver()
    saver.start()
    print("Camera ready. Press Enter to capture pure metric data...")
    input()
    saver.capture_pure_data()
    saver.stop()