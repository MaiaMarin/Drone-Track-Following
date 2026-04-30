import pyrealsense2 as rs
import numpy as np

class Scanner:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure D455 streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Align depth to color so pixels match perfectly
        self.align = rs.align(rs.stream.color)

    def start(self):
        self.pipeline.start(self.config)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image

    def stop(self):
        self.pipeline.stop()