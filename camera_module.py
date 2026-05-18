import pyrealsense2 as rs
import numpy as np

class RGBDCamera:
    def __init__(self, width=640, height=480, fps=30):
        """Initializes the RealSense camera pipeline and aligns depth to color."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable both streams with matching resolutions
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Creating an align object to align depth frames to the color frame's viewport
        self.align = rs.align(rs.stream.color)
        self.is_running = False

    def start(self):
        """Starts the camera sensor stream."""
        if not self.is_running:
            self.pipeline.start(self.config)
            self.is_running = True

    def get_frames(self):
        """
        Retrieves synchronized and aligned RGB and Depth frames.
        Returns:
            success (bool): True if frames successfully retrieved.
            color_image (np.ndarray): BGR image matrix.
            depth_frame_raw (np.ndarray): 16-bit raw depth data (in millimeters).
        """
        if not self.is_running:
            return False, None, None
            
        # Wait for a coherent pair of frames (depth and color)
        frames = self.pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return False, None, None
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return True, color_image, depth_image

    def stop(self):
        """Safely stops the camera stream."""
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
