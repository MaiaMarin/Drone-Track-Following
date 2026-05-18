import cv2
import numpy as np
from camera_module import RealsenseCamera

def empty(a):
    pass

def main():
    cam = RealsenseCamera()
    
    # Create the interactive slider window
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 500, 300)
    
    # Initialize sliders with a very broad range so you can see things initially
    cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "Trackbars", 40, 179, empty)
    cv2.createTrackbar("Sat Min", "Trackbars", 10, 255, empty)
    cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "Trackbars", 20, 255, empty)
    cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

    print("🔧 Calibration Mode Active. Adjust sliders to isolate the track.")
    print("Press 'q' when finished to print your final values.")

    try:
        while True:
            color_img, _ = cam.get_frames()
            if color_img is None:
                continue
                
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            
            # Read current slider positions
            h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
            h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
            s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
            s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
            v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
            v_max = cv2.getTrackbarPos("Val Max", "Trackbars")
            
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            
            # Apply mask
            mask = cv2.inRange(hsv, lower, upper)
            
            # Show live results
            cv2.imshow("Original Camera Feed", color_img)
            cv2.imshow("Live Mask Calibration", mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n✅ CALIBRATION COMPLETE! Copy these lines into perception_module.py:")
                print(f"self.WOOD_LOW = np.array([{h_min}, {s_min}, {v_min}])")
                print(f"self.WOOD_HIGH = np.array([{h_max}, {s_max}, {v_max}])\n")
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()

if __name__ == "__main__":
    main()