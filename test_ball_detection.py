"""
Ball detection test script for the ball-dodging robot vision system.
Tests both color-based and Hough circle detection methods.
"""

import cv2
import numpy as np
import time
from object_detection import detect_ball_color, detect_ball_hough, load_ball_color


def test_ball_detection(camera_index=0):
    """
    Test ball detection methods with a live camera feed.
    Displays both color-based and Hough circle detection results.

    Args:
        camera_index: Index of the camera to use
    """
    # Initialize camera
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    # Load saved ball color or use default tennis ball color (yellow-green)
    try:
        lower_color, upper_color = load_ball_color()
    except:
        # Default tennis ball color range (yellow-green)
        lower_color = np.array([25, 50, 50])  # Lower bound for tennis ball (adjust as needed)
        upper_color = np.array([65, 255, 255])  # Upper bound for tennis ball (adjust as needed)

    # Parameters for Hough circle detection
    min_radius = 10
    max_radius = 100

    # Create window
    window_name = 'Ball Detection Test'
    cv2.namedWindow(window_name)

    # Create trackbars for real-time Hough circle parameter adjustment
    cv2.createTrackbar('Min Radius', window_name, min_radius, 100, lambda x: None)
    cv2.createTrackbar('Max Radius', window_name, max_radius, 300, lambda x: None)
    cv2.createTrackbar('Param1', window_name, 100, 300, lambda x: None)
    cv2.createTrackbar('Param2', window_name, 30, 100, lambda x: None)

    # Create trackbars for color detection
    cv2.createTrackbar('H min', window_name, lower_color[0], 179, lambda x: None)
    cv2.createTrackbar('S min', window_name, lower_color[1], 255, lambda x: None)
    cv2.createTrackbar('V min', window_name, lower_color[2], 255, lambda x: None)
    cv2.createTrackbar('H max', window_name, upper_color[0], 179, lambda x: None)
    cv2.createTrackbar('S max', window_name, upper_color[1], 255, lambda x: None)
    cv2.createTrackbar('V max', window_name, upper_color[2], 255, lambda x: None)

    # Lists to store position history for trajectory visualization
    color_positions = []
    hough_positions = []
    max_history = 30  # Number of past positions to keep for trajectory

    print("Ball detection test started.")
    print("Use trackbars to adjust detection parameters in real-time.")
    print("Press 's' to save color calibration.")
    print("Press 'c' to clear trajectory history.")
    print("Press 'q' to quit.")

    while True:
        # Read frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame. Check camera connection.")
            break

        # Get current trackbar values for Hough circle detection
        min_radius = cv2.getTrackbarPos('Min Radius', window_name)
        max_radius = cv2.getTrackbarPos('Max Radius', window_name)
        param1 = cv2.getTrackbarPos('Param1', window_name)
        param2 = cv2.getTrackbarPos('Param2', window_name)

        # Get current trackbar values for color detection
        h_min = cv2.getTrackbarPos('H min', window_name)
        s_min = cv2.getTrackbarPos('S min', window_name)
        v_min = cv2.getTrackbarPos('V min', window_name)
        h_max = cv2.getTrackbarPos('H max', window_name)
        s_max = cv2.getTrackbarPos('S max', window_name)
        v_max = cv2.getTrackbarPos('V max', window_name)

        # Update color ranges
        lower_color = np.array([h_min, s_min, v_min])
        upper_color = np.array([h_max, s_max, v_max])

        # Create a copy for visualization
        display_frame = frame.copy()

        # Detect ball using color method
        ball_color, mask = detect_ball_color(frame, lower_color, upper_color)

        # Customize Hough transform parameters
        blurred = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        # Process Hough circles result
        ball_hough = None
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            if len(circles) > 0:
                circles = sorted(circles, key=lambda x: x[2], reverse=True)
                x, y, radius = circles[0]
                ball_hough = (x, y, radius)

        # Update position history
        if ball_color:
            color_positions.append(ball_color[:2])  # Store x, y only
            if len(color_positions) > max_history:
                color_positions.pop(0)

        if ball_hough:
            hough_positions.append(ball_hough[:2])  # Store x, y only
            if len(hough_positions) > max_history:
                hough_positions.pop(0)

        # Draw ball detection results on display frame
        if ball_color:
            x, y, radius = ball_color
            cv2.circle(display_frame, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_frame, f"Color: ({x}, {y}), r={radius}",
                        (x + radius + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if ball_hough:
            x, y, radius = ball_hough
            cv2.circle(display_frame, (x, y), radius, (255, 0, 0), 2)
            cv2.circle(display_frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(display_frame, f"Hough: ({x}, {y}), r={radius}",
                        (x + radius + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw trajectories
        if len(color_positions) > 1:
            for i in range(1, len(color_positions)):
                # Gradient color based on position in trajectory (newer = brighter)
                alpha = i / len(color_positions)
                color = (0, int(255 * alpha), 0)
                cv2.line(display_frame,
                         (color_positions[i - 1][0], color_positions[i - 1][1]),
                         (color_positions[i][0], color_positions[i][1]),
                         color, 2)

        if len(hough_positions) > 1:
            for i in range(1, len(hough_positions)):
                # Gradient color based on position in trajectory (newer = brighter)
                alpha = i / len(hough_positions)
                color = (int(255 * alpha), 0, 0)
                cv2.line(display_frame,
                         (hough_positions[i - 1][0], hough_positions[i - 1][1]),
                         (hough_positions[i][0], hough_positions[i][1]),
                         color, 2)

        # Add legend
        cv2.putText(display_frame, "Green Circle: Color Detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Blue Circle: Hough Detection", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Add current HSV color range
        hsv_text = f"HSV Range: [{h_min},{s_min},{v_min}] to [{h_max},{s_max},{v_max}]"
        cv2.putText(display_frame, hsv_text, (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display frames
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow(window_name, display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit
            break
        elif key == ord('s'):
            # Save color calibration
            from object_detection import save_ball_color
            save_ball_color(lower_color, upper_color)
            print("Saved current color calibration.")
        elif key == ord('c'):
            # Clear trajectory history
            color_positions = []
            hough_positions = []
            print("Cleared trajectory history.")

    # Clean up
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_ball_detection()