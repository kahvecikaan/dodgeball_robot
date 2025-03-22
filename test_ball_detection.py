"""
Simplified ball detection test script for the ball-dodging robot vision system.
Uses color-based detection optimized for tennis balls with fixed color values.
"""

import cv2
import numpy as np
from object_detection import detect_ball_color


def test_ball_detection(camera_index=0):
    """
    Test color-based ball detection with a live camera feed.
    Uses fixed tennis ball color values and displays detection results with trajectory.

    Args:
        camera_index: Index of the camera to use
    """
    # Initialize camera
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    # Fixed tennis ball color range (yellow-green)
    lower_color = np.array([25, 50, 50])  # Lower bound for tennis ball
    upper_color = np.array([65, 255, 255])  # Upper bound for tennis ball

    # Create window
    window_name = 'Tennis Ball Detection'
    cv2.namedWindow(window_name)

    # List to store position history for trajectory visualization
    ball_positions = []
    max_history = 30  # Number of past positions to keep for trajectory

    print("Tennis ball detection test started.")
    print("Using fixed HSV color range for tennis ball detection.")
    print("Press 'c' to clear trajectory history.")
    print("Press 'q' to quit.")

    while True:
        # Read frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame. Check camera connection.")
            break

        # Create a copy for visualization
        display_frame = frame.copy()

        # Detect ball using color method
        ball_pos, mask = detect_ball_color(frame, lower_color, upper_color)

        # Update position history if ball is detected
        if ball_pos:
            ball_positions.append(ball_pos[:2])  # Store x, y only
            if len(ball_positions) > max_history:
                ball_positions.pop(0)

            # Draw ball detection results
            x, y, radius = ball_pos
            cv2.circle(display_frame, (x, y), radius, (0, 255, 0), 2)  # Ball outline
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)  # Center point

            # Display coordinates and radius
            cv2.putText(display_frame, f"Ball: ({x}, {y}), r={radius}",
                        (x + radius + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw trajectory
            if len(ball_positions) > 1:
                for i in range(1, len(ball_positions)):
                    # Gradient color based on position in trajectory (newer = brighter)
                    alpha = i / len(ball_positions)
                    color = (0, int(255 * alpha), 0)
                    cv2.line(display_frame,
                             (ball_positions[i - 1][0], ball_positions[i - 1][1]),
                             (ball_positions[i][0], ball_positions[i][1]),
                             color, 2)

        # Add title and instructions
        cv2.putText(display_frame, "Tennis Ball Detector", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show current HSV color range (now fixed)
        hsv_text = f"HSV Range: [{lower_color[0]},{lower_color[1]},{lower_color[2]}] to [{upper_color[0]},{upper_color[1]},{upper_color[2]}]"
        cv2.putText(display_frame, hsv_text, (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display frames
        cv2.imshow('Original', frame)
        # cv2.imshow('Mask', mask)
        cv2.imshow(window_name, display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit
            break
        elif key == ord('c'):
            # Clear trajectory history
            ball_positions = []
            print("Cleared trajectory history.")

    # Clean up
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_ball_detection()