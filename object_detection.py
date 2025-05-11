"""
Object detection for ball-dodging robot vision system.
Handles detection of the ball and robot in camera frames.
"""

import cv2
import numpy as np
from playground_setup import transform_point


def detect_ball_color(frame, lower_color, upper_color, min_radius=5, max_radius=100):
    """
    Detect a ball in the frame based on color thresholding.

    Args:
        frame: Camera frame (BGR)
        lower_color: Lower bound of color range in HSV
        upper_color: Upper bound of color range in HSV
        min_radius: Minimum radius in pixels to consider a valid ball
        max_radius: Maximum radius in pixels to consider a valid ball

    Returns:
        ball_pos: (x, y, radius) of the detected ball, or None if not detected
        mask: Binary mask showing the detected ball
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create binary mask where pixels within the specified range become white and all others become black
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return None
    if not contours:
        return None, mask

    # Find the best ball candidate based on circularity and size
    best_ball = None
    best_score = 0

    for contour in contours:
        # Skip tiny contours
        if cv2.contourArea(contour) < np.pi * min_radius ** 2:
            continue

        # Find the minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(contour)

        # Skip contours that are too large
        if radius > max_radius:
            continue

        # Calculate circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Calculate area ratio (area of contour / area of enclosing circle)
        # A perfect circle would have a ratio close to 1
        circle_area = np.pi * radius ** 2
        area_ratio = area / circle_area if circle_area > 0 else 0

        # Combined score based on circularity and area ratio
        # Higher is better, perfect ball would have score close to 2
        score = circularity + area_ratio

        if score > best_score and circularity > 0.7:  # Increased circularity threshold
            best_score = score
            best_ball = (int(x), int(y), int(radius))

    # Return the best ball candidate if one was found
    if best_ball and best_score > 1.3:  # Threshold for accepting a ball
        return best_ball, mask

    return None, mask


def detect_robot(frame, aruco_dict, parameters, robot_marker_id=42, homography_matrix=None):
    """
    Detect the robot's position using its ArUco marker.
    For 1D movement, we only need to track the x-coordinate.

    Args:
        frame: Camera frame
        aruco_dict: ArUco dictionary
        parameters: ArUco detection parameters
        robot_marker_id: ID of the marker on the robot
        homography_matrix: Homography matrix for coordinate transformation

    Returns:
        robot_pos: Position of the robot in pixel coordinates (x, y, orientation)
                  or playground coordinates if homography_matrix is provided
        visualized_frame: Frame with robot marker visualized
    """
    # Make a copy for visualization
    visualized_frame = frame.copy()

    # Convert to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create and use ArUco detector with the updated API
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Draw detected markers on the visualized frame
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(visualized_frame, corners, ids)

    # Check if robot marker is detected
    if ids is None or robot_marker_id not in ids.flatten():
        return None, visualized_frame

    # Get robot marker
    idx = np.where(ids.flatten() == robot_marker_id)[0][0]
    robot_corners = corners[int(idx)][0]

    # Calculate center point
    center_x = np.mean(robot_corners[:, 0])
    center_y = np.mean(robot_corners[:, 1])

    # Calculate orientation (angle in radians)
    # Using the first two corners of the marker
    dx = robot_corners[1][0] - robot_corners[0][0]
    dy = robot_corners[1][1] - robot_corners[0][1]
    orientation = np.arctan2(dy, dx)

    # Add robot visualization
    cv2.circle(visualized_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
    end_x = int(center_x + 30 * np.cos(orientation))
    end_y = int(center_y + 30 * np.sin(orientation))
    cv2.line(visualized_frame, (int(center_x), int(center_y)), (end_x, end_y), (0, 0, 255), 2)

    # If homography matrix is provided, transform to playground coordinates
    if homography_matrix is not None:
        playground_coords = transform_point((center_x, center_y), homography_matrix)
        # For 1D robot, we only need x-coordinate in the playground
        return (playground_coords[0], playground_coords[1], orientation), visualized_frame

    return (center_x, center_y, orientation), visualized_frame


def ball_color_calibration(camera_index=0):
    """
    Interactive tool to calibrate ball color detection.

    Args:
        camera_index: Camera index to use

    Returns:
        lower_color: Lower HSV bounds for ball detection
        upper_color: Upper HSV bounds for ball detection
    """
    # Initialize camera
    camera = cv2.VideoCapture(camera_index)

    # Default color range for a tennis ball
    lower_color = np.array([25, 50, 50])
    upper_color = np.array([65, 255, 255])

    # Create a trackbars window
    cv2.namedWindow('Ball Color Calibration')

    # Create trackbars
    cv2.createTrackbar('H min', 'Ball Color Calibration', int(lower_color[0]), 179, lambda _: None)
    cv2.createTrackbar('S min', 'Ball Color Calibration', int(lower_color[1]), 255, lambda _: None)
    cv2.createTrackbar('V min', 'Ball Color Calibration', int(lower_color[2]), 255, lambda _: None)
    cv2.createTrackbar('H max', 'Ball Color Calibration', int(upper_color[0]), 179, lambda _: None)
    cv2.createTrackbar('S max', 'Ball Color Calibration', int(upper_color[1]), 255, lambda _: None)
    cv2.createTrackbar('V max', 'Ball Color Calibration', int(upper_color[2]), 255, lambda _: None)

    print("Ball color calibration started.")
    print("Place the ball in the camera view and adjust sliders until only the ball is visible in the mask.")
    print("Press 'q' to save and exit, 'r' to reset to defaults.")

    while True:
        # Read frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame. Check camera connection.")
            break

        # Get current trackbar values
        h_min = cv2.getTrackbarPos('H min', 'Ball Color Calibration')
        s_min = cv2.getTrackbarPos('S min', 'Ball Color Calibration')
        v_min = cv2.getTrackbarPos('V min', 'Ball Color Calibration')
        h_max = cv2.getTrackbarPos('H max', 'Ball Color Calibration')
        s_max = cv2.getTrackbarPos('S max', 'Ball Color Calibration')
        v_max = cv2.getTrackbarPos('V max', 'Ball Color Calibration')

        # Update color ranges
        lower_color = np.array([h_min, s_min, v_min])
        upper_color = np.array([h_max, s_max, v_max])

        # Detect ball with current settings
        ball_pos, mask = detect_ball_color(frame, lower_color, upper_color)

        # Draw detected ball position
        result = frame.copy()
        if ball_pos:
            x, y, radius = ball_pos
            cv2.circle(result, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

        # Display original, mask, and result
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Save and exit
            break
        elif key == ord('r'):
            # Reset to defaults
            cv2.setTrackbarPos('H min', 'Ball Color Calibration', 5)
            cv2.setTrackbarPos('S min', 'Ball Color Calibration', 100)
            cv2.setTrackbarPos('V min', 'Ball Color Calibration', 100)
            cv2.setTrackbarPos('H max', 'Ball Color Calibration', 15)
            cv2.setTrackbarPos('S max', 'Ball Color Calibration', 255)
            cv2.setTrackbarPos('V max', 'Ball Color Calibration', 255)

    # Clean up
    camera.release()
    cv2.destroyAllWindows()

    print(f"Ball color calibration completed:")
    print(f"Lower bounds: H={lower_color[0]}, S={lower_color[1]}, V={lower_color[2]}")
    print(f"Upper bounds: H={upper_color[0]}, S={upper_color[1]}, V={upper_color[2]}")

    return lower_color, upper_color


def save_ball_color(lower_color, upper_color, filename='ball_color.npz'):
    """Save ball color calibration to file"""
    np.savez(filename, lower_color=lower_color, upper_color=upper_color)
    print(f"Ball color calibration saved to {filename}")


def load_ball_color(filename='ball_color.npz'):
    """Load ball color calibration from file"""
    try:
        data = np.load(filename)
        lower_color = data['lower_color']
        upper_color = data['upper_color']
        print(f"Loaded ball color calibration:")
        print(f"Lower bounds: H={lower_color[0]}, S={lower_color[1]}, V={lower_color[2]}")
        print(f"Upper bounds: H={upper_color[0]}, S={upper_color[1]}, V={upper_color[2]}")
        return lower_color, upper_color
    except Exception as e:
        print(f"Error loading ball color calibration: {e}")
        print("Using default tennis ball color range")
        return np.array([25, 50, 50]), np.array([65, 255, 255])


if __name__ == "__main__":
    # When run directly, launch the ball color calibration tool
    lower_clr, upper_clr = ball_color_calibration()
    save_ball_color(lower_clr, upper_clr)