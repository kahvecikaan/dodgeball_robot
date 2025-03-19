"""
Camera calibration for ball-dodging robot vision system.
This module handles calibration to convert between pixel coordinates and real-world measurements.
"""

import time
import cv2
import numpy as np


def simple_calibration(camera, known_distance_cm=None, pixel_distance=None):
    """
    Simple camera calibration using a known reference measurement.

    Args:
        camera: OpenCV VideoCapture object
        known_distance_cm: Real-world distance in cm (if already known)
        pixel_distance: Distance in pixels (if already known)

    Returns:
        scale_factor: Centimeters per pixel
        camera_matrix: Simple camera matrix
        dist_coeffs: Distortion coefficients (assumed minimal)
    """
    # Get a sample frame to determine dimensions
    success, frame = camera.read()
    if not success:
        raise ValueError("Failed to capture frame from camera")

    height, width = frame.shape[:2]

    if known_distance_cm is None or pixel_distance is None:
        # Interactive calibration with live feed
        known_distance_cm, pixel_distance = interactive_calibration(camera)

    # Calculate scale factor (cm/pixel)
    scale_factor = known_distance_cm / pixel_distance

    # Create a simple camera matrix
    camera_matrix = np.array([
        [width, 0, width / 2],
        [0, height, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    # For laptop cameras, we can assume minimal distortion
    dist_coeffs = np.zeros((5, 1))

    return scale_factor, camera_matrix, dist_coeffs


def interactive_calibration(camera):
    """
    Interactive calibration where user selects two points on a reference object.
    Uses live camera feed instead of a static image.

    Args:
        camera: OpenCV VideoCapture object

    Returns:
        known_distance_cm: User-provided real-world distance
        pixel_distance: Calculated pixel distance between selected points
    """
    points = []
    current_frame = None
    display_frame = None
    point_just_added = False

    def click_event(event, x, y, flags, params):
        nonlocal display_frame, point_just_added
        if event == cv2.EVENT_LBUTTONDOWN and display_frame is not None:
            points.append((x, y))
            point_just_added = True
            # Create a copy of the current frame to draw on
            display_frame = current_frame.copy()

            # Draw all points collected so far
            for point in points:
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1)

            # If we have two points, draw a line between them
            if len(points) == 2:
                cv2.line(display_frame, points[0], points[1], (0, 255, 0), 2)
                # Display distance in pixels on the image
                mid_point = ((points[0][0] + points[1][0]) // 2,
                             (points[0][1] + points[1][1]) // 2)
                distance = np.sqrt((points[1][0] - points[0][0]) ** 2 +
                                   (points[1][1] - points[0][1]) ** 2)
                cv2.putText(display_frame, f"{distance:.1f} pixels", mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # Show the updated frame immediately
            cv2.imshow(window_name, display_frame)
            cv2.waitKey(5)  # Process GUI events

    # Create window and set mouse callback
    window_name = 'Calibration - Select two points on reference object'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)

    print("Instructions:")
    print("1. Place a ruler or measuring tape in the camera view")
    print("2. Click on two points with a known distance between them")
    print("3. Press 'ESC' to cancel or any key once you've selected two points")

    # Main loop for the live video feed
    while len(points) < 2:
        # Capture a new frame
        ret, current_frame = camera.read()
        if not ret:
            print("Failed to capture frame")
            continue

        # Create a copy to draw on
        if display_frame is None or not point_just_added:
            display_frame = current_frame.copy()

            # Redraw any existing points
            for point in points:
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1)

            # If we have two points, redraw the line
            if len(points) == 2:
                cv2.line(display_frame, points[0], points[1], (0, 255, 0), 2)
                mid_point = ((points[0][0] + points[1][0]) // 2,
                                 (points[0][1] + points[1][1]) // 2)
                distance = np.sqrt((points[1][0] - points[0][0]) ** 2 +
                                       (points[1][1] - points[0][1]) ** 2)
                cv2.putText(display_frame, f"{distance:.1f} pixels", mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Reset the flag
        point_just_added = False

        # Add instruction text
        cv2.putText(display_frame, "Place a ruler in view and click on two points",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Points selected: {len(points)}/2",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow(window_name, display_frame)

        # Check for key press to exit
        key = cv2.waitKey(30)
        if key == 27:  # ESC key
            cv2.destroyAllWindows()
            raise ValueError("Calibration cancelled by user")
        elif key != -1 and len(points) == 2:
            # If any key is pressed, and we have 2 points, break the loop
            break

    # Calculate pixel distance between the two selected points
    pixel_distance = np.sqrt((points[1][0] - points[0][0]) ** 2 +
                             (points[1][1] - points[0][1]) ** 2)

    # Capture the final frame with the points and line
    final_frame = display_frame.copy()

    # Ask user for the known distance
    cv2.putText(final_frame, "Enter the actual distance in cm in the terminal",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(window_name, final_frame)
    cv2.waitKey(500)  # Short delay so the user can see the message

    known_distance_cm = float(input("Enter the actual distance in cm: "))

    # Display the result briefly
    result_text = f"Scale factor: {known_distance_cm / pixel_distance:.6f} cm/pixel"
    cv2.putText(final_frame, result_text,
                (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow(window_name, final_frame)
    cv2.waitKey(2000)  # Show result for 2 seconds

    cv2.destroyAllWindows()
    return known_distance_cm, pixel_distance


def save_calibration(scale_factor, camera_matrix, dist_coeffs, filename='calibration_data.npz'):
    """Save calibration data to a file"""
    np.savez(filename,
             scale_factor=scale_factor,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)
    print(f"Calibration saved to {filename}")


def load_calibration(filename='calibration_data.npz'):
    """Load calibration data from a file"""
    try:
        data = np.load(filename)
        scale_factor = data['scale_factor']
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print(f"Loaded calibration data: scale_factor={scale_factor:.6f} cm/pixel")
        return scale_factor, camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None, None, None


def run_calibration():
    """Run the calibration process"""
    # Initialize camera
    camera = cv2.VideoCapture(0)  # Use laptop camera (index 0)

    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera")
        return None

    # Allow camera to warm up
    print("Initializing camera...")
    for _ in range(5):
        success, _ = camera.read()
        time.sleep(0.1)

    try:
        # Run calibration
        print("Starting calibration...")
        scale_factor, camera_matrix, dist_coeffs = simple_calibration(camera)

        # Save calibration data
        save_calibration(scale_factor, camera_matrix, dist_coeffs)

        # Release camera
        camera.release()

        print("Calibration complete!")
        print(f"Scale factor: {scale_factor:.6f} cm/pixel")

        return scale_factor, camera_matrix, dist_coeffs

    except Exception as e:
        print(f"Calibration error: {e}")
        camera.release()
        return None


if __name__ == "__main__":
    # Run calibration when this file is executed directly
    run_calibration()