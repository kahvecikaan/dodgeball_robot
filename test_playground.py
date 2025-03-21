import cv2
import numpy as np
import time
from playground_setup import create_aruco_dict, detect_aruco_markers, transform_point, inverse_transform_point


def test_playground_coordinates(camera_index=0):
    """
    Interactive test for playground coordinate system with visualization.

    Args:
        camera_index: Index of camera to use
    """
    # Initialize camera
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    # Create ArUco dictionary and parameters
    aruco_dict, parameters = create_aruco_dict()

    # Create window and set mouse callback
    window_name = 'Playground Coordinate Test'
    cv2.namedWindow(window_name)

    # Variables to store coordinate system data
    homography_matrix = None
    marker_centers = None
    playground_dims = None

    # Variables to store clicked points
    clicked_points = []

    # Define mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))

    # Set up the mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)

    # Instructions
    print("\nPlayground Coordinate System Test")
    print("=================================")
    print("SETUP INSTRUCTIONS:")
    print("1. Place the four ArUco markers (IDs 0, 1, 2, 3) in a rectangular configuration")
    print("2. The markers should form a rectangle or square on a flat surface")
    print("3. Make sure to note the EXACT distances between marker centers in centimeters")
    print("   - Distance from Marker 0 to Marker 1 (top width)")
    print("   - Distance from Marker 0 to Marker 3 (left height)")

    # Flag for setup state
    is_setup_complete = False

    # Main loop
    while True:
        # Capture frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame. Check camera connection.")
            break

        # Display original frame
        display_frame = frame.copy()

        # Detect ArUco markers
        corners, ids, frame_markers = detect_aruco_markers(frame, aruco_dict, parameters)

        # If markers detected but setup not complete
        if ids is not None and not is_setup_complete:
            # Convert ids to a flat array
            ids_flat = ids.flatten()

            # Check if all four corner markers are visible
            required_markers = [0, 1, 2, 3]
            if all(marker_id in ids_flat for marker_id in required_markers):
                # Extract marker centers
                marker_centers = []
                for marker_id in required_markers:
                    # Find index of this marker ID
                    idx = np.where(ids_flat == marker_id)[0][0]
                    # Calculate center by averaging the 4 corners
                    marker_center = np.mean(corners[idx][0], axis=0)
                    marker_centers.append(marker_center)

                # Draw centers and IDs
                for i, center in enumerate(marker_centers):
                    center_x, center_y = int(center[0]), int(center[1])
                    cv2.circle(display_frame, (center_x, center_y), 5, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"ID: {required_markers[i]}",
                                (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw lines connecting the markers
                for i in range(4):
                    pt1 = (int(marker_centers[i][0]), int(marker_centers[i][1]))
                    pt2 = (int(marker_centers[(i + 1) % 4][0]), int(marker_centers[(i + 1) % 4][1]))
                    cv2.line(display_frame, pt1, pt2, (0, 255, 255), 2)

                # Ask for real-world dimensions if not already provided
                if playground_dims is None:
                    cv2.imshow(window_name, display_frame)
                    cv2.waitKey(1)

                    print("\nAll four markers detected! Now we need the real-world dimensions.")
                    print("Please measure the EXACT distances between marker CENTERS in centimeters.")

                    try:
                        width = float(input("Enter the width (distance from marker 0 to marker 1) in cm: "))
                        height = float(input("Enter the height (distance from marker 0 to marker 3) in cm: "))
                        playground_dims = (width, height)

                        # Define playground corners in real-world coordinates (cm)
                        playground_corners_real = np.array([
                            [0, 0],  # Top-left (ID 0)
                            [width, 0],  # Top-right (ID 1)
                            [width, height],  # Bottom-right (ID 2)
                            [0, height]  # Bottom-left (ID 3)
                        ], dtype=np.float32)

                        # Calculate homography matrix
                        marker_centers_array = np.array(marker_centers, dtype=np.float32)
                        homography_matrix, _ = cv2.findHomography(marker_centers_array, playground_corners_real)

                        is_setup_complete = True
                        print("\nSetup complete! Playground coordinate system established.")
                        print(f"Playground dimensions: {width}cm x {height}cm")
                        print("\nInteraction Instructions:")
                        print("- Click anywhere in the image to mark points and see coordinates")
                        print("- Press 'c' to clear all marked points")
                        print("- Press 'r' to recalibrate the coordinate system")
                        print("- Press 's' to take a snapshot")
                        print("- Press 'q' to quit")

                    except ValueError:
                        print("Invalid input. Please enter numeric values.")

        # If setup is complete, process clicked points and draw coordinate grid
        if is_setup_complete:
            # Draw coordinate grid
            draw_coordinate_grid(display_frame, homography_matrix, playground_dims, grid_spacing=10)

            # Process and display all clicked points
            for i, point in enumerate(clicked_points):
                # Transform pixel to playground coordinates
                playground_coords = transform_point(point, homography_matrix)

                # Visualize the clicked point
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1)

                # Add text with coordinates
                text = f"Point {i + 1}: ({playground_coords[0]:.1f}, {playground_coords[1]:.1f})cm"
                cv2.putText(display_frame, text,
                            (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow(window_name, display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit
            break
        elif key == ord('r'):
            # Recalibrate
            homography_matrix = None
            marker_centers = None
            playground_dims = None
            clicked_points = []
            is_setup_complete = False
            print("\nRecalibrating coordinate system...")
        elif key == ord('c'):
            # Clear all clicked points
            clicked_points = []
            print("Cleared all marked points")
        elif key == ord('s'):
            # Take snapshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f'playground_snapshot_{timestamp}.jpg'
            cv2.imwrite(filename, display_frame)
            print(f"Snapshot saved as '{filename}'")

    # Clean up
    camera.release()
    cv2.destroyAllWindows()


def draw_coordinate_grid(frame, homography_matrix, playground_dims, grid_spacing=10):
    """
    Draw a coordinate grid on the frame.

    Args:
        frame: Frame to draw on
        homography_matrix: Homography matrix for coordinate transformation
        playground_dims: Playground dimensions (width, height) in cm
        grid_spacing: Grid spacing in cm
    """
    width, height = playground_dims

    # Draw grid lines
    for x in range(0, int(width) + 1, grid_spacing):
        # Get start and end points in playground coordinates
        start_point = (x, 0)
        end_point = (x, height)

        # Transform to pixel coordinates
        start_pixel = inverse_transform_point(start_point, homography_matrix)
        end_pixel = inverse_transform_point(end_point, homography_matrix)

        # Convert to integers for drawing
        start_pixel = (int(start_pixel[0]), int(start_pixel[1]))
        end_pixel = (int(end_pixel[0]), int(end_pixel[1]))

        # Draw vertical grid line
        cv2.line(frame, start_pixel, end_pixel, (100, 100, 100), 1)

        # Add coordinate label at the top
        if x % (grid_spacing * 2) == 0:  # Label every other line to avoid clutter
            label_point = (int(start_pixel[0]), int(start_pixel[1]) - 10)
            cv2.putText(frame, f"{x}", label_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

    for y in range(0, int(height) + 1, grid_spacing):
        # Get start and end points in playground coordinates
        start_point = (0, y)
        end_point = (width, y)

        # Transform to pixel coordinates
        start_pixel = inverse_transform_point(start_point, homography_matrix)
        end_pixel = inverse_transform_point(end_point, homography_matrix)

        # Convert to integers for drawing
        start_pixel = (int(start_pixel[0]), int(start_pixel[1]))
        end_pixel = (int(end_pixel[0]), int(end_pixel[1]))

        # Draw horizontal grid line
        cv2.line(frame, start_pixel, end_pixel, (100, 100, 100), 1)

        # Add coordinate label on the left
        if y % (grid_spacing * 2) == 0:  # Label every other line to avoid clutter
            label_point = (int(start_pixel[0]) - 25, int(start_pixel[1]))
            cv2.putText(frame, f"{y}", label_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

    # Draw axes labels
    x_label_pixel = inverse_transform_point((width / 2, -5), homography_matrix)
    y_label_pixel = inverse_transform_point((-5, height / 2), homography_matrix)

    cv2.putText(frame, "X (cm)",
                (int(x_label_pixel[0]), int(x_label_pixel[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, "Y (cm)",
                (int(y_label_pixel[0]), int(y_label_pixel[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


if __name__ == "__main__":
    # Run the test
    test_playground_coordinates(camera_index=0)