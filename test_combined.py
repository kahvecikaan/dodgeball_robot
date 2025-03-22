"""
Combined test script for ball-dodging robot vision system.
Integrates ball detection, playground coordinate system visualization,
throw data recording, and previous throw visualization.
"""

import csv
import time

import cv2
import numpy as np

from object_detection import detect_ball_color
from playground_setup import create_aruco_dict, detect_aruco_markers, transform_point, inverse_transform_point


def test_combined_system(camera_index=0, csv_filename="throw_data.csv"):
    """
    Combined test for ball detection and playground coordinate system.
    Records throw data to CSV and allows visualization of previous throws.

    Args:
        camera_index: Index of camera to use
        csv_filename: Filename to save throw data
    """
    # Initialize camera
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    # Create ArUco dictionary and parameters
    aruco_dict, parameters = create_aruco_dict()

    # Create window
    window_name = 'Combined Ball and Playground Test'
    cv2.namedWindow(window_name)

    # Load default tennis ball color
    lower_color = np.array([25, 50, 50])
    upper_color = np.array([65, 255, 255])

    # Variables to store coordinate system data
    homography_matrix = None
    playground_dims = None
    is_setup_complete = False

    # Lists to store position history for trajectory visualization
    ball_pixel_positions = []
    ball_playground_positions = []
    max_history = 30  # Number of past positions to keep for trajectory

    # Variables for throw recording
    is_recording = False
    current_throw = []
    all_throws = []
    throw_counter = 0
    start_time = 0

    # Variable for throw visualization
    show_previous_throws = False

    # Define colors for visualizing different throws
    throw_colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 165, 0),  # Orange
        (255, 0, 100), # Purple
        (181, 130, 28), # Mediterranean Blue
        (28, 181, 163), # Pea
    ]

    print("\nCombined Ball Detection and Playground Coordinate Test with Throw Recording")
    print("=========================================================================")
    print("SETUP INSTRUCTIONS:")
    print("1. Place the four ArUco markers (IDs 0, 1, 2, 3) in a rectangular configuration")
    print("2. The markers should form a rectangle or square on a flat surface")
    print("3. Place the ball in the camera view")
    print("\nINTERACTION:")
    print("- Press 'r' to start/stop recording a throw")
    print("- Press 'v' to toggle visualization of previous throws")
    print("- Press 's' to save all recorded throws to CSV")
    print("- Press 'c' to clear trajectory history")
    print("- Press 'x' to recalibrate the coordinate system")
    print("- Press 'q' to quit (without saving)")

    while True:
        # Capture frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame. Check camera connection.")
            break

        # Make a copy for visualization
        display_frame = frame.copy()

        # Step 1: Set up playground coordinate system if not already done
        if not is_setup_complete:
            # Detect ArUco markers
            corners, ids, frame_markers = detect_aruco_markers(frame, aruco_dict, parameters)
            display_frame = frame_markers  # Use marked frame for display

            # Check if all required corner markers are detected
            if ids is not None:
                # Convert ids to a flat array for easier comparison
                ids_flat = ids.flatten() # to convert from [[0],[1],[2],[3]] to [0,1,2,3]
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

                    # Convert to numpy array
                    marker_centers = np.array(marker_centers, dtype=np.float32)

                    # Ask for real-world dimensions if not already provided
                    # If dimensions are known, skip the prompt
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
                            homography_matrix, _ = cv2.findHomography(marker_centers, playground_corners_real)
                            is_setup_complete = True

                            print("\nSetup complete! Playground coordinate system established.")
                            print(f"Playground dimensions: {width}cm x {height}cm")
                            print("\nTracking ball and visualizing trajectory...")
                            print("Press 'r' to start recording a throw, press 'r' again to stop recording.")

                        except ValueError:
                            print("Invalid input. Please enter numeric values.")
                    else:
                        # Use existing dimensions with new marker positions
                        width, height = playground_dims
                        playground_corners_real = np.array([
                            [0, 0],  # Top-left (ID 0)
                            [width, 0],  # Top-right (ID 1)
                            [width, height],  # Bottom-right (ID 2)
                            [0, height]  # Bottom-left (ID 3)
                        ], dtype=np.float32)

                        # Calculate new homography matrix with existing dimensions
                        homography_matrix, _ = cv2.findHomography(marker_centers, playground_corners_real)
                        is_setup_complete = True
                        print("\nPlayground recalibrated with existing dimensions.")
                        print(f"Playground dimensions: {width}cm x {height}cm")
                else:
                    missing = [m for m in required_markers if m not in ids_flat]
                    message = f"Missing markers: {missing}"
                    cv2.putText(display_frame, message, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "No markers detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Step 2: If setup is complete, detect ball and show coordinate grid
        if is_setup_complete:
            # Draw coordinate grid
            draw_coordinate_grid(display_frame, homography_matrix, playground_dims, grid_spacing=10)

            # Get ArUco marker positions to update display
            corners, ids, _ = detect_aruco_markers(frame, aruco_dict, parameters)

            # Draw markers if detected
            if ids is not None:
                for i, id in enumerate(ids.flatten()):
                    # Calculate center of marker
                    center = np.mean(corners[i][0], axis=0)
                    center = (int(center[0]), int(center[1]))

                    # Draw circle at marker center
                    cv2.circle(display_frame, center, 5, (0, 255, 255), -1)
                    cv2.putText(display_frame, f"ID: {id}", (center[0] + 10, center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Show previous throws if toggled on
            if show_previous_throws and all_throws:
                # Draw trajectories for all recorded throws
                for throw_idx, throw in enumerate(all_throws):
                    if throw:  # Only if throw has data
                        # Get color for this throw (cycle through colors if we have more throws than colors)
                        color = throw_colors[throw_idx % len(throw_colors)]

                        # Extract pixel coordinates for drawing
                        pixel_points = [(point[1], point[2]) for point in throw]

                        # Draw trajectory line
                        for i in range(1, len(pixel_points)):
                            cv2.line(display_frame,
                                     pixel_points[i - 1],
                                     pixel_points[i],
                                     color, 2)

                        # Draw circles at beginning and end
                        cv2.circle(display_frame, pixel_points[0], 7, color, -1)
                        cv2.circle(display_frame, pixel_points[-1], 7, color, 2)

                        # Add throw number
                        cv2.putText(display_frame, f"#{throw_idx + 1}",
                                    pixel_points[0],
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Detect ball using color method
            ball_pos, mask = detect_ball_color(frame, lower_color, upper_color)

            # Update trajectory if ball is detected
            if ball_pos:
                x, y, radius = ball_pos

                # Add to pixel position history
                ball_pixel_positions.append((x, y))
                if len(ball_pixel_positions) > max_history:
                    ball_pixel_positions.pop(0)

                # Transform to playground coordinates
                playground_coords = transform_point((x, y), homography_matrix)
                ball_playground_positions.append(playground_coords)
                if len(ball_playground_positions) > max_history:
                    ball_playground_positions.pop(0)

                # Record position if currently recording a throw
                if is_recording:
                    elapsed_time = time.time() - start_time
                    # Store: time, pixel x, pixel y, real x, real y
                    current_throw.append((elapsed_time, x, y, playground_coords[0], playground_coords[1]))

                # Draw current ball position
                cv2.circle(display_frame, (x, y), radius, (0, 255, 0), 2)  # Ball outline
                cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)  # Center point

                # Display coordinates
                pixel_text = f"Pixel: ({x}, {y})"
                playground_text = f"Real: ({playground_coords[0]:.1f}, {playground_coords[1]:.1f}) cm"

                cv2.putText(display_frame, pixel_text, (x + radius + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, playground_text, (x + radius + 10, y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw current trajectory in pixel space (only if not showing previous throws)
            if not show_previous_throws and len(ball_pixel_positions) > 1:
                for i in range(1, len(ball_pixel_positions)):
                    # Gradient color based on position in trajectory (newer = brighter)
                    alpha = i / len(ball_pixel_positions)
                    color = (0, int(255 * alpha), 0)

                    cv2.line(display_frame,
                             ball_pixel_positions[i - 1],
                             ball_pixel_positions[i],
                             color, 2)

            # Add playground coordinate speed estimate if we have enough history
            if len(ball_playground_positions) >= 2:
                # Calculate instantaneous velocity
                dt = 1 / 30.0  # Assuming 30 fps
                dx = ball_playground_positions[-1][0] - ball_playground_positions[-2][0]
                dy = ball_playground_positions[-1][1] - ball_playground_positions[-2][1]
                speed = np.sqrt(dx * dx + dy * dy) / dt

                # Display speed
                cv2.putText(display_frame, f"Speed: {speed:.1f} cm/s", (10, display_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Show recording and visualization status
            status_y = 30  # Starting y-position for status text

            # Show visualization mode
            if show_previous_throws:
                cv2.putText(display_frame, f"SHOWING {throw_counter} PREVIOUS THROWS",
                            (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                status_y += 30

            # Show recording status
            if is_recording:
                # Recording indicator with red pulsing effect
                pulse = int(127 * np.sin(time.time() * 8) + 128)  # Pulsing effect
                cv2.putText(display_frame, f"RECORDING THROW #{throw_counter + 1} - {len(current_throw)} points",
                            (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, pulse), 2)
                # Draw red border around frame
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1] - 1, display_frame.shape[0] - 1),
                              (0, 0, 255), 3)
                status_y += 30
            else:
                # Show total recorded throws
                cv2.putText(display_frame, f"Recorded Throws: {throw_counter}",
                            (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                status_y += 30

            # Add help text (keypress options)
            cv2.putText(display_frame, "r: record  v: view throws  s: save  q: quit",
                        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display the result
        cv2.imshow(window_name, display_frame)

        # Handle key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            # Quit without saving
            if throw_counter > 0 and all_throws:
                print(f"Quitting without saving {throw_counter} throws. Press 's' first if you want to save.")
            break
        elif key == ord('s'):
            # Save throws to CSV
            if throw_counter > 0:
                # If currently recording, finalize the current throw
                if is_recording and current_throw:
                    all_throws.append(current_throw)
                    throw_counter += 1
                    is_recording = False
                    current_throw = []
                    print(f"Throw #{throw_counter} recorded with {len(current_throw)} points")

                # Save all throws
                save_throws_to_csv(all_throws, csv_filename)
                print(f"Saved {throw_counter} throws to {csv_filename}")
            else:
                print("No throws to save.")
        elif key == ord('x'):
            # Recalibrate coordinate system
            homography_matrix = None
            # playground_dims remains intact
            is_setup_complete = False
            ball_pixel_positions = []
            ball_playground_positions = []
            # Keep the recorded throws
            print("\nRecalibrating coordinate system...")
        elif key == ord('c'):
            # Clear trajectory
            ball_pixel_positions = []
            ball_playground_positions = []
            print("Cleared trajectory history.")
        elif key == ord('v'):
            # Toggle visualization of previous throws
            show_previous_throws = not show_previous_throws
            if show_previous_throws:
                print(f"Showing {throw_counter} previous throws")
            else:
                print("Hiding previous throws")
        elif key == ord('r'):
            # Toggle recording state
            if is_recording:
                # Stop recording current throw
                is_recording = False
                if current_throw:  # Only save if we have data
                    all_throws.append(current_throw)
                    throw_counter += 1
                    print(f"Throw #{throw_counter} recorded with {len(current_throw)} points")
                    current_throw = []  # Reset for next throw
            else:
                # Start recording new throw
                is_recording = True
                current_throw = []  # Clear any old data
                start_time = time.time()  # Reset time reference
                print(f"Started recording throw #{throw_counter + 1}")

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


def save_throws_to_csv(all_throws, filename):
    """
    Save recorded throws to a CSV file.

    Args:
        all_throws: List of throws, where each throw is a list of position tuples
        filename: Name of CSV file to write
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['throw_id', 'time', 'x_pixel', 'y_pixel', 'x_cm', 'y_cm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for throw_id, throw in enumerate(all_throws):
            for point in throw:
                time_val, x_pixel, y_pixel, x_cm, y_cm = point
                writer.writerow({
                    'throw_id': throw_id + 1,
                    'time': f"{time_val:.3f}",
                    'x_pixel': int(x_pixel),
                    'y_pixel': int(y_pixel),
                    'x_cm': f"{x_cm:.2f}",
                    'y_cm': f"{y_cm:.2f}"
                })


if __name__ == "__main__":
    # Run the combined test
    test_combined_system()