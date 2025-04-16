"""
Enhanced combined test script for ball-dodging robot vision system.
Includes trajectory prediction, robot detection, collision warning and dodge commands.
"""

import csv
import time

import cv2
import numpy as np

from object_detection import detect_ball_color, detect_robot
from playground_setup import create_aruco_dict, detect_aruco_markers, transform_point, inverse_transform_point
from trajectory_prediction import KalmanTracker
from dodge_command import DodgeCommandModule

def test_combined_system(camera_index=0, csv_filename="throw_data.csv", arduino_port=None, disable_dodge=False):
    """
    Combined test for ball detection and playground coordinate system.
    Records throw data to CSV, predicts trajectories, and detects potential collisions.

    Args:
        camera_index: Index of camera to use
        csv_filename: Filename to save throw data
        arduino_port: Serial port for Arduino (auto-detect if not specified)
        disable_dodge: If True, don't initialize the dodge command module
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

    # Load default tennis ball color (narrowed range for better specificity)
    lower_color = np.array([28, 80, 80])
    upper_color = np.array([60, 255, 255])

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

    # Trajectory prediction variables
    kalman_tracker = None
    prediction_enabled = True

    # Variables for temporal consistency tracking
    detection_counter = 0
    min_detection_confidence = 2  # Reduced from 3 to 2 to be more responsive

    # Variables for persistent prediction display
    last_valid_trajectory = []
    last_valid_landing_point = None
    prediction_active = False

    # Robot tracking variables
    robot_position = None
    robot_width = 15  # Width of robot in cm (adjust as needed)
    collision_detected = False
    collision_point = None

    # Initialize dodge command module
    dodge_module = None
    if not disable_dodge:
        dodge_module = DodgeCommandModule(port=arduino_port, robot_width=robot_width)
        if dodge_module.connected:
            print("Dodge command module connected")
        else:
            print("Failed to connect dodge command module. Only visualization will be available.")

    # Define colors for visualizing different throws
    throw_colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 165, 0),  # Orange
        (255, 0, 100),  # Purple
        (181, 130, 28),  # Mediterranean Blue
        (28, 181, 163),  # Pea
    ]

    print("\nEnhanced Ball Detection and Robot Collision System")
    print("=================================================")
    print("SETUP INSTRUCTIONS:")
    print("1. Place the four ArUco markers (IDs 0, 1, 2, 3) in a rectangular configuration")
    print("2. The markers should form a rectangle or square on a flat surface")
    print("3. Place marker ID 42 on the robot at the bottom of the playground")
    print("4. Place the ball in the camera view")
    print("\nINTERACTION:")
    print("- Press 'r' to start/stop recording a throw")
    print("- Press 'v' to toggle visualization of previous throws")
    print("- Press 'p' to toggle trajectory prediction")
    print("- Press 's' to save all recorded throws to CSV")
    print("- Press 'c' to clear trajectory history and predictions")
    print("- Press 'x' to recalibrate the coordinate system")
    print("- Press 'q' to quit (without saving)")
    if dodge_module and dodge_module.connected:
        print("DODGE CONTROLS:")
        print("- Press 'a' to test dodge left")
        print("- Press 'd' to test dodge right")
        print("- Press 'e' to emergency stop")

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
                ids_flat = ids.flatten()
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

            # Detect robot (marker 42)
            robot_result, robot_vis_frame = detect_robot(
                frame, aruco_dict, parameters,
                robot_marker_id=42,
                homography_matrix=homography_matrix
            )

            # Update robot position if detected
            if robot_result:
                robot_position = robot_result[0]  # Get x-coordinate of robot

                # Update dodge module with robot position
                if dodge_module and dodge_module.connected:
                    dodge_module.update_robot_position(robot_position)

                # Draw robot visualization
                if not show_previous_throws:  # Only show if not displaying previous throws
                    # Calculate robot position in playground coordinates
                    width, height = playground_dims
                    robot_y = height  # Robot is at maximum y

                    # Convert to pixel coordinates for display
                    robot_center_pixel = inverse_transform_point((robot_position, robot_y), homography_matrix)
                    robot_left_pixel = inverse_transform_point((robot_position - robot_width / 2, robot_y),
                                                               homography_matrix)
                    robot_right_pixel = inverse_transform_point((robot_position + robot_width / 2, robot_y),
                                                                homography_matrix)

                    # Draw robot representation
                    cv2.line(display_frame,
                             (int(robot_left_pixel[0]), int(robot_left_pixel[1])),
                             (int(robot_right_pixel[0]), int(robot_right_pixel[1])),
                             (0, 0, 255), 4)

                    # Add robot label
                    cv2.putText(display_frame, f"Robot: {robot_position:.1f} cm",
                                (int(robot_center_pixel[0]) - 50, int(robot_center_pixel[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw dodge target position if active
            if dodge_module and dodge_module.connected and dodge_module.is_dodging and dodge_module.target_position is not None:
                # Calculate target position in playground coordinates
                width, height = playground_dims
                target_x = dodge_module.target_position
                target_y = height # Robot's y position is at max_y

                # Convert to pixel coordinates for display
                target_pixel = inverse_transform_point((target_x, target_y), homography_matrix)

                # Draw target position indicator
                cv2.circle(display_frame,
                           (int(target_pixel[0]), int(target_pixel[1])),
                           10, (0, 255, 0), 2)

                # Draw label
                cv2.putText(display_frame, "DODGE TARGET",
                            (int(target_pixel[0]) + 15, int(target_pixel[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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

            # Detect ball using basic color method - ignore mask return value as we don't use it
            ball_pos, _ = detect_ball_color(frame, lower_color, upper_color, min_radius=10)

            # Update temporal consistency tracking
            if ball_pos:
                detection_counter += 1
            else:
                detection_counter = max(0, detection_counter - 1)

            # Only process ball if detection is reliable
            is_detection_reliable = detection_counter >= min_detection_confidence

            # Update trajectory if ball is detected and reliable
            if ball_pos and is_detection_reliable:
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

                # Update trajectory prediction if enabled
                if prediction_enabled and len(ball_playground_positions) >= 2:
                    # Initialize or update Kalman tracker
                    if kalman_tracker is None:
                        kalman_tracker = KalmanTracker(initial_pos=playground_coords)
                    else:
                        kalman_tracker.update(playground_coords)

                    # Only predict trajectory when we have enough data
                    # and the ball is in the first half of the playground
                    width, height = playground_dims
                    if playground_coords[1] < height / 2:
                        # Predict future trajectory
                        predicted_trajectory = kalman_tracker.predict_trajectory(num_steps=30)

                        # Find where trajectory intersects with maximum y
                        landing_point = find_landing_point(predicted_trajectory, height)

                        # Update persistent prediction if we have a valid landing point
                        if landing_point:
                            last_valid_trajectory = predicted_trajectory
                            last_valid_landing_point = landing_point
                            prediction_active = True

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

            # Display the persistent prediction if active
            if prediction_active and prediction_enabled:
                # Draw the latest valid predicted trajectory
                if last_valid_trajectory:
                    draw_predicted_trajectory(display_frame, last_valid_trajectory, homography_matrix, (0, 255, 255))

                # Draw the latest valid landing point
                if last_valid_landing_point:
                    draw_landing_point(display_frame, last_valid_landing_point, homography_matrix)

                # Check for collision if robot is detected
                if robot_position is not None and last_valid_landing_point is not None:
                    # Use the check_collision function instead of inline code
                    collision_detected = check_collision(
                        last_valid_landing_point,
                        robot_position,
                        robot_width,
                        threshold=5
                    )

                    if collision_detected:
                        collision_point = last_valid_landing_point
                    else:
                        collision_point = None

                # Display collision warning if detected
                if collision_detected:
                    # Draw warning text
                    warning_text = "COLLISION WARNING!"
                    cv2.putText(display_frame, warning_text,
                                (display_frame.shape[1] // 2 - 150, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    # Calculate impact point in pixel coordinates
                    if collision_point:
                        # Send dodge command if module is available
                        if dodge_module and dodge_module.connected and not dodge_module.is_dodging:
                            dodge_module.process_collision(collision_point)

                        impact_pixel = inverse_transform_point(collision_point, homography_matrix)
                        x, y = int(impact_pixel[0]), int(impact_pixel[1])

                        # Draw flashing impact marker (pulsing effect)
                        flash_intensity = int(127 * np.sin(time.time() * 8) + 128)
                        cv2.circle(display_frame, (x, y), 15, (0, 0, flash_intensity), -1)
                        cv2.circle(display_frame, (x, y), 20, (0, 0, 255), 3)

                        # Draw warning arrows pointing to impact point
                        arrow_length = 30
                        cv2.arrowedLine(display_frame,
                                        (x - arrow_length, y - arrow_length),
                                        (x - 5, y - 5),
                                        (0, 0, 255), 2, tipLength=0.3)
                        cv2.arrowedLine(display_frame,
                                        (x + arrow_length, y - arrow_length),
                                        (x + 5, y - 5),
                                        (0, 0, 255), 2, tipLength=0.3)

            # Add playground coordinate speed estimate if we have enough history
            if len(ball_playground_positions) >= 3:
                # Calculate average velocity over last few frames
                velocities = []
                for i in range(1, min(5, len(ball_playground_positions))):
                    dx = ball_playground_positions[-i][0] - ball_playground_positions[-i-1][0]
                    dy = ball_playground_positions[-i][1] - ball_playground_positions[-i-1][1]
                    dt = 1 / 30.0 # Assuming 30 FPS
                    velocities.append(np.sqrt(dx*dx + dy*dy) / dt)

                # Average velocity
                if velocities:
                    speed = sum(velocities) / len(velocities)
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

            # Show prediction status
            if prediction_enabled:
                cv2.putText(display_frame, "TRAJECTORY PREDICTION ON",
                            (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                status_y += 30

                # Show landing point if available
                if prediction_active and last_valid_landing_point:
                    cv2.putText(display_frame, f"Predicted landing: X = {last_valid_landing_point[0]:.1f} cm",
                                (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    status_y += 30

            # Show dodge status if connected
            if dodge_module and dodge_module.connected:
                if dodge_module.is_dodging:
                    # Dodge status with target position
                    dodge_text = f"DODGING to X = {dodge_module.target_position:.1f} cm"
                    cv2.putText(display_frame, dodge_text,
                                (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
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
            cv2.putText(display_frame, "r: record  v: view throws  p: toggle prediction  s: save  q: quit",
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
            kalman_tracker = None
            last_valid_trajectory = []
            last_valid_landing_point = None
            prediction_active = False
            collision_detected = False
            collision_point = None
            # Keep the recorded throws
            print("\nRecalibrating coordinate system...")
        elif key == ord('c'):
            # Clear trajectory and prediction
            ball_pixel_positions = []
            ball_playground_positions = []
            kalman_tracker = None
            last_valid_trajectory = []
            last_valid_landing_point = None
            prediction_active = False
            collision_detected = False
            collision_point = None
            print("Cleared trajectory history and predictions.")
        elif key == ord('v'):
            # Toggle visualization of previous throws
            show_previous_throws = not show_previous_throws
            if show_previous_throws:
                print(f"Showing {throw_counter} previous throws")
            else:
                print("Hiding previous throws")
        elif key == ord('p'):
            # Toggle trajectory prediction
            prediction_enabled = not prediction_enabled
            if not prediction_enabled:
                prediction_active = False
                collision_detected = False
            print(f"Trajectory prediction: {'ON' if prediction_enabled else 'OFF'}")
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
        elif key == ord('e'):
            if dodge_module and dodge_module.connected:
                dodge_module.emergency_stop()
                print("Emergency stop sent")
        elif key == ord('d'): # Test dodge right
            if dodge_module and dodge_module.connected and robot_position is not None:
                dodge_module.test_dodge_right(20.0)
                print("Test dodge right initiated")
        elif key == ord('a'): # Test dodge left
            if dodge_module and dodge_module.connected and robot_position is not None:
                dodge_module.test_dodge_left(20.0)
                print("Test dodge left initiated")

    # Clean up
    if dodge_module and dodge_module.connected:
        dodge_module.close()

    camera.release()
    cv2.destroyAllWindows()

def find_landing_point(trajectory, max_y):
    """
    Find where the ball's trajectory intersects with the maximum y-coordinate.

    Args:
        trajectory: List of predicted positions [(x1, y1), (x2, y2), ...]
        max_y: Maximum y-coordinate of the playground

    Returns:
        landing_point: (x, y) coordinates of the landing point, or None if not found
    """
    if not trajectory or len(trajectory) < 2:
        return None

    # Check each segment of the trajectory
    for i in range(1, len(trajectory)):
        y1 = trajectory[i - 1][1]
        y2 = trajectory[i][1]

        # If this segment crosses max_y
        if (y1 <= max_y <= y2) or (y2 <= max_y <= y1):
            x1 = trajectory[i - 1][0]
            x2 = trajectory[i][0]

            # If the segment is vertical (or nearly so)
            if abs(x2 - x1) < 0.001:
                x_intersect = x1
            else:
                # Linear interpolation to find x at max_y
                t = (max_y - y1) / (y2 - y1)
                x_intersect = x1 + t * (x2 - x1)

            return x_intersect, max_y

    # If we've examined the entire trajectory and found no intersection
    # Check if the last point is beyond max_y
    if trajectory[-1][1] > max_y:
        # Extrapolate from the last two points
        x1, y1 = trajectory[-2]
        x2, y2 = trajectory[-1]

        # If the segment is vertical (or nearly so)
        if abs(x2 - x1) < 0.001:
            x_intersect = x1
        else:
            # Linear extrapolation to find x at max_y
            t = (max_y - y1) / (y2 - y1)
            x_intersect = x1 + t * (x2 - x1)

        return x_intersect, max_y

    return None


def check_collision(landing_point, robot_position, robot_width, threshold=5):
    """
    Check if predicted landing point will collide with the robot.

    Args:
        landing_point: (x, y) coordinates of predicted landing point
        robot_position: x-coordinate of robot's center
        robot_width: Width of the robot in cm
        threshold: Additional threshold in cm to account for uncertainty

    Returns:
        collision: Boolean indicating whether a collision is predicted
    """
    if landing_point is None or robot_position is None:
        return False

    # Calculate distance between landing point x and robot x
    distance = abs(landing_point[0] - robot_position)

    # Check if landing point is within robot's width plus threshold
    collision_threshold = (robot_width / 2) + threshold

    return distance < collision_threshold


def draw_predicted_trajectory(frame, trajectory, homography_matrix, color):
    """
    Draw the predicted trajectory on the frame.

    Args:
        frame: Frame to draw on
        trajectory: List of predicted positions in real-world coordinates
        homography_matrix: Homography matrix for coordinate transformation
        color: Color to draw the trajectory (B, G, R)
    """
    if not trajectory or len(trajectory) < 2:
        return

    # Convert trajectory points to pixel coordinates
    pixel_points = []
    for point in trajectory:
        pixel_point = inverse_transform_point(point, homography_matrix)
        pixel_points.append((int(pixel_point[0]), int(pixel_point[1])))

    # Draw lines connecting the points
    for i in range(1, len(pixel_points)):
        cv2.line(frame, pixel_points[i - 1], pixel_points[i], color, 2, cv2.LINE_AA)

    # Draw small circles at each predicted position
    for point in pixel_points:
        cv2.circle(frame, point, 2, color, -1)


def draw_landing_point(frame, landing_point, homography_matrix):
    """
    Draw the predicted landing point on the frame.

    Args:
        frame: Frame to draw on
        landing_point: (x, y) coordinates of landing point in real-world coordinates
        homography_matrix: Homography matrix for coordinate transformation
    """
    if landing_point is None:
        return

    # Convert landing point to pixel coordinates
    pixel_point = inverse_transform_point(landing_point, homography_matrix)
    x, y = int(pixel_point[0]), int(pixel_point[1])

    # Draw attention-grabbing marker
    cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)  # Filled circle
    cv2.circle(frame, (x, y), 12, (0, 255, 255), 2)  # Outer ring

    # Draw crosshairs
    cv2.line(frame, (x - 15, y), (x - 5, y), (0, 255, 255), 2)  # Left
    cv2.line(frame, (x + 5, y), (x + 15, y), (0, 255, 255), 2)  # Right
    cv2.line(frame, (x, y - 15), (x, y - 5), (0, 255, 255), 2)  # Top
    cv2.line(frame, (x, y + 5), (x, y + 15), (0, 255, 255), 2)  # Bottom

    # Add text label with x-coordinate
    label = f"X: {landing_point[0]:.1f} cm"
    cv2.putText(frame, label, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


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