"""
Enhanced combined test script for ball-dodging robot vision system.
Includes trajectory prediction, robot detection, collision warning and dodge commands.
"""
import argparse
import csv
import time

import cv2 
import numpy as np

from object_detection import detect_ball_color, detect_robot
from playground_setup import create_aruco_dict, detect_aruco_markers, transform_point, inverse_transform_point
from trajectory_prediction import KalmanTracker
from dodge_command import DodgeCommandModule

COLORS = {
    'bg_dark': (10, 10, 10),
    'grid': (60, 60, 60),
    'grid_label': (120, 120, 200),
    'position': (52, 152, 219),  # Blue for positions/measurements
    'prediction': (241, 196, 15),  # Yellow for predictions
    'warning': (231, 76, 60),  # Red for warnings
    'command': (46, 204, 113),  # Green for commands/actions
    'status': (149, 165, 166),  # Gray for status text
    'robot': (192, 57, 43),  # Dark red for robot
    'record': (192, 57, 43),  # Dark red for recording
    'panel_bg': (0, 0, 0, 0.7),  # Semi-transparent black for panels
    'panel_border': (200, 200, 200),  # Light gray for panel borders
    'info_text': (255, 255, 255),  # White for text in panels
    'target': (39, 174, 96)  # Dark green for target
}


def draw_panel(frame, title, content_lines, x, y, width, height, title_color, border_color=(200, 200, 200)):
    """
    Draw a semi-transparent panel with title and content.

    Args:
        frame: Frame to draw on
        title: Panel title
        content_lines: List of lines to display in panel
        x: x coordinate of the top-left corner
        y: y coordinate of the top-left corner
        width: Width of the panel
        height: Height of the panel
        title_color: Color for the title
        border_color: Color for panel border
    """
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), COLORS['bg_dark'], -1)

    # Add overlay to frame with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw border
    cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 1)

    # Draw title with background
    title_height = 30
    cv2.rectangle(frame, (x, y), (x + width, y + title_height), title_color, -1)
    cv2.putText(frame, title, (x + 10, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw content
    for i, line in enumerate(content_lines):
        y_pos = y + title_height + 25 + i * 25
        cv2.putText(frame, line, (x + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['info_text'], 1)


def draw_controls_bar(frame, dodge_enabled=False):
    """Draw a control reference bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_height = 40
    y = h - bar_height

    # Draw background
    cv2.rectangle(frame, (0, y), (w, h), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, y), (w, h), (100, 100, 100), 1)

    # Prepare control text
    controls = [
        "[R] Record",
        "[V] View Throws",
        "[P] Toggle Prediction",
        "[S] Save",
        "[Q] Quit"
    ]

    if dodge_enabled:
        controls.extend(["[A/D] Test Dodge", "[E] Emergency Stop"])

    # Draw controls as a single line
    control_text = "  |  ".join(controls)
    cv2.putText(frame, control_text, (10, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


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
    window_name = 'Dodgeball Robot Vision System'
    cv2.namedWindow(window_name)

    # Load default tennis ball color (narrowed range for better specificity)
    lower_color = np.array([77, 104, 116])
    upper_color = np.array([118, 251, 255])

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

    dodge_info = None

    # Initialize dodge command module
    dodge_module = None
    if not disable_dodge: #disable_dodge
        try:
            dodge_module = DodgeCommandModule(port=arduino_port, robot_width=robot_width)
            if dodge_module.connected:
                print("Dodge command module connected")
                connected_port = dodge_module.get_connected_port()
                #print(f"Dodge command module connected on port: {connected_port}")  # Debug
            else:
                print("Failed to connect to physical Arduino.")
                print("Running in simulation mode - dodge commands will processed but not sent to hardware")
                # Initialize with simulated condition
                dodge_module.connected = True  # Force connected state for testing
        except Exception as e:
            print(f"Error initializing dodge module: {e}")
            print("Running without dodge capability")

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
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['warning'], 2)
            else:
                cv2.putText(display_frame, "No markers detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['warning'], 2)

        # Step 2: If setup is complete, detect ball and show coordinate grid
        if is_setup_complete:
            # Draw coordinate grid
            draw_coordinate_grid(display_frame, homography_matrix, playground_dims, grid_spacing=20)

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

                    # Draw robot representation with improved visibility
                    cv2.line(display_frame,
                             (int(robot_left_pixel[0]), int(robot_left_pixel[1])),
                             (int(robot_right_pixel[0]), int(robot_right_pixel[1])),
                             COLORS['robot'], 5)  # Thicker line

                    # Add larger circle at center
                    cv2.circle(display_frame,
                               (int(robot_center_pixel[0]), int(robot_center_pixel[1])),
                               8, COLORS['robot'], -1)

                    # Add robot label with background for better visibility
                    label = f"Robot: {robot_position:.1f} cm"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    label_x = int(robot_center_pixel[0]) - text_size[0] // 2
                    label_y = int(robot_center_pixel[1]) - 15

                    # Background for text
                    cv2.rectangle(display_frame,
                                  (label_x - 5, label_y - 20),
                                  (label_x + text_size[0] + 5, label_y + 5),
                                  COLORS['bg_dark'], -1)
                    cv2.putText(display_frame, label,
                                (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['info_text'], 2)

            # Draw dodge target position if active
            if dodge_module and dodge_module.connected and dodge_module.is_dodging and dodge_module.target_position is not None:
                
                #if dodge_module.is_dodging:
                    #print("Dodgemodule is dodging") debug
                # Calculate target position in playground coordinates
                width, height = playground_dims
                target_x = dodge_module.target_position
                target_y = height  # Robot's y position is at max_y

                # Convert to pixel coordinates for display
                target_pixel = inverse_transform_point((target_x, target_y), homography_matrix)
                t_x, t_y = int(target_pixel[0]), int(target_pixel[1])

                # Convert current robot position to pixel coordinates
                robot_pixel = inverse_transform_point((robot_position, target_y), homography_matrix)
                r_x, r_y = int(robot_pixel[0]), int(robot_pixel[1])

                # Draw arrow from current position to target (thicker and brighter)
                cv2.arrowedLine(display_frame,
                                (r_x, r_y),
                                (t_x, t_y),
                                COLORS['target'], 3, tipLength=0.2)  # Thicker line

                # Draw target position with pulsing effect
                pulse = int(127 * np.sin(time.time() * 4) + 128)
                cv2.circle(display_frame, (t_x, t_y), 10, COLORS['target'], -1)  # Inner circle
                cv2.circle(display_frame, (t_x, t_y), 18, (0, pulse, 0), 2)  # Outer ring with pulse

                # Add text indicating target with better visibility
                target_label = f"TARGET: {target_x:.1f} cm"

                # Background for text
                text_size = cv2.getTextSize(target_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame,
                              (t_x + 20 - 5, t_y - 5),
                              (t_x + 20 + text_size[0] + 5, t_y + text_size[1] + 5),
                              COLORS['bg_dark'], -1)

                cv2.putText(display_frame, target_label,
                            (t_x + 20, t_y + text_size[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['target'], 2)

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

                        # Draw trajectory line with semi-transparent fill
                        points_array = np.array(pixel_points, np.int32)
                        points_array = points_array.reshape((-1, 1, 2))

                        # Draw semi-transparent fill under the trajectory
                        overlay = display_frame.copy()
                        cv2.polylines(overlay, [points_array], False, color, 3, cv2.LINE_AA)
                        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

                        # Draw circles at beginning and end
                        cv2.circle(display_frame, pixel_points[0], 8, color, -1)
                        cv2.circle(display_frame, pixel_points[-1], 8, color, 2)

                        # Add throw number with background
                        throw_label = f"#{throw_idx + 1}"
                        text_size = cv2.getTextSize(throw_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        label_bg_x = pixel_points[0][0] - 5
                        label_bg_y = pixel_points[0][1] - 25

                        cv2.rectangle(display_frame,
                                      (label_bg_x, label_bg_y),
                                      (label_bg_x + text_size[0] + 10, label_bg_y + text_size[1] + 10),
                                      COLORS['bg_dark'], -1)
                        cv2.putText(display_frame, throw_label,
                                    (label_bg_x + 5, label_bg_y + text_size[1] + 5),
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
                    # and the ball is in the 3/4 of the playground
                    width, height = playground_dims
                    if playground_coords[1] < height * 3 / 4:
                        # Predict future trajectory
                        predicted_trajectory = kalman_tracker.predict_trajectory(num_steps=30)

                        # Find where trajectory intersects with maximum y
                        landing_point = find_landing_point(predicted_trajectory, height)

                        # Update persistent prediction if we have a valid landing point
                        if landing_point:
                            last_valid_trajectory = predicted_trajectory
                            last_valid_landing_point = landing_point
                            prediction_active = True

                # Draw current ball position with improved visibility
                cv2.circle(display_frame, (x, y), radius, COLORS['position'], 2)  # Ball outline
                cv2.circle(display_frame, (x, y), 6, COLORS['position'], -1)  # Center point

                # Display coordinates with background for better readability
                position_panel_lines = [
                    f"Pixel: ({x}, {y})",
                    f"Real: ({playground_coords[0]:.1f}, {playground_coords[1]:.1f}) cm"
                ]

                # Calculate speed if we have enough history
                if len(ball_playground_positions) >= 3:
                    # Calculate average velocity over last few frames
                    velocities = []
                    for i in range(1, min(5, len(ball_playground_positions))):
                        dx = ball_playground_positions[-i][0] - ball_playground_positions[-i - 1][0]
                        dy = ball_playground_positions[-i][1] - ball_playground_positions[-i - 1][1]
                        dt = 1 / 30.0  # Assuming 30 FPS
                        velocities.append(np.sqrt(dx * dx + dy * dy) / dt)

                    # Average velocity
                    if velocities:
                        speed = sum(velocities) / len(velocities)
                        position_panel_lines.append(f"Speed: {speed:.1f} cm/s")

                # Draw position panel near the ball
                panel_width = 220
                panel_height = 25 * len(position_panel_lines) + 40
                panel_x = min(x + radius + 15, display_frame.shape[1] - panel_width - 10)
                panel_y = max(y - panel_height - 10, 10)

                draw_panel(display_frame, "Ball Data", position_panel_lines,
                           panel_x, panel_y, panel_width, panel_height,
                           COLORS['position'], COLORS['position'])

            # Draw current trajectory in pixel space (only if not showing previous throws)
            if not show_previous_throws and len(ball_pixel_positions) > 1:
                # Draw trajectory with gradient thickness and color
                for i in range(1, len(ball_pixel_positions)):
                    # Gradient color based on position in trajectory (newer = brighter)
                    alpha = i / len(ball_pixel_positions)
                    color = (0, int(100 * alpha) + 155, 0)  # Brighter green for newer segments
                    thickness = 1 + int(alpha * 3)  # Thicker lines for newer segments

                    cv2.line(display_frame,
                             ball_pixel_positions[i - 1],
                             ball_pixel_positions[i],
                             color, thickness, cv2.LINE_AA)

            # Display the persistent prediction if active
            if prediction_active and prediction_enabled:
                # Draw the latest valid predicted trajectory
                if last_valid_trajectory:
                    draw_predicted_trajectory(display_frame, last_valid_trajectory, homography_matrix,
                                              COLORS['prediction'])

                # Draw the latest valid landing point
                if last_valid_landing_point:
                    draw_landing_point(display_frame, last_valid_landing_point, homography_matrix)

                # Check for collision if robot is detected
                if robot_position is not None and last_valid_landing_point is not None:
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
                    # Create pulsing effect for warning
                    pulse = int(127 * np.sin(time.time() * 8) + 128)
                    warning_color = (0, 0, pulse)

                    # Draw warning panel at top of screen
                    warning_panel_width = 400
                    warning_panel_height = 60
                    warning_panel_x = display_frame.shape[1] // 2 - warning_panel_width // 2
                    warning_panel_y = 10

                    # Semi-transparent background
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay,
                                  (warning_panel_x, warning_panel_y),
                                  (warning_panel_x + warning_panel_width, warning_panel_y + warning_panel_height),
                                  COLORS['warning'], -1)
                    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

                    # Add border that pulses
                    cv2.rectangle(display_frame,
                                  (warning_panel_x, warning_panel_y),
                                  (warning_panel_x + warning_panel_width, warning_panel_y + warning_panel_height),
                                  warning_color, 3)

                    # Warning text
                    warning_text = "COLLISION WARNING!"
                    text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    text_x = warning_panel_x + (warning_panel_width - text_size[0]) // 2
                    text_y = warning_panel_y + warning_panel_height // 2 + 10

                    cv2.putText(display_frame, warning_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                    # Calculate impact point in pixel coordinates
                    if collision_point:
                        # Send dodge command if module is available
                        if dodge_module and dodge_module.connected and not dodge_module.is_dodging:
                            # Store the direction and robot position before dodge
                            pre_dodge_position = robot_position
                            dodge_direction = "left" if collision_point[0] > robot_position else "right"

                            # Process collision
                            dodge_result = dodge_module.process_collision(collision_point)

                            if dodge_result and dodge_module.target_position is not None:
                                # Calculate dodge distance
                                dodge_distance = abs(dodge_module.target_position - pre_dodge_position)

                                # Store dodge info for display
                                dodge_info = {
                                    "direction": dodge_direction,
                                    "distance": dodge_distance,
                                    "target": dodge_module.target_position
                                }

                        impact_pixel = inverse_transform_point(collision_point, homography_matrix)
                        x, y = int(impact_pixel[0]), int(impact_pixel[1])

                        # Draw flashing impact marker with pulsing effect
                        flash_intensity = int(127 * np.sin(time.time() * 8) + 128)
                        pulsing_color = (0, 0, flash_intensity)
                        cv2.circle(display_frame, (x, y), 15, pulsing_color, -1)
                        cv2.circle(display_frame, (x, y), 25, COLORS['warning'], 3)

                        # Draw warning arrows pointing to impact point
                        arrow_length = 30
                        cv2.arrowedLine(display_frame,
                                        (x - arrow_length, y - arrow_length),
                                        (x - 5, y - 5),
                                        COLORS['warning'], 3, tipLength=0.3)
                        cv2.arrowedLine(display_frame,
                                        (x + arrow_length, y - arrow_length),
                                        (x + 5, y - 5),
                                        COLORS['warning'], 3, tipLength=0.3)

            # Create status panel at top-left
            status_panel_lines = [f"Recorded Throws: {throw_counter}"]

            # Show total recorded throws

            # Recording status
            if is_recording:
                # Add recording indicator with pulse
                pulse = int(127 * np.sin(time.time() * 8) + 128)
                status_panel_lines.append(f"RECORDING THROW #{throw_counter + 1}")
                status_panel_lines.append(f"Points: {len(current_throw)}")

                # Draw red border around frame for recording indicator
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1] - 1, display_frame.shape[0] - 1),
                              (0, 0, pulse), 4)

            # Show previous throws status
            if show_previous_throws:
                status_panel_lines.append(f"SHOWING {throw_counter} PREVIOUS THROWS")

            # Draw status panel
            status_panel_width = 280
            status_panel_height = 35 + 25 * len(status_panel_lines)

            draw_panel(display_frame, "Status", status_panel_lines,
                       10, 10, status_panel_width, status_panel_height,
                       COLORS['status'], COLORS['status'])

            # Initialize panel height variable
            pred_panel_height = 0

            # Create prediction panel at top-right
            if prediction_enabled:
                prediction_panel_lines = ["Trajectory Prediction ACTIVE"]

                # Show landing point if available
                if prediction_active and last_valid_landing_point:
                    prediction_panel_lines.append(f"Predicted landing: X = {last_valid_landing_point[0]:.1f} cm")

                    # Add time to impact if available
                    if kalman_tracker and last_valid_landing_point:
                        time_to_impact = kalman_tracker.calculate_time_to_impact(last_valid_landing_point)
                        if time_to_impact is not None:
                            prediction_panel_lines.append(f"Time to impact: {time_to_impact:.2f} sec")

                # Draw prediction panel
                pred_panel_width = 300
                pred_panel_height = 35 + 25 * len(prediction_panel_lines)
                pred_panel_x = display_frame.shape[1] - pred_panel_width - 10

                draw_panel(display_frame, "Prediction", prediction_panel_lines,
                           pred_panel_x, 10, pred_panel_width, pred_panel_height,
                           COLORS['prediction'], COLORS['prediction'])

            # Show dodge command information if available
            if dodge_info:
                # Create dodge command panel
                dodge_panel_lines = [
                    f"Direction: {dodge_info['direction'].upper()}",
                    f"Distance: {dodge_info['distance']:.1f} cm",
                    f"Target position: {dodge_info['target']:.1f} cm"
                ]

                # Draw dodge panel
                dodge_panel_width = 300
                dodge_panel_height = 35 + 25 * len(dodge_panel_lines)
                dodge_panel_x = 10
                dodge_panel_y = 10 + status_panel_height + 10  # Below status panel

                draw_panel(display_frame, "DODGE COMMAND ISSUED", dodge_panel_lines,
                           dodge_panel_x, dodge_panel_y, dodge_panel_width, dodge_panel_height,
                           COLORS['command'], COLORS['command'])

            # Show dodge status if connected
            if dodge_module and dodge_module.connected:
                if dodge_module.is_dodging:
                    # Create active dodge panel
                    dodge_status_lines = [
                        f"Target: X = {dodge_module.target_position:.1f} cm",
                    ]

                    if dodge_module.dodge_start_time:
                        elapsed = time.time() - dodge_module.dodge_start_time
                        dodge_status_lines.append(f"Time elapsed: {elapsed:.1f} sec")

                    # Draw dodge status panel
                    dodge_status_width = 250
                    dodge_status_height = 35 + 25 * len(dodge_status_lines)
                    dodge_status_x = display_frame.shape[1] - dodge_status_width - 10

                    # Calculate y position based on whether prediction panel exists
                    dodge_status_y = 10  # Default position
                    if pred_panel_height > 0:
                        dodge_status_y = 10 + pred_panel_height + 10  # Below prediction panel

                    # Use pulsing effect for active dodge
                    pulse = int(127 * np.sin(time.time() * 4) + 128)
                    dodge_color = (0, pulse, 0)

                    # Draw panel with pulsing border
                    draw_panel(display_frame, "ACTIVELY DODGING", dodge_status_lines,
                               dodge_status_x, dodge_status_y, dodge_status_width, dodge_status_height,
                               COLORS['command'], dodge_color)

            # Draw controls bar at bottom
            draw_controls_bar(display_frame, dodge_module and dodge_module.connected)

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
        elif key == ord('d'):  # Test dodge right
            if dodge_module and dodge_module.connected and robot_position is not None:
                dodge_module.test_dodge_right(20.0)
                print("Test dodge right initiated")
        elif key == ord('a'):   # Test dodge left
            if dodge_module and dodge_module.connected and robot_position is not None:
                dodge_module.test_dodge_left(20.0)
                print("Test dodge left initiated")
                if dodge_module.connected:
                    command = {"command": "dodge", "parameters": {"direction": "left", "distance": 20}}
                    #print(f"Sending command: {command}")  # Debug
                    success = dodge_module.send_command(command)
                    if success:
                        print("Command sent successfully")
                    else:
                        print("Failed to send command")

    # Clean up
    if dodge_module and dodge_module.connected:
        dodge_module.close() #come back to this

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

    # Create a polyline for the trajectory
    points_array = np.array(pixel_points, np.int32)
    points_array = points_array.reshape((-1, 1, 2))

    # Draw a semi-transparent area under the trajectory for better visibility
    overlay = frame.copy()

    # Draw filled curve with semi-transparency
    if len(points_array) > 4:  # Need enough points for smooth curve
        # Create a wider polygon for fill
        hull = cv2.convexHull(points_array)
        cv2.fillConvexPoly(overlay, hull, (*color, 50))  # Semi-transparent fill

        # Add the overlay with transparency
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw the main trajectory line with gradient thickness
    for i in range(1, len(pixel_points)):
        # Thicker lines for future points (opposite of real trajectory)
        progress = i / len(pixel_points)
        thickness = 1 + int((1 - progress) * 2)  # 1-3 pixels, thicker at beginning

        cv2.line(frame, pixel_points[i - 1], pixel_points[i], color, thickness, cv2.LINE_AA)

    # Draw circles at key points - start, end, and a few in between
    cv2.circle(frame, pixel_points[0], 5, color, -1)  # Start point
    cv2.circle(frame, pixel_points[-1], 5, color, -1)  # End point

    # Draw direction arrow near the middle of the trajectory
    mid_idx = len(pixel_points) // 2
    if 0 < mid_idx < len(pixel_points) - 1:
        cv2.arrowedLine(frame,
                        pixel_points[mid_idx - 1],
                        pixel_points[mid_idx + 1],
                        color, 2, tipLength=0.3)


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

    # Create pulsing effect
    pulse = int(127 * np.sin(time.time() * 4) + 128)

    # Draw filled circle with pulsing outer rings
    cv2.circle(frame, (x, y), 8, COLORS['prediction'], -1)  # Filled circle
    cv2.circle(frame, (x, y), 15, (0, pulse, pulse), 2)  # Inner ring
    cv2.circle(frame, (x, y), 25, COLORS['prediction'], 2)  # Outer ring

    # Draw crosshairs with better visibility
    cv2.line(frame, (x - 20, y), (x - 10, y), COLORS['prediction'], 2)  # Left
    cv2.line(frame, (x + 10, y), (x + 20, y), COLORS['prediction'], 2)  # Right
    cv2.line(frame, (x, y - 20), (x, y - 10), COLORS['prediction'], 2)  # Top
    cv2.line(frame, (x, y + 10), (x, y + 20), COLORS['prediction'], 2)  # Bottom

    # Add text label with better visibility
    label = f"X: {landing_point[0]:.1f} cm"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

    # Background for text
    cv2.rectangle(frame,
                  (x + 15 - 5, y - text_size[1] - 5),
                  (x + 15 + text_size[0] + 5, y + 5),
                  COLORS['bg_dark'], -1)

    cv2.putText(frame, label, (x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['prediction'], 2)


def draw_coordinate_grid(frame, homography_matrix, playground_dims, grid_spacing=20):
    """
    Draw a more subtle coordinate grid on the frame.

    Args:
        frame: Frame to draw on
        homography_matrix: Homography matrix for coordinate transformation
        playground_dims: Playground dimensions (width, height) in cm
        grid_spacing: Grid spacing in cm
    """
    width, height = playground_dims

    # Draw grid lines with reduced opacity
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

        # Draw vertical grid line - thinner and more subtle
        cv2.line(frame, start_pixel, end_pixel, COLORS['grid'], 1, cv2.LINE_AA)

        # Add coordinate label at the top with better visibility
        if x % (grid_spacing * 2) == 0:  # Label every other line to avoid clutter
            label_point = (int(start_pixel[0]), int(start_pixel[1]) - 10)

            # Draw background for better readability
            label = f"{x}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame,
                          (label_point[0] - text_size[0] // 2 - 3, label_point[1] - text_size[1] - 3),
                          (label_point[0] + text_size[0] // 2 + 3, label_point[1] + 3),
                          COLORS['bg_dark'], -1)

            cv2.putText(frame, label,
                        (label_point[0] - text_size[0] // 2, label_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['grid_label'], 1)

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

        # Draw horizontal grid line - thinner and more subtle
        cv2.line(frame, start_pixel, end_pixel, COLORS['grid'], 1, cv2.LINE_AA)

        # Add coordinate label on the left with better visibility
        if y % (grid_spacing * 2) == 0:  # Label every other line to avoid clutter
            label_point = (int(start_pixel[0]) - 20, int(start_pixel[1]))

            # Draw background for better readability
            label = f"{y}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame,
                          (label_point[0] - 3, label_point[1] - text_size[1] // 2 - 3),
                          (label_point[0] + text_size[0] + 3, label_point[1] + text_size[1] // 2 + 3),
                          COLORS['bg_dark'], -1)

            cv2.putText(frame, label, label_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['grid_label'], 1)

    # Draw axes labels with better visibility
    # X axis label
    x_label = "X (cm)"
    x_label_pixel = inverse_transform_point((width / 2, -10), homography_matrix)
    x_label_x = int(x_label_pixel[0])
    x_label_y = int(x_label_pixel[1])

    # Background for X label
    x_text_size = cv2.getTextSize(x_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame,
                  (x_label_x - x_text_size[0] // 2 - 5, x_label_y - x_text_size[1] - 5),
                  (x_label_x + x_text_size[0] // 2 + 5, x_label_y + 5),
                  COLORS['bg_dark'], -1)

    cv2.putText(frame, x_label,
                (x_label_x - x_text_size[0] // 2, x_label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Y axis label
    y_label = "Y (cm)"
    y_label_pixel = inverse_transform_point((-10, height / 2), homography_matrix)
    y_label_x = int(y_label_pixel[0])
    y_label_y = int(y_label_pixel[1])

    # Background for Y label
    y_text_size = cv2.getTextSize(y_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame,
                  (y_label_x - 5, y_label_y - y_text_size[1] // 2 - 5),
                  (y_label_x + y_text_size[0] + 5, y_label_y + y_text_size[1] // 2 + 5),
                  COLORS['bg_dark'], -1)    

    cv2.putText(frame, y_label,
                (y_label_x, y_label_y + y_text_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


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
    parser = argparse.ArgumentParser(description='Ball-dodging robot vision system')
    parser.add_argument('--disable_dodge', action='store_true', help='Disable dodge command module')
    args = parser.parse_args()
    test_combined_system(disable_dodge=args.disable_dodge)