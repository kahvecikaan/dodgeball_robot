"""
Collision detection for ball-dodging robot vision system.
Handles detection of potential collisions between the ball and robot.
"""

import numpy as np


def detect_collision_1d(ball_trajectory, robot_position, robot_width, robot_lane_y, ball_radius=5, fps = 30):
    """
    Detect if the ball will collide with the robot's one-dimensional path.

    Args:
        ball_trajectory: List of predicted (x, y) positions of the ball in playground coordinates
        robot_position: Current x-coordinate of the robot center in playground coordinates
        robot_width: Width of the robot in cm
        robot_lane_y: The y-coordinate of the robot's movement lane in playground coordinates
        ball_radius: Radius of the ball in cm
        fps: Average frames per second

    Returns:
        collision_detected: Boolean indicating if collision is detected
        collision_time: Time step when collision will occur
        collision_x: X-coordinate where collision will occur
        time_to_impact: Approximate time to impact in seconds (if fps is provided)
    """
    # Half width of the robot
    half_width = robot_width / 2

    # Check each point in the predicted trajectory
    for t, ball_pos in enumerate(ball_trajectory):
        ball_x, ball_y = ball_pos

        # Calculate y-distance to robot's lane
        y_distance = abs(ball_y - robot_lane_y)

        # Check if the ball is near the robot's lane in y-direction
        if y_distance < ball_radius:
            # Check if the ball's x-position overlaps with the robot
            if robot_position - half_width < ball_x < robot_position + half_width:
                return True, t, ball_x, t / fps

    return False, None, None, None


def line_intersection(line1, line2):
    """
    Calculate the intersection point of two lines.

    Args:
        line1: ((x1, y1), (x2, y2)) - Two points defining the first line
        line2: ((x3, y3), (x4, y4)) - Two points defining the second line

    Returns:
        intersection: (x, y) coordinates of intersection point, or None if lines are parallel
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines do not intersect

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def calculate_lane_intersection(ball_trajectory, robot_lane_y):
    """
    Calculate where the ball's trajectory intersects with the robot's lane.

    Args:
        ball_trajectory: List of predicted (x, y) positions of the ball in playground coordinates
        robot_lane_y: The y-coordinate of the robot's movement lane in playground coordinates

    Returns:
        intersection_point: (x, y) coordinates where the trajectory crosses the lane, or None
        intersection_time: Approximate time step when intersection occurs
    """
    if len(ball_trajectory) < 2:
        return None, None

    # Define the robot's lane as a horizontal line
    lane_line = ((0, robot_lane_y), (1000, robot_lane_y))  # Arbitrarily large x-range

    # Check each segment of the trajectory
    for i in range(1, len(ball_trajectory)):
        # Define trajectory segment
        trajectory_segment = (ball_trajectory[i - 1], ball_trajectory[i])

        # Check if segment crosses the lane
        y1, y2 = trajectory_segment[0][1], trajectory_segment[1][1]
        if (y1 <= robot_lane_y <= y2) or (y2 <= robot_lane_y <= y1):
            # Calculate intersection point
            intersection = line_intersection(lane_line, trajectory_segment)
            if intersection:
                # Linear interpolation to estimate the time of intersection
                total_y_diff = abs(y2 - y1)
                if total_y_diff > 0:
                    segment_fraction = abs(robot_lane_y - y1) / total_y_diff
                    intersection_time = i - 1 + segment_fraction
                    return intersection, intersection_time

    return None, None


def advanced_collision_detection(ball_trajectory, ball_radius, robot_position, robot_width, robot_lane_y, fps=30):
    """
    Advanced collision detection that considers the ball's trajectory more precisely.

    Args:
        ball_trajectory: List of predicted (x, y) positions of the ball in playground coordinates
        ball_radius: Radius of the ball in cm
        robot_position: Current x-coordinate of the robot center in playground coordinates
        robot_width: Width of the robot in cm
        robot_lane_y: The y-coordinate of the robot's movement lane in playground coordinates
        fps: Estimated frames per second for time calculations

    Returns:
        collision_detected: Boolean indicating if collision is detected
        collision_point: (x, y) coordinates where collision is predicted
        time_to_impact: Estimated time until impact in seconds
    """
    # Find where trajectory crosses the robot's lane
    intersection_point, intersection_time = calculate_lane_intersection(ball_trajectory, robot_lane_y)

    if intersection_point is None:
        return False, None, None

    # Check if the intersection is within the robot's width range
    robot_left = robot_position - robot_width / 2
    robot_right = robot_position + robot_width / 2

    if robot_left - ball_radius <= intersection_point[0] <= robot_right + ball_radius:
        # Calculate time to impact
        time_to_impact = intersection_time / fps if fps > 0 else None
        return True, intersection_point, time_to_impact

    return False, intersection_point, None


def visualize_collision(frame, homography_matrix, ball_trajectory, robot_position, robot_width,
                        robot_lane_y, collision_point=None, ball_radius=5):
    """
    Create a comprehensive visualization of the collision detection system.

    This function transforms playground coordinates (centimeters) back to image
    coordinates (pixels) for visualization, then draws the robot's lane, the robot's
    position, the ball's trajectory, and the predicted collision point if one exists.

    Args:
        frame: Camera frame
        homography_matrix: Transformation matrix for coordinate conversion
        ball_trajectory: List of predicted (x, y) positions in playground coordinates (cm)
        robot_position: Current x-coordinate of the robot in playground coordinates (cm)
        robot_width: Width of the robot in cm
        robot_lane_y: Y-coordinate of the robot's lane in playground coordinates (cm)
        collision_point: Point where collision is detected (if any)
        ball_radius: Radius of the ball in cm

    Returns:
        frame_viz: Frame with comprehensive collision visualization
    """
    import cv2
    from playground_setup import inverse_transform_point

    # Create a copy for visualization
    frame_viz = frame.copy()

    # Inverse transform for visualization (convert from cm to pixels)
    inv_homography = np.linalg.inv(homography_matrix)

    # ---- Draw robot lane ----
    lane_start = inverse_transform_point((0, robot_lane_y), inv_homography)
    lane_end = inverse_transform_point((1000, robot_lane_y), inv_homography)
    cv2.line(frame_viz,
             (int(lane_start[0]), int(lane_start[1])),
             (int(lane_end[0]), int(lane_end[1])),
             (0, 255, 0), 2)

    # Add lane label
    lane_mid = inverse_transform_point((robot_position + 50, robot_lane_y - 10), inv_homography)
    cv2.putText(frame_viz, "Robot Lane",
                (int(lane_mid[0]), int(lane_mid[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ---- Draw robot position ----
    robot_center = inverse_transform_point((robot_position, robot_lane_y), inv_homography)
    robot_left = inverse_transform_point((robot_position - robot_width / 2, robot_lane_y), inv_homography)
    robot_right = inverse_transform_point((robot_position + robot_width / 2, robot_lane_y), inv_homography)

    # Draw robot as a rectangle
    robot_height = 30  # Visual height in pixels for the robot representation
    top_offset = 15  # Offset from lane center to top of robot rectangle
    cv2.rectangle(frame_viz,
                  (int(robot_left[0]), int(robot_left[1] - top_offset)),
                  (int(robot_right[0]), int(robot_right[1] + (robot_height - top_offset))),
                  (0, 0, 255), 2)

    # Add robot label
    cv2.putText(frame_viz, "Robot",
                (int(robot_center[0] - 20), int(robot_center[1] - top_offset - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ---- Draw predicted trajectory ----
    # First draw lines connecting trajectory points
    for i in range(1, len(ball_trajectory)):
        pt1 = inverse_transform_point(ball_trajectory[i - 1], inv_homography)
        pt2 = inverse_transform_point(ball_trajectory[i], inv_homography)
        cv2.line(frame_viz,
                 (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])),
                 (255, 0, 0), 2)

    # Then draw circles representing the ball's size at each trajectory point
    for ball_pos in ball_trajectory:
        # Transform ball center to pixel coordinates
        ball_pixel = inverse_transform_point(ball_pos, inv_homography)

        # Transform a point ball_radius away to find pixel radius
        radius_point = inverse_transform_point((ball_pos[0] + ball_radius, ball_pos[1]), inv_homography)
        pixel_radius = int(np.sqrt((ball_pixel[0] - radius_point[0]) ** 2 +
                                   (ball_pixel[1] - radius_point[1]) ** 2))

        # Draw circle representing ball size (minimum 3 pixels for visibility)
        cv2.circle(frame_viz,
                   (int(ball_pixel[0]), int(ball_pixel[1])),
                   max(pixel_radius, 3),
                   (255, 0, 0), 1)

    # Add trajectory label
    if len(ball_trajectory) > 0:
        first_pt = inverse_transform_point(ball_trajectory[0], inv_homography)
        cv2.putText(frame_viz, "Predicted Trajectory",
                    (int(first_pt[0] + 10), int(first_pt[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # ---- Draw collision point if detected ----
    if collision_point:
        # Transform collision point to pixel coordinates
        collision_pixel = inverse_transform_point(collision_point, inv_homography)

        # Draw attention-grabbing red circle at collision point
        cv2.circle(frame_viz,
                   (int(collision_pixel[0]), int(collision_pixel[1])),
                   10, (0, 0, 255), -1)

        # Add collision warning text
        cv2.putText(frame_viz, "COLLISION!",
                    (int(collision_pixel[0] - 40), int(collision_pixel[1] - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Add collision coordinates
        cv2.putText(frame_viz, f"({collision_point[0]:.1f}, {collision_point[1]:.1f}) cm",
                    (int(collision_pixel[0] - 40), int(collision_pixel[1] + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ---- Add coordinate system indicator ----
    # Draw a small indicator in the corner showing coordinate system orientation
    origin_x, origin_y = 30, 30
    cv2.arrowedLine(frame_viz, (origin_x, origin_y), (origin_x + 20, origin_y), (0, 255, 255), 2)
    cv2.arrowedLine(frame_viz, (origin_x, origin_y), (origin_x, origin_y + 20), (0, 255, 255), 2)
    cv2.putText(frame_viz, "X", (origin_x + 25, origin_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame_viz, "Y", (origin_x - 5, origin_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return frame_viz


if __name__ == "__main__":
    print("Collision detection module loaded successfully")