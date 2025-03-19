"""
Playground setup for ball-dodging robot vision system.
Handles ArUco marker detection and coordinate system transformation.
"""

import os
import cv2
import numpy as np


def create_aruco_dict():
    """Create ArUco dictionary and parameters"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    return aruco_dict, parameters


def generate_aruco_markers(output_dir='aruco_markers', marker_ids=None, size=400):
    """
    Generate ArUco markers for printing.

    Args:
        output_dir: Directory to save marker images
        marker_ids: List of marker IDs to generate
        size: Size of marker images in pixels
    """
    if marker_ids is None:
        marker_ids = [0, 1, 2, 3, 42]
    aruco_dict, _ = create_aruco_dict()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save markers
    for marker_id in marker_ids:
        marker_img = np.zeros((size, size), dtype=np.uint8)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size, marker_img, 1)

        # Add text label
        labeled_img = cv2.copyMakeBorder(marker_img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(labeled_img, f"ArUco Marker ID: {marker_id}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

        # Save marker
        filename = os.path.join(output_dir, f"aruco_marker_{marker_id}.png")
        cv2.imwrite(filename, labeled_img)
        print(f"Generated marker ID {marker_id}: {filename}")


def detect_aruco_markers(frame, aruco_dict, parameters):
    """
    Detect ArUco markers in a frame.

    Args:
        frame: Camera frame
        aruco_dict: ArUco dictionary
        parameters: ArUco detection parameters

    Returns:
        corners: Detected marker corners
        ids: Detected marker IDs
        frame_markers: Frame with markers visualized
    """
    # Convert to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Visualize detected markers
    frame_markers = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_markers, corners, ids)

    return corners, ids, frame_markers


def setup_playground(frame, aruco_dict, parameters, marker_ids=None, playground_dims=(200, 300)):
    """
    Set up the playground coordinate system using ArUco markers.

    Args:
        frame: Camera frame
        aruco_dict: ArUco dictionary
        parameters: ArUco detection parameters
        marker_ids: IDs of the four corner markers [top-left, top-right, bottom-right, bottom-left]
        playground_dims: Real-world dimensions (width, length) in cm

    Returns:
        homography_matrix: Transformation matrix
        playground_corners: Four corners of the playground in pixel coordinates
        frame_markers: Frame with markers visualized
    """
    # Detect all markers
    if marker_ids is None:
        marker_ids = [0, 1, 2, 3]
    corners, ids, frame_markers = detect_aruco_markers(frame, aruco_dict, parameters)

    if ids is None:
        print("No ArUco markers detected")
        return None, None, frame_markers

    # Check if all required markers are detected
    # Convert ids to a flat array to make comparison easier
    ids_flat = ids.flatten() if ids is not None else []
    if not all(marker_id in ids_flat for marker_id in marker_ids):
        missing = [m for m in marker_ids if m not in ids_flat]
        print(f"Missing markers: {missing}")
        return None, None, frame_markers

    # Define playground corners in real-world coordinates (cm)
    playground_width, playground_length = playground_dims
    playground_corners_real = np.array([
        [0, 0],  # Top-left
        [playground_width, 0],  # Top-right
        [playground_width, playground_length],  # Bottom-right
        [0, playground_length]  # Bottom-left
    ], dtype=np.float32)

    # Extract marker centers
    marker_centers = []
    for marker_id in marker_ids:
        # Find index of this marker ID in the detected IDs
        idx = np.where(ids_flat == marker_id)[0][0]
        # Calculate center by averaging the 4 corners
        marker_center = np.mean(corners[idx][0], axis=0)
        marker_centers.append(marker_center)

        # Draw ID and center on visualization
        center_x, center_y = int(marker_center[0]), int(marker_center[1])
        cv2.circle(frame_markers, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.putText(frame_markers, f"ID: {marker_id}", (center_x + 10, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    marker_centers = np.array(marker_centers, dtype=np.float32)

    # Calculate homography matrix
    homography_matrix, _ = cv2.findHomography(marker_centers, playground_corners_real)

    # Draw playground boundaries on visualization
    # Connect the markers to show the playground boundaries
    for i in range(4):
        pt1 = (int(marker_centers[i][0]), int(marker_centers[i][1]))
        pt2 = (int(marker_centers[(i + 1) % 4][0]), int(marker_centers[(i + 1) % 4][1]))
        cv2.line(frame_markers, pt1, pt2, (0, 255, 255), 2)

    return homography_matrix, marker_centers, frame_markers


def transform_point(point, homography_matrix):
    """
    Transform a point from image coordinates to playground coordinates.

    Args:
        point: Point in image coordinates (x, y)
        homography_matrix: Transformation matrix

    Returns:
        transformed_point: Point in playground coordinates (cm)
    """
    # Reshape point for transformation
    point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)

    # Apply transformation
    transformed_point = cv2.perspectiveTransform(point_array, homography_matrix)

    # Return as simple tuple
    return transformed_point[0][0][0], transformed_point[0][0][1]


def inverse_transform_point(point, homography_matrix):
    """
    Transform a point from playground coordinates to image coordinates.

    Args:
        point: Point in playground coordinates (cm)
        homography_matrix: Transformation matrix

    Returns:
        transformed_point: Point in image coordinates (pixels)
    """
    # Calculate inverse homography
    inv_homography = np.linalg.inv(homography_matrix)

    # Reshape point for transformation
    point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)

    # Apply transformation
    transformed_point = cv2.perspectiveTransform(point_array, inv_homography)

    # Return as simple tuple
    return transformed_point[0][0][0], transformed_point[0][0][1]


if __name__ == "__main__":
    # Generate ArUco markers for printing when this file is executed directly
    generate_aruco_markers()

    print("\nInstructions for setting up your playground:")
    print("1. Print the generated ArUco markers (aruco_markers folder)")
    print("2. Place markers at the four corners of your playground area")
    print("   - Marker 0: Top-left corner")
    print("   - Marker 1: Top-right corner")
    print("   - Marker 2: Bottom-right corner")
    print("   - Marker 3: Bottom-left corner")
    print("3. Place marker 42 on your robot")
    print("\nMake sure markers are placed flat and are clearly visible to the camera")