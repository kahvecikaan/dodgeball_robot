import cv2
import numpy as np


def test_aruco_detection(camera_index=0):
    """
    Test ArUco marker detection with a live camera feed.

    Args:
        camera_index: Index of the camera to use
    """
    # Initialize camera
    camera = cv2.VideoCapture(camera_index)

    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    # Create ArUco dictionary and detection parameters
    # You may need to change the dictionary type based on what you printed
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    print("ArUco detection test started.")
    print("Press 'q' to exit.")

    while True:
        # Read frame from camera
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame. Check camera connection.")
            break

        # Create detector with the updated API
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)

        # Draw detected markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Print information about detected markers
            for i, marker_id in enumerate(ids.flatten()):
                # Calculate center of marker
                c = corners[i][0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))

                # Add text with marker ID
                cv2.putText(frame, f"ID: {marker_id}",
                            (center_x, center_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('ArUco Marker Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_aruco_detection()