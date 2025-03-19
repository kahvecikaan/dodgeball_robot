import cv2
import numpy as np
import time


def detect_ball_color(frame, lower_color, upper_color, min_radius=5):
    """
    Detect a ball in the frame based on color thresholding.

    Args:
        frame: Camera frame (BGR)
        lower_color: Lower bound of color range in HSV
        upper_color: Upper bound of color range in HSV
        min_radius: Minimum radius in pixels to consider a valid ball

    Returns:
        ball_pos: (x, y, radius) of the detected ball, or None if not detected
        mask: Binary mask showing the detected ball
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create color mask
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

    # Find the largest contour (presumably the ball)
    largest_contour = max(contours, key=cv2.contourArea)

    # Check if contour is big enough
    if cv2.contourArea(largest_contour) < np.pi * min_radius ** 2:
        return None, mask

    # Find the minimum enclosing circle
    ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

    # Additional check for circularity
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    # If the contour is approximately circular (circularity > 0.6) and big enough
    if circularity > 0.6 and radius > min_radius:
        return (int(x), int(y), int(radius)), mask

    return None, mask


def main():
    # Initialize camera
    print("Starting camera...")
    camera = cv2.VideoCapture(0)  # Use default camera (usually the built-in webcam)

    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    # Allow camera to warm up
    time.sleep(1)

    # Default color range for an orange ball
    # You can adjust these values for your specific ball
    lower_color = np.array([5, 100, 100])  # Orange (low end)
    upper_color = np.array([15, 255, 255])  # Orange (high end)

    # Try to load saved color settings if they exist
    try:
        data = np.load('ball_color.npz')
        lower_color = data['lower_color']
        upper_color = data['upper_color']
        print("Loaded saved color calibration")
    except:
        print("Using default orange ball color settings")

    print("\nControls:")
    print("  q - Quit")
    print("  + - Increase minimum radius")
    print("  - - Decrease minimum radius")
    print("  c - Enter color calibration mode")

    # Initial settings
    min_radius = 10
    show_mask = False
    calibration_mode = False

    while True:
        # Read frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Detect ball
        ball_pos, mask = detect_ball_color(frame, lower_color, upper_color, min_radius)

        # Create output frame
        output = frame.copy()

        # Draw detection result
        if ball_pos:
            x, y, radius = ball_pos
            # Draw circle around the ball
            cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
            # Display position and radius
            cv2.putText(output, f"Position: ({x}, {y})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output, f"Radius: {radius}px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(output, "No ball detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display settings
        cv2.putText(output, f"Min radius: {min_radius}px", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the output
        cv2.imshow('Ball Detection', output)

        # Display mask if enabled
        if show_mask:
            cv2.imshow('Mask', mask)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit
            break
        elif key == ord('+') or key == ord('='):
            # Increase minimum radius
            min_radius += 1
            print(f"Min radius: {min_radius}px")
        elif key == ord('-') or key == ord('_'):
            # Decrease minimum radius
            min_radius = max(1, min_radius - 1)
            print(f"Min radius: {min_radius}px")
        elif key == ord('m'):
            # Toggle mask display
            show_mask = not show_mask
            if show_mask:
                cv2.imshow('Mask', mask)
            else:
                cv2.destroyWindow('Mask')
        elif key == ord('c'):
            # Enter color calibration mode
            calibration_mode = True
            cv2.destroyAllWindows()

            # Create color calibration window with trackbars
            cv2.namedWindow('Color Calibration')
            cv2.createTrackbar('H min', 'Color Calibration', lower_color[0], 179, lambda x: None)
            cv2.createTrackbar('S min', 'Color Calibration', lower_color[1], 255, lambda x: None)
            cv2.createTrackbar('V min', 'Color Calibration', lower_color[2], 255, lambda x: None)
            cv2.createTrackbar('H max', 'Color Calibration', upper_color[0], 179, lambda x: None)
            cv2.createTrackbar('S max', 'Color Calibration', upper_color[1], 255, lambda x: None)
            cv2.createTrackbar('V max', 'Color Calibration', upper_color[2], 255, lambda x: None)

            print("\nColor Calibration Mode")
            print("  Adjust sliders until only the ball is visible in the mask")
            print("  q - Save and exit calibration")
            print("  r - Reset to defaults")

            while calibration_mode:
                # Read frame
                ret, frame = camera.read()
                if not ret:
                    break

                # Get current trackbar values
                h_min = cv2.getTrackbarPos('H min', 'Color Calibration')
                s_min = cv2.getTrackbarPos('S min', 'Color Calibration')
                v_min = cv2.getTrackbarPos('V min', 'Color Calibration')
                h_max = cv2.getTrackbarPos('H max', 'Color Calibration')
                s_max = cv2.getTrackbarPos('S max', 'Color Calibration')
                v_max = cv2.getTrackbarPos('V max', 'Color Calibration')

                # Update color ranges
                lower_color = np.array([h_min, s_min, v_min])
                upper_color = np.array([h_max, s_max, v_max])

                # Detect ball with current settings
                ball_pos, mask = detect_ball_color(frame, lower_color, upper_color, min_radius)

                # Create output
                result = frame.copy()
                if ball_pos:
                    x, y, radius = ball_pos
                    cv2.circle(result, (x, y), radius, (0, 255, 0), 2)
                    cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

                # Show current HSV values
                hsv_text = f"H:{h_min}-{h_max}, S:{s_min}-{s_max}, V:{v_min}-{v_max}"
                cv2.putText(result, hsv_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display images
                cv2.imshow('Color Calibration', result)
                cv2.imshow('Mask', mask)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    # Save and exit calibration
                    calibration_mode = False
                    np.savez('ball_color.npz', lower_color=lower_color, upper_color=upper_color)
                    print("Color calibration saved")
                    print(f"Lower color: {lower_color}")
                    print(f"Upper color: {upper_color}")
                    break
                elif key == ord('r'):
                    # Reset to defaults
                    cv2.setTrackbarPos('H min', 'Color Calibration', 5)
                    cv2.setTrackbarPos('S min', 'Color Calibration', 100)
                    cv2.setTrackbarPos('V min', 'Color Calibration', 100)
                    cv2.setTrackbarPos('H max', 'Color Calibration', 15)
                    cv2.setTrackbarPos('S max', 'Color Calibration', 255)
                    cv2.setTrackbarPos('V max', 'Color Calibration', 255)

            # Clean up calibration windows
            cv2.destroyAllWindows()

    # Clean up
    camera.release()
    cv2.destroyAllWindows()
    print("Testing finished")


if __name__ == "__main__":
    main()