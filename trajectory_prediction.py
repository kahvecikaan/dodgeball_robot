"""
Trajectory prediction for ball-dodging robot vision system.
Handles prediction of ball trajectory based on historical positions.
"""

import time
from collections import deque
import cv2
import numpy as np


class KalmanTracker:
    """
    Kalman filter based tracker for ball trajectory prediction.
    """

    def __init__(self, initial_pos=None, process_noise=0.01, measurement_noise=0.1):
        """
        Initialize the Kalman filter tracker.

        Args:
            initial_pos: Initial position (x, y) if available
            process_noise: Process noise coefficient (how much we expect the ball's motion to vary)
            measurement_noise: Measurement noise coefficient (how accurate our measurements are)
        """
        # Initialize Kalman filter
        # State: [x, y, vx, vy] (position and velocity)
        self.kalman = cv2.KalmanFilter(4, 2)

        # State transition matrix (physics model)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx (assuming constant velocity, TODO?)
            [0, 0, 0, 1]  # vy = vy (assuming constant velocity, TODO?)
        ], np.float32)

        # Measurement matrix (we only measure position, not velocity) - they must be inferred from position changes
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Error covariance matrix (equal certainty about all state variables)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        # Initialize state
        if initial_pos is not None:
            self.kalman.statePost = np.array([
                [initial_pos[0]],
                [initial_pos[1]],
                [0],
                [0]
            ], np.float32)

        # Flag to check if filter has been initialized
        self.initialized = initial_pos is not None

        # Store history of positions for visualization and calculations
        self.history = deque(maxlen=30)
        self.timestamp_history = deque(maxlen=30)

    def update(self, pos, timestamp=None):
        """
        Update the tracker with a new position measurement.

        Args:
            pos: New position (x, y)
            timestamp: Timestamp of the measurement (optional)

        Returns:
            Corrected position (x, y)
        """
        if timestamp is None:
            timestamp = time.time()

        # Convert position to measurement format
        measurement = np.array([[pos[0]], [pos[1]]], np.float32)

        if not self.initialized:
            # Initialize the filter with first measurement
            self.kalman.statePost = np.array([
                [pos[0]],
                [pos[1]],
                [0],
                [0]
            ], np.float32)
            self.initialized = True
            corrected_pos = pos
        else:
            # Predict (modifies the internal Kalman state)
            self.kalman.predict()

            # Update with measurement
            corrected = self.kalman.correct(measurement)

            # Extract position
            corrected_pos = corrected[0, 0], corrected[1, 0]

        # Add to history
        self.history.append(corrected_pos)
        self.timestamp_history.append(timestamp)

        return corrected_pos

    def predict_trajectory(self, num_steps=30):
        """
        Predict future trajectory of the ball.

        Args:
            num_steps: Number of steps to predict

        Returns:
            trajectory: List of predicted positions [(x1, y1), (x2, y2), ...]
        """
        if not self.initialized:
            return []

        # Create a deep copy of the Kalman filter to simulate future positions without affecting the original filter
        temp_kalman = cv2.KalmanFilter(4, 2)
        temp_kalman.transitionMatrix = self.kalman.transitionMatrix.copy()
        temp_kalman.measurementMatrix = self.kalman.measurementMatrix.copy()
        temp_kalman.processNoiseCov = self.kalman.processNoiseCov.copy()
        temp_kalman.measurementNoiseCov = self.kalman.measurementNoiseCov.copy()
        temp_kalman.errorCovPost = self.kalman.errorCovPost.copy()
        temp_kalman.statePost = self.kalman.statePost.copy()

        # Generate predictions
        trajectory = []
        for _ in range(num_steps):
            # Predict next state
            predicted = temp_kalman.predict()

            # Extract position
            pos = (predicted[0, 0], predicted[1, 0])
            trajectory.append(pos)

        return trajectory

    def get_velocity(self):
        """
        Get the current velocity estimate.

        Returns:
            velocity: (vx, vy) velocity vector
        """
        if not self.initialized:
            return 0, 0

        state = self.kalman.statePost
        return state[2, 0], state[3, 0]

    def get_speed(self):
        """
        Get the current speed estimate (magnitude of velocity).

        Returns:
            speed: Speed in units/frame
        """
        vx, vy = self.get_velocity()
        return np.sqrt(vx ** 2 + vy ** 2)

    def calculate_time_to_impact(self, intersection_point):
        """
        Calculate time until ball reaches a specific point.

        Args:
            intersection_point: Point of interest (x, y)

        Returns:
            time_to_impact: Estimated time to impact in seconds, or None if not possible
        """
        if not self.initialized or len(self.history) < 2:
            return None

        # Get current position and velocity
        current_pos = self.history[-1]
        vx, vy = self.get_velocity()

        # Check if velocity is too small
        speed = np.sqrt(vx ** 2 + vy ** 2)
        if speed < 0.1:  # Very slow or stationary
            return None

        # Vector from current position to intersection point
        dx = intersection_point[0] - current_pos[0]
        dy = intersection_point[1] - current_pos[1]

        # Distance to intersection point
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # Calculate time to impact (distance / speed)
        time_to_impact = distance / speed

        return time_to_impact

    def get_history(self):
        """
        Get the history of ball positions.

        Returns:
            history: List of positions [(x1, y1), (x2, y2), ...]
        """
        return list(self.history)


def calculate_average_fps(timestamp_history):
    """
    Calculate average frames per second from timestamp history.

    Args:
        timestamp_history: List of timestamps

    Returns:
        fps: Average frames per second
    """
    if len(timestamp_history) < 2:
        return None

    # Calculate time differences between consecutive frames
    time_diffs = [timestamp_history[i] - timestamp_history[i - 1]
                  for i in range(1, len(timestamp_history))]

    # Average time difference
    avg_time_diff = sum(time_diffs) / len(time_diffs)

    # FPS = 1 / time_between_frames
    fps = 1.0 / avg_time_diff if avg_time_diff > 0 else 0

    return fps


def predict_trajectory_simple(ball_positions, timesteps=15):
    """
    Simple linear trajectory prediction without Kalman filter.
    Useful when we need a quick trajectory estimate.

    Args:
        ball_positions: List of recent ball positions [(x1, y1), (x2, y2), ...]
        timesteps: Number of future steps to predict

    Returns:
        trajectory: List of predicted future positions
    """
    if len(ball_positions) < 2:
        return []

    # Calculate average velocity from recent positions
    velocities = []
    for i in range(1, len(ball_positions)):
        dx = ball_positions[i][0] - ball_positions[i - 1][0]
        dy = ball_positions[i][1] - ball_positions[i - 1][1]
        velocities.append((dx, dy))

    # Average velocity
    avg_vx = sum(v[0] for v in velocities) / len(velocities)
    avg_vy = sum(v[1] for v in velocities) / len(velocities)

    # Last known position
    last_x, last_y = ball_positions[-1]

    # Predict future trajectory
    trajectory = []
    for i in range(1, timesteps + 1):
        next_x = last_x + avg_vx * i
        next_y = last_y + avg_vy * i
        trajectory.append((next_x, next_y))

    return trajectory


if __name__ == "__main__":
    print("Trajectory prediction module loaded successfully")