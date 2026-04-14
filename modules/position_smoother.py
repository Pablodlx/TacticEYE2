"""
Position Smoothing with Kalman Filtering
==========================================

Provides trajectory smoothing to reduce measurement noise and jitter
in projected player positions using Kalman filtering.

Author: TacticEYE2
Date: 2026-04-14
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class KalmanFilter1D:
    """
    1D Kalman Filter for smoothing single-dimension positions.

    Reduces noise in noisy position measurements while maintaining
    responsiveness to actual movements.

    State: [position, velocity]
    Measurement: [position]
    """

    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 1.0,
        initial_value: float = 0.0,
        initial_estimate_error: float = 1.0
    ):
        """
        Initialize 1D Kalman filter.

        Args:
            process_variance: Q - How much we expect the true value to change between steps
                            Very small (0.001-0.01) = stable trajectories
                            Larger (0.1-1.0) = responsive to changes
            measurement_variance: R - Uncertainty in measurements
                                 Larger = trust measurements less
                                 Smaller = trust measurements more
            initial_value: Initial position estimate
            initial_estimate_error: Initial uncertainty in estimate
        """
        self.q = process_variance
        self.r = measurement_variance

        # State: [position, velocity]
        self.state = np.array([initial_value, 0.0])

        # Estimate error (covariance)
        self.p = initial_estimate_error

        # Kalman gain
        self.k = 0.0

    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement and return smoothed estimate.

        Args:
            measurement: Noisy measurement of position

        Returns:
            Smoothed position estimate
        """
        # Prediction step
        # New velocity = old velocity (tends to stay constant)
        # New position = old position + velocity
        predicted_state = self.state.copy()
        predicted_state[0] += predicted_state[1]  # pos += vel

        # Predict error covariance
        predicted_error = self.p + self.q

        # Update step
        # Calculate Kalman gain: how much to trust measurement vs prediction
        self.k = predicted_error / (predicted_error + self.r)

        # Update position based on measurement
        measurement_residual = measurement - predicted_state[0]
        self.state[0] = predicted_state[0] + self.k * measurement_residual

        # Update velocity estimate (how fast position is changing)
        # This helps predict future positions better
        self.state[1] = predicted_state[1] + self.k * measurement_residual / max(0.01, 1.0)

        # Update error covariance
        self.p = (1 - self.k) * predicted_error

        return self.state[0]

    def get_state(self) -> Tuple[float, float]:
        """Get current state (position, velocity)."""
        return float(self.state[0]), float(self.state[1])


class KalmanFilterPositionSmoother:
    """
    2D Position Smoother using independent Kalman filters for X and Y.

    Maintains separate Kalman filters for each player's X and Y coordinates
    to smooth trajectories while preserving natural movement patterns.
    """

    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 1.0,
        adaptive: bool = True
    ):
        """
        Initialize position smoother.

        Args:
            process_variance: Q - Process noise (how much we expect position to change)
                            Smaller = smoother but more lag
                            Larger = responsive but noisier
            measurement_variance: R - Measurement noise (sensor uncertainty)
            adaptive: If True, adjust variances based on velocity/confidence
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.adaptive = adaptive

        # Track separate Kalman filters for X and Y
        self.kalman_filters = {}  # track_id → {'x': KF, 'y': KF}

        # Tracking statistics for adaptive adjustment
        self.velocity_history = {}  # track_id → deque of velocities
        self.confidence_history = {}  # track_id → deque of confidences

    def smooth(
        self,
        track_id: int,
        measured_pos: np.ndarray,
        confidence: float = 1.0,
        is_detected: bool = True
    ) -> np.ndarray:
        """
        Smooth position using Kalman filtering.

        Args:
            track_id: Player ID
            measured_pos: Measured position [X, Y] in meters
            confidence: Confidence in measurement (0-1)
            is_detected: Whether position was directly detected (vs propagated)

        Returns:
            Smoothed position [X, Y]
        """
        # Initialize filters if needed
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = {
                'x': KalmanFilter1D(
                    process_variance=self.process_variance,
                    measurement_variance=self.measurement_variance,
                    initial_value=measured_pos[0]
                ),
                'y': KalmanFilter1D(
                    process_variance=self.process_variance,
                    measurement_variance=self.measurement_variance,
                    initial_value=measured_pos[1]
                )
            }
            self.velocity_history[track_id] = deque(maxlen=30)
            self.confidence_history[track_id] = deque(maxlen=30)

        kfs = self.kalman_filters[track_id]
        self.confidence_history[track_id].append(confidence)

        # Adaptively adjust measurement variance based on confidence
        if self.adaptive and confidence < 1.0:
            # Lower confidence = higher measurement noise = trust measurement less
            adjusted_r = self.measurement_variance / (confidence + 0.1)
        else:
            adjusted_r = self.measurement_variance

        # Update X and Y separately
        smoothed_x = kfs['x'].update(measured_pos[0])
        smoothed_y = kfs['y'].update(measured_pos[1])

        # Temporarily adjust R for this update
        old_r_x = kfs['x'].r
        old_r_y = kfs['y'].r
        kfs['x'].r = adjusted_r
        kfs['y'].r = adjusted_r

        smoothed_x = kfs['x'].update(measured_pos[0])
        smoothed_y = kfs['y'].update(measured_pos[1])

        # Restore original R
        kfs['x'].r = old_r_x
        kfs['y'].r = old_r_y

        result = np.array([smoothed_x, smoothed_y])

        # Track velocity for statistics
        if len(self.velocity_history[track_id]) > 0:
            prev_pos = self.kalman_filters[track_id]['x'].get_state()[0]  # Last X
            velocity = np.linalg.norm(result - np.array([prev_pos, 0]))
            self.velocity_history[track_id].append(velocity)

        return result

    def get_velocity(self, track_id: int) -> Optional[np.ndarray]:
        """
        Get estimated velocity of player (from Kalman filter state).

        Args:
            track_id: Player ID

        Returns:
            Velocity [vx, vy] in m/s, or None if not tracked
        """
        if track_id not in self.kalman_filters:
            return None

        kfs = self.kalman_filters[track_id]
        vx = kfs['x'].get_state()[1]
        vy = kfs['y'].get_state()[1]

        return np.array([vx, vy])

    def get_statistics(self, track_id: int) -> Dict:
        """
        Get smoothing statistics for a player.

        Args:
            track_id: Player ID

        Returns:
            Dict with 'avg_velocity', 'avg_confidence', etc.
        """
        if track_id not in self.kalman_filters:
            return {}

        stats = {
            'velocity': self.get_velocity(track_id),
        }

        if len(self.velocity_history[track_id]) > 0:
            velocities = list(self.velocity_history[track_id])
            stats['avg_velocity'] = np.mean(velocities)
            stats['max_velocity'] = np.max(velocities)

        if len(self.confidence_history[track_id]) > 0:
            confidences = list(self.confidence_history[track_id])
            stats['avg_confidence'] = np.mean(confidences)
            stats['min_confidence'] = np.min(confidences)

        return stats

    def reset_player(self, track_id: int):
        """Reset smoothing state for a player (e.g., after extended occlusion)."""
        if track_id in self.kalman_filters:
            del self.kalman_filters[track_id]
            del self.velocity_history[track_id]
            del self.confidence_history[track_id]
            logger.debug(f"Reset smoothing state for player {track_id}")

    def reset_all(self):
        """Reset all smoothing state."""
        self.kalman_filters = {}
        self.velocity_history = {}
        self.confidence_history = {}
        logger.info("Reset all smoothing state")


class TrajectoryValidator:
    """
    Validates player trajectories for physical plausibility.

    Checks velocity, acceleration, and distance traveled constraints
    to detect unrealistic movements that might indicate errors.
    """

    def __init__(
        self,
        max_velocity_ms: float = 10.0,
        max_accel_ms2: float = 5.0,
        fps: float = 30.0
    ):
        """
        Initialize validator.

        Args:
            max_velocity_ms: Maximum plausible velocity (m/s) - ~36 km/h for professionals
            max_accel_ms2: Maximum plausible acceleration (m/s²)
            fps: Frame rate for temporal calculations
        """
        self.max_velocity_ms = max_velocity_ms
        self.max_accel_ms2 = max_accel_ms2
        self.fps = fps
        self.position_history = {}  # track_id → deque of positions

    def validate(
        self,
        track_id: int,
        position: np.ndarray,
        frame_idx: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if position is physically plausible.

        Args:
            track_id: Player ID
            position: Position [X, Y] in meters
            frame_idx: Current frame index

        Returns:
            (is_valid: bool, reason_if_invalid: str or None)
        """
        if track_id not in self.position_history:
            self.position_history[track_id] = deque(maxlen=3)

        history = self.position_history[track_id]

        # Check 1: Velocity
        if len(history) > 0:
            prev_data = history[-1]
            prev_pos = prev_data['pos']
            time_delta = (frame_idx - prev_data['frame']) / self.fps

            if time_delta > 0:
                distance = np.linalg.norm(position - prev_pos)
                velocity = distance / time_delta

                if velocity > self.max_velocity_ms:
                    return False, f"Velocity {velocity:.1f} m/s exceeds limit {self.max_velocity_ms} m/s"

        # Check 2: Acceleration
        if len(history) >= 2:
            prev_data = history[-1]
            prev_prev_data = history[-2]

            prev_pos = prev_data['pos']
            prev_prev_pos = prev_prev_data['pos']

            time_delta_1 = (prev_data['frame'] - prev_prev_data['frame']) / self.fps
            time_delta_2 = (frame_idx - prev_data['frame']) / self.fps

            if time_delta_1 > 0 and time_delta_2 > 0:
                vel_1 = np.linalg.norm(prev_pos - prev_prev_pos) / time_delta_1
                vel_2 = np.linalg.norm(position - prev_pos) / time_delta_2

                accel = abs(vel_2 - vel_1) / time_delta_2

                if accel > self.max_accel_ms2:
                    return False, f"Acceleration {accel:.1f} m/s² exceeds limit {self.max_accel_ms2} m/s²"

        # All checks passed
        history.append({'pos': position.copy(), 'frame': frame_idx})
        return True, None

    def reset_player(self, track_id: int):
        """Reset history for a player."""
        if track_id in self.position_history:
            del self.position_history[track_id]
