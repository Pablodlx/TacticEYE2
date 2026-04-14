"""
Optical Flow Tracker for Heatmap Generation
=============================================

Provides frame-to-frame optical flow tracking to:
1. Enable fallback when keypoint detection fails
2. Validate camera motion consistency
3. Smooth player trajectories across frames

Author: TacticEYE2
Date: 2026-04-14
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


class OpticalFlowTracker:
    """
    Tracks camera motion and provides fallback positions via optical flow.

    When keypoint-based homography estimation fails or has low confidence,
    optical flow can propagate known positions forward to maintain trajectory
    continuity and estimate player locations.

    Algorithm:
    1. Compute Farneback optical flow between consecutive frames
    2. For each known position in frame t-1, look up flow vector at that location
    3. Propagate position to frame t using flow: pos_t = pos_t-1 + flow
    4. Convert pixel positions back to field coordinates
    """

    def __init__(self, fps: float = 30.0, window_size: int = 15, levels: int = 3):
        """
        Initialize optical flow tracker.

        Args:
            fps: Frame rate for temporal calculations
            window_size: Averaging window size for Farneback algorithm (must be odd)
            levels: Pyramid levels for hierarchical flow calculation
        """
        self.fps = fps
        self.prev_frame_gray = None
        self.prev_positions_px = {}  # track_id → (x_px, y_px)

        # Optical flow parameters
        self.window_size = window_size
        self.levels = levels
        self.pyr_scale = 0.5
        self.iterations = 3
        self.poly_n = 5
        self.poly_sigma = 1.2

        # Tracking state
        self.flow_confidence = {}  # track_id → confidence (0-1)
        self.flow_history = {}  # track_id → deque of flow vectors

    def compute_flow(self, frame_gray: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between previous and current frame.

        Args:
            frame_gray: Current grayscale frame (H×W)

        Returns:
            Flow field (H×W×2) where flow[y,x] = [flow_x, flow_y]
            Returns None if previous frame not available
        """
        if self.prev_frame_gray is None:
            return None

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame_gray,
                frame_gray,
                None,
                pyr_scale=self.pyr_scale,
                levels=self.levels,
                winsize=self.window_size,
                iterations=self.iterations,
                poly_n=self.poly_n,
                poly_sigma=self.poly_sigma,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
            return flow
        except Exception as e:
            logger.warning(f"Optical flow computation failed: {e}")
            return None

    def update(
        self,
        frame_rgb: np.ndarray,
        current_detections_px: List[Tuple[float, float]],
        track_ids: List[int],
        fps: float = None
    ) -> Dict[int, Tuple[float, float]]:
        """
        Compute optical flow and propagate previous positions forward.

        Args:
            frame_rgb: Current frame (H×W×3)
            current_detections_px: List of (x, y) in pixels for detected players this frame
            track_ids: Corresponding track IDs for current detections
            fps: Frame rate (overrides init value if provided)

        Returns:
            fallback_positions: Dict mapping track_id → (x_px, y_px) via optical flow
                               Only includes tracked players NOT detected in current frame
        """
        if fps is not None:
            self.fps = fps

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Compute optical flow
        flow = self.compute_flow(frame_gray)
        fallback_positions = {}

        if flow is not None:
            # For each tracked player NOT detected in this frame, propagate via flow
            current_track_ids = set(track_ids)

            for track_id, prev_pos_px in self.prev_positions_px.items():
                # Only propagate if player not detected this frame (missing/occluded)
                if track_id not in current_track_ids:
                    new_pos_px = self._propagate_via_flow(
                        prev_pos_px, flow, frame_gray.shape
                    )

                    if new_pos_px is not None:
                        fallback_positions[track_id] = new_pos_px

                        # Reduce confidence each frame (optical flow accumulates error)
                        self.flow_confidence[track_id] = max(
                            0.1,  # Minimum confidence
                            self.flow_confidence.get(track_id, 1.0) * 0.95
                        )

        # Update state with current detections
        self.prev_positions_px = {}
        for track_id, pos_px in zip(track_ids, current_detections_px):
            self.prev_positions_px[track_id] = pos_px
            self.flow_confidence[track_id] = 1.0  # Reset confidence for detected players

        # Update previous frame for next iteration
        self.prev_frame_gray = frame_gray.copy()

        return fallback_positions

    def _propagate_via_flow(
        self,
        pos_px: Tuple[float, float],
        flow: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Optional[Tuple[float, float]]:
        """
        Propagate a single position forward using optical flow.

        Args:
            pos_px: Previous position (x_px, y_px)
            flow: Optical flow field (H×W×2)
            frame_shape: Frame shape (H, W)

        Returns:
            New position (x_px, y_px) or None if out of bounds
        """
        x_float, y_float = pos_px
        h, w = frame_shape

        # Clip to image bounds
        x_int = int(np.round(np.clip(x_float, 0, w - 1)))
        y_int = int(np.round(np.clip(y_float, 0, h - 1)))

        # Get flow vector at this position (with bounds check)
        if 0 <= y_int < h and 0 <= x_int < w:
            flow_x, flow_y = flow[y_int, x_int]

            # Propagate position
            new_x = x_float + flow_x
            new_y = y_float + flow_y

            # Verify new position is within image
            if 0 <= new_x < w and 0 <= new_y < h:
                return (new_x, new_y)

        return None

    def get_confidence(self, track_id: int) -> float:
        """
        Get confidence score for fallback position of a player.

        Args:
            track_id: Player ID

        Returns:
            Confidence 0.0-1.0 (1.0 = just detected, <1.0 = propagated via flow)
        """
        return self.flow_confidence.get(track_id, 0.0)

    def has_previous_frame(self) -> bool:
        """Check if previous frame is available for flow computation."""
        return self.prev_frame_gray is not None

    def reset(self):
        """Reset tracker state (e.g., on new sequence or match restart)."""
        self.prev_frame_gray = None
        self.prev_positions_px = {}
        self.flow_confidence = {}
        self.flow_history = {}
        logger.info("Optical flow tracker reset")


class CameraMotionDetector:
    """
    Detects significant camera motion which can invalidate homography.

    Uses optical flow magnitude to detect pan, zoom, or other camera movements
    that would require homography recalibration.
    """

    def __init__(self, motion_threshold_ratio: float = 0.3):
        """
        Initialize detector.

        Args:
            motion_threshold_ratio: Fraction of pixels with significant flow
                                   to trigger motion detection
        """
        self.motion_threshold_ratio = motion_threshold_ratio
        self.prev_frame_gray = None
        self.motion_history = deque(maxlen=30)  # 1 second @ 30fps

    def detect_motion(
        self,
        frame_rgb: np.ndarray,
        flow_threshold: float = 1.0
    ) -> Tuple[bool, float]:
        """
        Detect significant camera motion.

        Args:
            frame_rgb: Current frame (H×W×3)
            flow_threshold: Pixel magnitude threshold for "significant" motion

        Returns:
            (has_motion: bool, motion_magnitude: float in 0.0-1.0)
        """
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        has_motion = False
        motion_magnitude = 0.0

        if self.prev_frame_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame_gray, frame_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )

            # Calculate flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

            # Calculate percentage of pixels with significant motion
            significant_motion_pixels = np.sum(magnitude > flow_threshold)
            total_pixels = magnitude.shape[0] * magnitude.shape[1]
            motion_ratio = significant_motion_pixels / total_pixels

            # Normalize magnitude for consistency
            motion_magnitude = min(1.0, np.mean(magnitude) / 5.0)  # ~5 px average = max

            has_motion = motion_ratio > self.motion_threshold_ratio

            if has_motion:
                logger.debug(
                    f"Significant camera motion detected: "
                    f"{motion_ratio*100:.1f}% pixels moved (threshold: {self.motion_threshold_ratio*100:.1f}%)"
                )

        self.prev_frame_gray = frame_gray.copy()
        self.motion_history.append((has_motion, motion_magnitude))

        return has_motion, motion_magnitude

    def has_consistent_motion(self, window_size: int = 5) -> bool:
        """
        Check if motion has been consistent over recent frames.

        Useful to distinguish camera motion (consistent) from player movement
        (localized/non-uniform).

        Args:
            window_size: Number of recent frames to consider

        Returns:
            True if motion detected in most recent frames (consistent)
        """
        if len(self.motion_history) < window_size:
            return False

        recent = list(self.motion_history)[-window_size:]
        motion_count = sum(1 for has_motion, _ in recent if has_motion)

        return motion_count >= window_size * 0.7  # 70% of frames have motion
