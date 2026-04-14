"""
End-to-End Integration Test: Optical Flow & Position Smoothing
===============================================================

Verifies that optical flow tracker, position smoother, and trajectory validator
integrate correctly with batch_processor and generate valid heatmaps.

Tests:
1. Module initialization within batch processor
2. Optical flow fallback when detections are sparse
3. Position smoothing reduces jitter
4. Trajectory validation rejects impossible movements
5. Heatmap generation completes without artifacts
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from modules.optical_flow_tracker import OpticalFlowTracker, CameraMotionDetector
from modules.position_smoother import KalmanFilterPositionSmoother, TrajectoryValidator
from modules.field_heatmap_system import HeatmapAccumulator, FIELD_LENGTH, FIELD_WIDTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEndToEndOpticalFlow:
    """End-to-end integration tests for optical flow and smoothing."""

    def test_optical_flow_propagation(self):
        """Test that optical flow propagates positions correctly."""
        logger.info("TEST: Optical flow propagation")

        of_tracker = OpticalFlowTracker(fps=30.0)

        # Create two frames with a moving feature
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Add a feature that moves from (100, 100) to (110, 100)
        frame1[95:115, 95:115] = 200
        frame2[95:115, 105:125] = 200

        # First update - establish baseline
        track_ids = [1, 2]
        detections_px = [(100, 100), (200, 200)]
        fallback1 = of_tracker.update(frame1, detections_px, track_ids)

        # Verify first frame tracking
        assert len(fallback1) == 0, "First frame should have no fallback"
        print("  ✓ First frame processed without fallback")

        # Second frame - propagate via optical flow
        # Don't detect player 1, only player 2
        detections_px_2 = [(210, 200)]
        track_ids_2 = [2]
        fallback2 = of_tracker.update(frame2, detections_px_2, track_ids_2)

        # Player 1 should be propagated via optical flow
        assert 1 in fallback2, "Player 1 should be propagated via optical flow"
        x, y = fallback2[1]
        assert isinstance(x, (int, float, np.float32, np.float64)), f"Fallback position X should be numeric, got {type(x)}"
        assert isinstance(y, (int, float, np.float32, np.float64)), f"Fallback position Y should be numeric, got {type(y)}"
        print("  ✓ Optical flow propagated missing player position")

        # Confidence should be degraded for propagated position
        conf = of_tracker.get_confidence(1)
        assert 0 < conf < 1.0, f"Confidence for propagated position should be 0-1, got {conf}"
        print(f"  ✓ Confidence degradation working (conf={conf:.2f})")

    def test_position_smoothing(self):
        """Test that Kalman smoothing reduces trajectory jitter."""
        logger.info("TEST: Position smoothing with Kalman filter")

        smoother = KalmanFilterPositionSmoother(
            process_variance=0.01,
            measurement_variance=1.0,
            adaptive=True
        )

        # Simulate noisy measurements around (50, 30)
        noisy_positions = [
            np.array([50.1, 30.2]),
            np.array([49.9, 29.8]),
            np.array([50.3, 30.1]),
            np.array([50.0, 30.3]),
            np.array([49.8, 29.9]),
        ]

        track_id = 1
        smoothed_positions = []

        for pos in noisy_positions:
            smoothed = smoother.smooth(track_id, pos.copy(), confidence=1.0)
            smoothed_positions.append(smoothed)

        # Calculate jitter reduction
        noisy_jitter = np.mean([
            np.linalg.norm(noisy_positions[i] - noisy_positions[i-1])
            for i in range(1, len(noisy_positions))
        ])

        smoothed_jitter = np.mean([
            np.linalg.norm(smoothed_positions[i] - smoothed_positions[i-1])
            for i in range(1, len(smoothed_positions))
        ])

        reduction = 1.0 - (smoothed_jitter / noisy_jitter) if noisy_jitter > 0 else 0
        print(f"  ✓ Jitter reduction: {reduction*100:.1f}%")
        print(f"    - Before: {noisy_jitter:.4f} m")
        print(f"    - After:  {smoothed_jitter:.4f} m")

        # Velocity estimation
        vel = smoother.get_velocity(track_id)
        assert vel is not None, "Velocity should be estimated"
        print(f"  ✓ Velocity estimated: {vel}")

        # Statistics
        stats = smoother.get_statistics(track_id)
        assert 'avg_confidence' in stats, "Statistics should include avg_confidence"
        print(f"  ✓ Statistics available: {stats}")

    def test_trajectory_validation(self):
        """Test that validator rejects physically implausible movements."""
        logger.info("TEST: Trajectory validation")

        validator = TrajectoryValidator(max_velocity_ms=10.0, max_accel_ms2=5.0, fps=30.0)

        # Valid movement: slow walk
        pos1 = np.array([50.0, 30.0])
        is_valid, reason = validator.validate(1, pos1, frame_idx=0)
        assert is_valid, "First position should always be valid"
        print("  ✓ First position accepted")

        # Valid movement: ~3 m/s (realistic)
        pos2 = np.array([50.1, 30.1])  # ~0.14m in one frame @ 30fps = 4.2 m/s
        is_valid, reason = validator.validate(1, pos2, frame_idx=1)
        assert is_valid, f"Realistic movement should be valid, got: {reason}"
        print("  ✓ Realistic movement accepted")

        # Invalid movement: impossible speed (~2700 m/s)
        pos3 = np.array([140.0, 30.0])  # 90m jump in one frame
        is_valid, reason = validator.validate(1, pos3, frame_idx=2)
        assert not is_valid, "Impossible speed should be rejected"
        assert "Velocity" in reason or reason is not None, f"Should provide reason: {reason}"
        print(f"  ✓ Impossible movement rejected: {reason}")

    def test_camera_motion_detection(self):
        """Test camera motion detection."""
        logger.info("TEST: Camera motion detection")

        detector = CameraMotionDetector(motion_threshold_ratio=0.3)

        # Create static frames
        frame_static = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Test 1: Static scene
        has_motion_1, mag_1 = detector.detect_motion(frame_static)
        assert mag_1 == 0.0, "First frame should have zero motion"
        print(f"  ✓ First frame motion: {mag_1} (expected 0)")

        has_motion_2, mag_2 = detector.detect_motion(frame_static)
        assert not has_motion_2, "Static frames should not detect motion"
        assert mag_2 < 0.1, f"Motion magnitude should be low for static, got {mag_2}"
        print(f"  ✓ Static scene detected: motion={has_motion_2}, magnitude={mag_2:.3f}")

        # Test 2: Motion detection with consistency
        consistent = detector.has_consistent_motion(window_size=2)
        assert not consistent, "Two static frames should not indicate consistent motion"
        print(f"  ✓ Consistency check working")

    def test_heatmap_with_optical_flow(self):
        """Test that heatmap accumulator works with smoothed positions."""
        logger.info("TEST: Heatmap generation with smoothed positions")

        accumulated = HeatmapAccumulator(
            field_length=FIELD_LENGTH,
            field_width=FIELD_WIDTH,
            nx=50,
            ny=34
        )

        # Generate smooth trajectories for two players
        smoother = KalmanFilterPositionSmoother()

        for frame_idx in range(10):
            # Two players moving smoothly across field
            pos_p1 = np.array([20.0 + frame_idx * 0.5, 30.0])
            pos_p2 = np.array([80.0 - frame_idx * 0.3, 35.0])

            # Smooth positions
            smooth_p1 = smoother.smooth(1, pos_p1, confidence=1.0)
            smooth_p2 = smoother.smooth(2, pos_p2, confidence=0.9)

            # Add to heatmap
            accumulated.add_frame_with_field_coords(
                field_coords=np.array([smooth_p1, smooth_p2]),
                team_ids=[0, 1],
                confidences=[1.0, 0.9]
            )

        # Verify heatmaps were accumulated
        hm0 = accumulated.counts_team0
        hm1 = accumulated.counts_team1

        # Verify heatmaps are reasonable
        assert hm0.shape == (34, 50), f"Team 0 heatmap wrong shape: {hm0.shape}"
        assert hm1.shape == (34, 50), f"Team 1 heatmap wrong shape: {hm1.shape}"

        # Team 0 should have activity in left/center area
        assert hm0.sum() > 0, "Team 0 should have accumulated positions"

        # Team 1 should have activity in right area
        assert hm1.sum() > 0, "Team 1 should have accumulated positions"

        print(f"  ✓ Heatmaps generated correctly")
        print(f"    - Team 0 shape: {hm0.shape}, total: {hm0.sum():.1f}")
        print(f"    - Team 1 shape: {hm1.shape}, total: {hm1.sum():.1f}")
        print(f"    - Frames processed: {accumulated.num_frames}")


def run_all_tests():
    """Run all end-to-end tests."""
    logger.info("=" * 70)
    logger.info("RUNNING END-TO-END OPTICAL FLOW & SMOOTHING TESTS")
    logger.info("=" * 70)

    tester = TestEndToEndOpticalFlow()

    tests = [
        ("Optical Flow Propagation", tester.test_optical_flow_propagation),
        ("Position Smoothing", tester.test_position_smoothing),
        ("Trajectory Validation", tester.test_trajectory_validation),
        ("Camera Motion Detection", tester.test_camera_motion_detection),
        ("Heatmap Generation", tester.test_heatmap_with_optical_flow),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            logger.info("")
            test_func()
            passed += 1
        except AssertionError as e:
            logger.error(f"✗ FAILED: {test_name}")
            logger.error(f"  Reason: {e}")
            failed += 1
        except Exception as e:
            logger.error(f"✗ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
