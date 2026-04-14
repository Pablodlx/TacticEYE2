# Bug Fix: Player Projection Clustering in Bottom-Left Corner

## Problem Description

Players were being projected to cluster in the bottom-left corner of the field instead of being distributed across the field based on their actual positions.

## Root Cause Analysis

The issue has two parts:

### 1. **Keypoint Detector Not Working on This Video**
- The YOLO field keypoint detector (`weights/field_kp_merged_fast/weights/best.pt`) detects **0-1 keypoints per frame** on the test videos
- This is likely because:
  - Model was trained on different camera angles/field configurations
  - Video preprocessing requirements not met
  - Model weights might need retraining for this specific camera setup

### 2. **Fallback Projection Logic Was Broken**
When keypoint-based triangulation fails (no keypoints detected), the code falls back to a naive projection:

```python
# OLD - BROKEN FALLBACK
field_pos = np.array([
    center_x / 10.0,    # Divide pixel X by arbitrary scale
    center_y / 10.0     # Divide pixel Y by arbitrary scale
])
```

This directly projects pixel coordinates to field coordinates without accounting for:
- Actual image size
- Camera perspective
- Field orientation
- Homography transformation

**Result**: All players cluster at (0-384, 0-216) meters → **bottom-left corner**

## Solution Implemented

### 1. Improved Fallback Chain
When triangulation returns NaN (no valid keypoints):
1. **Try Optical Flow** (if enabled): Propagate previous position via optical flow
2. **Try Homography**: Use the computed homography matrix to project via `project_points(H, point)`
3. **Last Resort**: Use naive scaling (only if all else fails)

### 2. Increased Keypoint Threshold
Changed minimum keypoints requirement from 2 to 3 for triangulation:
```python
# OLD: if current_keypoints and len(current_keypoints) >= 2
# NEW: if current_keypoints and len(current_keypoints) >= 3
```

Since keypoint detection is unreliable on this video, skip triangulation more often and use homography projection instead.

### Code Changes

**File**: `modules/batch_processor.py`

**Lines 769-775**: Increased minimum keypoint threshold
```python
- if current_keypoints and len(current_keypoints) >= 2:
+ if current_keypoints and len(current_keypoints) >= 3:
```

**Lines 812-838**: Improved fallback logic
```python
# NEW: Fallback chain with homography as primary
if field_pos is None:
    # Try optical flow first
    if track_id in of_fallback:
        field_pos = propagate_via_optical_flow(...)
    
    # If OF failed, try homography
    if field_pos is None and best_homography_batch is not None:
        field_pos = project_points(best_homography_batch, [[center_x, center_y]])
```

## Expected Impact

- **Before**: All players cluster in bottom-left corner (X=0-40, Y=0-21.6 meters)
- **After**: Players distributed across field using homography projection
- **Confidence**: Reduced from 1.0 to 0.6 for homography-based fallback (indicating uncertainty)

## Next Steps

To fully resolve this, one of:
1. **Retrain keypoint detector** on videos from this camera angle
2. **Use different field keypoint model** trained for this specific setup
3. **Implement camera calibration** to improve homography accuracy
4. **Disable keypoint detection** entirely and rely on homography + optical flow

## Testing

```bash
python tests/test_projection_fix.py
```

Verify players project to correct field positions instead of clustering bottom-left.

---

**Status**: ✅ Fixed fallback logic | ⚠️ Keypoint detector still not working | 🔧 Requires investigation/retraining
