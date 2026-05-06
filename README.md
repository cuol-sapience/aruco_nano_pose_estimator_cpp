# aruco_nano_pose_estimator_cpp

ROS 2 Humble C++ node for ArUco marker detection and pose estimation using **Aruco Nano**.

The node:
- subscribes to one or more camera image topics,
- detects ArUco markers with ArUco Nano,
- refines marker corners with subpixel refinement,
- estimates marker pose with `SOLVEPNP_IPPE_SQUARE` followed by iterative PnP refinement,
- transforms the pose from camera frame to the drone body frame,
- optionally filters the marker position in the SLAM world frame using odometry,
- publishes raw and filtered pose outputs.

## Features

- ArUco Nano-based detection
- Subpixel corner refinement
- Iterative PnP pose estimation
- Multi-camera support
- Optional static-landmark Kalman filtering
- Optional debug image saving

## Assumptions

- ROS 2 **Humble**
- The drone body frame is **FRD**
- Camera extrinsics provided in the node are `^B T_C`:
  - camera frame to drone body frame
- SLAM odometry provides a body pose consistent with the same body-frame convention

## Dependencies

Typical dependencies:
- `rclcpp`
- `sensor_msgs`
- `geometry_msgs`
- `std_msgs`
- `nav_msgs`
- `cv_bridge`
- OpenCV
- ArUco Nano header/source integrated in this package

## Build

From your workspace root:

```bash
colcon build --packages-select aruco_nano_pose_estimator_cpp
source install/setup.bash
```

## Run

Example:

```bash
ros2 run aruco_nano_pose_estimator_cpp aruco_nano_pose_estimator
```

With parameters:

```bash
ros2 run aruco_nano_pose_estimator_cpp aruco_nano_pose_estimator --ros-args \
  -p marker_length:=0.2 \
  -p drone_frame_id:=drone_frd \
  -p slam_odom_topic:=/dlio/odom_node/odom \
  -p use_track_filter:=true \
  -p save_debug_images:=false
```


```bash
ros2 run aruco_nano_pose_estimator_cpp aruco_nano_pose_estimator --ros-args \
  -p marker_length:=1.0 \
  -p dictionary_name:=DICT_4X4_50 \
  -p drone_frame_id:=drone_base \
  -p save_debug_images:=true \
  -p debug_output_dir:=/mnt/nova_ssd/said_stuff/aruco_debug/aruco_nano_debug_pre \
  -p debug_max_save_rate_hz:=24.0 \
  -p preprocess_enable:=true \
  -p preprocess_try_original_first:=true \
  -p preprocess_use_gamma:=true \
  -p preprocess_gamma:=1.35 \
  -p preprocess_use_clahe:=true \
  -p preprocess_clahe_clip_limit:=2.5 \
  -p preprocess_clahe_tile_grid_size:=8 \
  -p preprocess_use_median_blur:=false \
  -p preprocess_use_adaptive_threshold:=true \
  -p preprocess_adaptive_block_size:=31 \
  -p preprocess_adaptive_c:=5.0 \
  -p preprocess_use_otsu_threshold:=true \
  -p use_track_filter:=false
```
## Published Topics

- `/aruco/pose`  
  Final selected pose output in the drone body frame

- `/aruco/raw_pose`  
  Raw pose from PnP before filtering

- `/aruco/landing_place_type`  
  Marker semantic label

- `/aruco/message`  
  Human-readable debug/status string

## Subscribed Topics

- Camera image topics configured in the source
- SLAM odometry topic, default:

```text
/dlio/odom_node/odom
```

## Important Parameters

- `marker_length`  
  Physical marker side length in meters

- `drone_frame_id`  
  Output body frame name

- `dictionary_name`  
  ArUco dictionary name

- `processing_period_ms`  
  Processing timer period

- `slam_odom_topic`  
  Odometry topic used for world-frame filtering

- `use_track_filter`  
  Enable static-landmark filtering

- `publish_raw_pose`  
  Publish raw PnP pose

- `max_odom_time_diff_sec`  
  Maximum allowed image/odometry timestamp difference

- `max_track_age_sec`  
  Maximum age for prediction-only publishing

- `save_debug_images`  
  Enable writing annotated debug images

- `debug_output_dir`  
  Path for saved debug images

## Notes

- The filter assumes the marker is **static in the world**.
- If both cameras observe the same marker in the same cycle, the node keeps only one final measurement for `/aruco/pose`.
- If the SLAM frame convention is inconsistent with the drone FRD frame, filtered output will be wrong.
- Camera intrinsics and extrinsics should be verified carefully before flight use.

## Recommended Validation

Check:
- raw pose vs filtered pose,
- frame consistency,
- marker pose continuity during short occlusions,
- timestamp alignment between images and odometry,
- reprojection error thresholds for your camera and marker size.
