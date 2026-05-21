// =============================================================================
// aruco_nano_pose_estimator.cpp  (position-only output)
//
// Target: ROS 2 Humble / Iron / Jazzy, OpenCV 4.7+ (for solvePnPGeneric +
// SOLVEPNP_IPPE_SQUARE), Jetson Orin Nano with JetPack 5.1.x or 6.x.
//
// Companion to aruco_nano.h (Rafael Munoz-Salinas, v10, header-only). The
// header is used as-is; we configure it through DetectorParameters.
//
// IMPORTANT: This node only OUTPUTS the marker's position. The published
// PoseStamped messages keep their orientation field set to identity
// (x=0, y=0, z=0, w=1). Downstream consumers must ignore the orientation.
//
// We still compute full 6-DoF pose internally because we need the marker
// pose's rotation part for:
//   1. Disambiguating SOLVEPNP_IPPE_SQUARE (two solutions are mirror-image
//      poses about the marker plane; they differ in BOTH rotation AND
//      translation, so picking the wrong one gives the wrong position).
//   2. Rotating the camera-frame measurement covariance into the world
//      frame for the Kalman filter update.
//
// Key behaviour:
//   - aruco_nano refines corners internally on the raw grayscale; we do NOT
//     run a second cornerSubPix pass.
//   - IPPE_SQUARE pose ambiguity is resolved via solvePnPGeneric +
//     disambiguation against the KF prior (or marker-normal-toward-camera
//     fallback when no prior is available).
//   - Optional ITERATIVE refinement runs only from the chosen seed.
//   - 3-state position-only Kalman filter, static-landmark random-walk
//     process model, anisotropic measurement covariance (lateral vs depth)
//     built in camera frame and rotated to world frame.
//   - SLAM odometry is interpolated to the image timestamp (linear position,
//     SLERP orientation - orientation of the BODY in the world, which we
//     still need to express measurements in world coordinates).
//   - aruco_nano DetectorParameters are exposed as ROS parameters.
// =============================================================================

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/tracking.hpp>

#include "aruco_nano_pose_estimator_cpp/aruco_nano.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

class ArucoNanoPoseEstimator : public rclcpp::Node
{
public:
  ArucoNanoPoseEstimator()
  : Node("aruco_nano_pose_estimator")
  {
    // ---- ROS parameters --------------------------------------------------
    marker_length_ = this->declare_parameter<double>("marker_length", 1.0);
    drone_frame_id_ = this->declare_parameter<std::string>("drone_frame_id", "drone_frd");
    processing_period_ms_ = this->declare_parameter<int>("processing_period_ms", 33);
    dictionary_name_ = this->declare_parameter<std::string>("dictionary_name", "DICT_4X4_50");

    save_debug_images_ = this->declare_parameter<bool>("save_debug_images", false);
    debug_output_dir_ = this->declare_parameter<std::string>("debug_output_dir", "/tmp/aruco_nano_debug");
    debug_save_only_with_valid_detections_ =
      this->declare_parameter<bool>("debug_save_only_with_valid_detections", true);
    debug_max_save_rate_hz_ = this->declare_parameter<double>("debug_max_save_rate_hz", 2.0);

    slam_odom_topic_ = this->declare_parameter<std::string>("slam_odom_topic", "/dlio/odom_node/odom");
    use_track_filter_ = this->declare_parameter<bool>("use_track_filter", true);
    publish_raw_pose_ = this->declare_parameter<bool>("publish_raw_pose", true);
    max_odom_time_diff_sec_ = this->declare_parameter<double>("max_odom_time_diff_sec", 0.10);
    max_track_age_sec_ = this->declare_parameter<double>("max_track_age_sec", 1.0);
    odom_buffer_size_ = this->declare_parameter<int>("odom_buffer_size", 200);

    use_iterative_pnp_ = this->declare_parameter<bool>("use_iterative_pnp", true);
    max_reprojection_error_px_ = this->declare_parameter<double>("max_reprojection_error_px", 4.0);

    // Static-landmark world-position random-walk: sigma in m/sqrt(s).
    // For a static marker the value is small; it sets how fast the
    // covariance grows when the marker is out of view.
    kf_process_pos_rate_sigma_ =
      this->declare_parameter<double>("kf_process_pos_rate_sigma", 0.05);
    kf_measurement_noise_floor_m_ =
      this->declare_parameter<double>("kf_measurement_noise_floor_m", 0.03);
    kf_speed_noise_scale_ = this->declare_parameter<double>("kf_speed_noise_scale", 0.3);

    // Ratio (depth sigma) / (lateral sigma) clamp. Theoretical scaling for
    // planar marker pose gives the depth sigma as much larger than lateral;
    // this is a safety floor.
    kf_depth_to_lateral_ratio_min_ =
      this->declare_parameter<double>("kf_depth_to_lateral_ratio_min", 3.0);

    // aruco_nano detector parameters
    aruco_box_filter_size_ = this->declare_parameter<int>("aruco_box_filter_size", 15);
    aruco_thres_ = this->declare_parameter<int>("aruco_thres", 3);
    aruco_min_size_ = this->declare_parameter<int>("aruco_min_size", 10);
    aruco_max_attempts_per_candidate_ =
      this->declare_parameter<int>("aruco_max_attempts_per_candidate", 5);
    aruco_error_correction_rate_ =
      this->declare_parameter<double>("aruco_error_correction_rate", 0.3);
    aruco_detect_inverted_marker_ =
      this->declare_parameter<bool>("aruco_detect_inverted_marker", false);

    // Optional detection preprocessing (for difficult lighting)
    preprocess_enable_ = this->declare_parameter<bool>("preprocess_enable", false);
    preprocess_try_original_first_ =
      this->declare_parameter<bool>("preprocess_try_original_first", true);
    preprocess_use_gamma_ = this->declare_parameter<bool>("preprocess_use_gamma", false);
    preprocess_gamma_ = this->declare_parameter<double>("preprocess_gamma", 1.35);
    preprocess_use_clahe_ = this->declare_parameter<bool>("preprocess_use_clahe", true);
    preprocess_clahe_clip_limit_ =
      this->declare_parameter<double>("preprocess_clahe_clip_limit", 2.5);
    preprocess_clahe_tile_grid_size_ =
      this->declare_parameter<int>("preprocess_clahe_tile_grid_size", 8);
    preprocess_use_median_blur_ =
      this->declare_parameter<bool>("preprocess_use_median_blur", false);
    preprocess_median_blur_ksize_ =
      this->declare_parameter<int>("preprocess_median_blur_ksize", 5);
    preprocess_use_adaptive_threshold_ =
      this->declare_parameter<bool>("preprocess_use_adaptive_threshold", true);
    preprocess_adaptive_block_size_ =
      this->declare_parameter<int>("preprocess_adaptive_block_size", 31);
    preprocess_adaptive_c_ =
      this->declare_parameter<double>("preprocess_adaptive_c", 5.0);
    preprocess_use_otsu_threshold_ =
      this->declare_parameter<bool>("preprocess_use_otsu_threshold", false);

    validateParameters();

    if (save_debug_images_) {
      try {
        std::filesystem::create_directories(debug_output_dir_);
      } catch (const std::exception & e) {
        throw std::runtime_error(
          std::string("Failed to create debug_output_dir '") +
          debug_output_dir_ + "': " + e.what());
      }
    }

    if (preprocess_use_clahe_) {
      clahe_ = cv::createCLAHE(
        preprocess_clahe_clip_limit_,
        cv::Size(preprocess_clahe_tile_grid_size_, preprocess_clahe_tile_grid_size_));
    }
    if (preprocess_use_gamma_) {
      gamma_lut_ = makeGammaLut(preprocess_gamma_);
    }

    marker_labels_[1] = "Medical";
    marker_labels_[2] = "Supply";

    // Configure aruco_nano via DetectorParameters.
    dictionary_ = cv::aruco::getPredefinedDictionary(dictionaryNameToEnum(dictionary_name_));
    aruco_nano::DetectorParameters det_params;
    det_params.dicts = { dictionary_ };
    det_params.boxFilterSize = aruco_box_filter_size_;
    det_params.thres = aruco_thres_;
    det_params.minSize = aruco_min_size_;
    det_params.maxAttemptsPerCandidate = aruco_max_attempts_per_candidate_;
    det_params.errorCorrectionRate = aruco_error_correction_rate_;
    det_params.detectInvertedMarker = aruco_detect_inverted_marker_;
    // Use the vector ctor so aruco_nano does not append our dict again.
    detector_ = std::make_unique<aruco_nano::ArucoDetector>(det_params.dicts, det_params);

    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/aruco/pose", rclcpp::QoS(10));
    raw_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/aruco/raw_pose", rclcpp::QoS(10));
    type_pub_ = this->create_publisher<std_msgs::msg::String>(
      "/aruco/landing_place_type", rclcpp::QoS(10));
    msg_pub_  = this->create_publisher<std_msgs::msg::String>(
      "/aruco/message", rclcpp::QoS(10));

    slam_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      slam_odom_topic_,
      rclcpp::QoS(50),
      std::bind(&ArucoNanoPoseEstimator::slamOdomCallback, this, std::placeholders::_1));

    addCamera(
      "/camera/front/image_raw",
      cv::Matx44d(
         0.0, -0.57357644,  0.81915204,  0.0,
         1.0,  0.0,         0.0,         0.0,
         0.0,  0.81915204,  0.57357644,  0.0,
         0.0,   0.0,        0.0,        1.0),
      (cv::Mat_<double>(3, 3) <<
        1134.3171058472644, 0.0, 945.03097055585,
        0.0, 1134.9884203778502, 607.0921091018031,
        0.0, 0.0, 1.0),
      (cv::Mat_<double>(1, 5) <<
        0.02717223068021268, -0.060021841339080874, 0.0007009591920443065,
        -4.569105989855734e-05, 0.020871849677932104));

    addCamera(
      "/camera/down/image_raw",
      cv::Matx44d(
         0.0, -1.0, 0.0, 0.0,
         1.0,  0.0, 0.0, 0.0,
         0.0,  0.0, 1.0, 0.0,
         0.0,  0.0, 0.0, 1.0),
      (cv::Mat_<double>(3, 3) <<
        1134.3171058472644, 0.0, 945.03097055585,
        0.0, 1134.9884203778502, 607.0921091018031,
        0.0, 0.0, 1.0),
      (cv::Mat_<double>(1, 5) <<
        0.02717223068021268, -0.060021841339080874, 0.0007009591920443065,
        -4.569105989855734e-05, 0.020871849677932104));

    const auto image_qos = rclcpp::SensorDataQoS().keep_last(1);
    for (auto & [topic, camera] : cameras_) {
      camera->sub = this->create_subscription<sensor_msgs::msg::Image>(
        topic, image_qos,
        [this, topic](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          this->storeLatestImage(msg, topic);
        });
    }

    processing_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(processing_period_ms_),
      [this]() { this->onTimer(); });

    RCLCPP_INFO(
      this->get_logger(),
      "ArucoNanoPoseEstimator (position-only) started. marker_length=%.6f, "
      "dictionary=%s, frame_id=%s, period=%d ms, preprocess_enable=%s, "
      "save_debug_images=%s, slam_odom_topic=%s, use_track_filter=%s, "
      "aruco_error_correction_rate=%.2f",
      marker_length_, dictionary_name_.c_str(), drone_frame_id_.c_str(),
      processing_period_ms_,
      preprocess_enable_ ? "true" : "false",
      save_debug_images_ ? "true" : "false",
      slam_odom_topic_.c_str(),
      use_track_filter_ ? "true" : "false",
      aruco_error_correction_rate_);
  }

private:
  // ---- Types ---------------------------------------------------------------

  struct CameraCalibration
  {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
  };

  struct CameraState
  {
    CameraCalibration calibration;
    cv::Matx44d T_drone_camera = cv::Matx44d::eye();

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub;

    sensor_msgs::msg::Image::ConstSharedPtr latest_msg;
    std::mutex latest_msg_mutex;

    std::string debug_topic_tag;
    std::mutex debug_state_mutex;
    double last_debug_save_time_sec{-1.0};
    uint64_t debug_image_sequence{0U};
  };

  // Body pose in the world from SLAM odometry. The orientation here is the
  // BODY's orientation in the world; we still need it to express marker
  // positions in world coordinates. It is NOT the marker's orientation.
  struct SlamOdomSample
  {
    double t_sec{0.0};
    cv::Vec3d p_world_body{0.0, 0.0, 0.0};
    std::array<double, 4> q_world_body{0.0, 0.0, 0.0, 1.0};  // (x, y, z, w)
    cv::Vec3d v_body{0.0, 0.0, 0.0};
  };

  struct InterpolatedOdom
  {
    bool valid{false};
    double t_sec{0.0};
    cv::Vec3d p_world_body{0.0, 0.0, 0.0};
    std::array<double, 4> q_world_body{0.0, 0.0, 0.0, 1.0};
    cv::Vec3d v_body{0.0, 0.0, 0.0};
  };

  // 3-state Kalman filter tracking marker POSITION in the world frame.
  // No orientation is stored; we only need where the marker is.
  struct MarkerTrack
  {
    bool initialized{false};
    int marker_id{-1};
    std::string label;
    cv::KalmanFilter kf;
    double last_prediction_t_sec{0.0};
    double last_measurement_t_sec{0.0};

    MarkerTrack()
    : kf(3, 3, 0, CV_64F)
    {
      kf.transitionMatrix = cv::Mat::eye(3, 3, CV_64F);
      kf.measurementMatrix = cv::Mat::eye(3, 3, CV_64F);
      cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-4));
      cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-2));
      cv::setIdentity(kf.errorCovPost, cv::Scalar(1.0));
      kf.statePost = cv::Mat::zeros(3, 1, CV_64F);
      kf.statePre = cv::Mat::zeros(3, 1, CV_64F);
    }
  };

  struct CycleMeasurementCandidate
  {
    bool valid{false};
    int marker_id{-1};
    std::string label;
    geometry_msgs::msg::PoseStamped pose_msg;
    std::string source_topic;
    std::string output_kind;
    double reprojection_error_px{std::numeric_limits<double>::infinity()};
  };

  struct DetectionCandidateImage
  {
    std::string name;
    cv::Mat detect_image;
  };

  struct DetectionSelection
  {
    bool valid{false};
    std::string candidate_name{"none"};
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    int relevant_count{0};
    int total_count{0};
    double area_score{0.0};
  };

  // ---- State ---------------------------------------------------------------

  std::map<std::string, std::shared_ptr<CameraState>> cameras_;
  std::map<int, std::string> marker_labels_;
  std::map<int, MarkerTrack> tracks_;
  std::map<int, CycleMeasurementCandidate> cycle_measurement_candidates_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr raw_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr type_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr msg_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr slam_odom_sub_;
  rclcpp::TimerBase::SharedPtr processing_timer_;

  cv::aruco::Dictionary dictionary_;
  std::unique_ptr<aruco_nano::ArucoDetector> detector_;

  mutable std::mutex odom_buffer_mutex_;
  std::deque<SlamOdomSample> odom_buffer_;

  double marker_length_{1.0};
  std::string drone_frame_id_{"drone_frd"};
  int processing_period_ms_{33};
  std::string dictionary_name_{"DICT_4X4_50"};

  bool save_debug_images_{false};
  std::string debug_output_dir_{"/tmp/aruco_nano_debug"};
  bool debug_save_only_with_valid_detections_{true};
  double debug_max_save_rate_hz_{2.0};

  std::string slam_odom_topic_;
  bool use_track_filter_{true};
  bool publish_raw_pose_{true};
  double max_odom_time_diff_sec_{0.10};
  double max_track_age_sec_{1.0};
  int odom_buffer_size_{200};

  bool use_iterative_pnp_{true};
  double max_reprojection_error_px_{4.0};

  double kf_process_pos_rate_sigma_{0.05};
  double kf_measurement_noise_floor_m_{0.03};
  double kf_speed_noise_scale_{0.3};
  double kf_depth_to_lateral_ratio_min_{3.0};

  int aruco_box_filter_size_{15};
  int aruco_thres_{3};
  int aruco_min_size_{10};
  int aruco_max_attempts_per_candidate_{5};
  double aruco_error_correction_rate_{0.3};
  bool aruco_detect_inverted_marker_{false};

  bool preprocess_enable_{false};
  bool preprocess_try_original_first_{true};
  bool preprocess_use_gamma_{false};
  double preprocess_gamma_{1.35};
  bool preprocess_use_clahe_{true};
  double preprocess_clahe_clip_limit_{2.5};
  int preprocess_clahe_tile_grid_size_{8};
  bool preprocess_use_median_blur_{false};
  int preprocess_median_blur_ksize_{5};
  bool preprocess_use_adaptive_threshold_{true};
  int preprocess_adaptive_block_size_{31};
  double preprocess_adaptive_c_{5.0};
  bool preprocess_use_otsu_threshold_{false};

  cv::Ptr<cv::CLAHE> clahe_;
  cv::Mat gamma_lut_;

  // ---- Static utilities ----------------------------------------------------

  static int dictionaryNameToEnum(const std::string & name)
  {
    static const std::map<std::string, int> dict_map = {
      {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
      {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
      {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
      {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
      {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
      {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
      {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
      {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
      {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
      {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
      {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
      {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
      {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
      {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
      {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
      {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
      {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL}};

    const auto it = dict_map.find(name);
    if (it == dict_map.end()) {
      throw std::runtime_error("Unsupported dictionary_name: " + name);
    }
    return it->second;
  }

  static std::string sanitizeTopicName(const std::string & topic)
  {
    std::string sanitized;
    sanitized.reserve(topic.size());
    for (char ch : topic) {
      sanitized.push_back(std::isalnum(static_cast<unsigned char>(ch)) ? ch : '_');
    }
    while (!sanitized.empty() && sanitized.front() == '_') {
      sanitized.erase(sanitized.begin());
    }
    if (sanitized.empty()) sanitized = "camera";
    return sanitized;
  }

  static double timeMsgToSec(const builtin_interfaces::msg::Time & t)
  {
    return static_cast<double>(t.sec) + 1e-9 * static_cast<double>(t.nanosec);
  }

  static builtin_interfaces::msg::Time secToTimeMsg(double t_sec)
  {
    builtin_interfaces::msg::Time msg;
    if (t_sec < 0.0) { msg.sec = 0; msg.nanosec = 0u; return msg; }
    const double sec_part = std::floor(t_sec);
    msg.sec = static_cast<int32_t>(sec_part);
    msg.nanosec = static_cast<uint32_t>(std::round((t_sec - sec_part) * 1e9));
    if (msg.nanosec >= 1000000000u) { msg.sec += 1; msg.nanosec -= 1000000000u; }
    return msg;
  }

  static std::array<double, 4> normalizeQuaternion(const std::array<double, 4> & q)
  {
    const double n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n <= 1e-12) return {0.0, 0.0, 0.0, 1.0};
    return {q[0]/n, q[1]/n, q[2]/n, q[3]/n};
  }

  // SLERP between two unit quaternions (x, y, z, w). Used ONLY for
  // interpolating the SLAM body orientation between odom samples - this is
  // a body-frame interpolation, not a marker-orientation filter.
  static std::array<double, 4> slerpQuaternion(
    const std::array<double, 4> & q0_in,
    const std::array<double, 4> & q1_in,
    double t)
  {
    const auto q0 = normalizeQuaternion(q0_in);
    auto q1 = normalizeQuaternion(q1_in);

    double dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3];
    if (dot < 0.0) { for (auto & v : q1) v = -v; dot = -dot; }

    constexpr double kSlerpLinearThreshold = 0.9995;
    if (dot > kSlerpLinearThreshold) {
      std::array<double, 4> r = {
        q0[0] + t * (q1[0] - q0[0]),
        q0[1] + t * (q1[1] - q0[1]),
        q0[2] + t * (q1[2] - q0[2]),
        q0[3] + t * (q1[3] - q0[3])};
      return normalizeQuaternion(r);
    }
    const double theta_0 = std::acos(std::clamp(dot, -1.0, 1.0));
    const double sin_theta_0 = std::sin(theta_0);
    const double theta = theta_0 * t;
    const double sin_theta = std::sin(theta);
    const double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
    const double s1 = sin_theta / sin_theta_0;
    std::array<double, 4> r = {
      s0 * q0[0] + s1 * q1[0],
      s0 * q0[1] + s1 * q1[1],
      s0 * q0[2] + s1 * q1[2],
      s0 * q0[3] + s1 * q1[3]};
    return normalizeQuaternion(r);
  }

  static cv::Mat makeGammaLut(double gamma)
  {
    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
      const double v = std::pow(static_cast<double>(i) / 255.0, gamma);
      lut.at<unsigned char>(0, i) = static_cast<unsigned char>(
        std::round(std::clamp(v, 0.0, 1.0) * 255.0));
    }
    return lut;
  }

  static cv::Matx33d quaternionToRotationMatrix(const std::array<double, 4> & q_in)
  {
    const auto q = normalizeQuaternion(q_in);
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    return cv::Matx33d(
      1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),       2.0 * (x*z + y*w),
      2.0 * (x*y + z*w),       1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w),
      2.0 * (x*z - y*w),       2.0 * (y*z + x*w),       1.0 - 2.0 * (x*x + y*y));
  }

  static cv::Vec3d rotateVec(const cv::Matx33d & R, const cv::Vec3d & v)
  {
    return cv::Vec3d(
      R(0, 0) * v[0] + R(0, 1) * v[1] + R(0, 2) * v[2],
      R(1, 0) * v[0] + R(1, 1) * v[1] + R(1, 2) * v[2],
      R(2, 0) * v[0] + R(2, 1) * v[1] + R(2, 2) * v[2]);
  }

  static double quadrilateralArea(const std::vector<cv::Point2f> & corners)
  {
    if (corners.size() < 4U) return 0.0;
    return std::fabs(cv::contourArea(corners));
  }

  static cv::Matx44d matrixFromPose(
    const cv::Vec3d & p, const std::array<double, 4> & q)
  {
    cv::Matx44d T = cv::Matx44d::eye();
    const cv::Matx33d R = quaternionToRotationMatrix(q);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) T(r, c) = R(r, c);
    }
    T(0, 3) = p[0]; T(1, 3) = p[1]; T(2, 3) = p[2];
    return T;
  }

  static cv::Matx44d invertTransform(const cv::Matx44d & T)
  {
    cv::Matx33d R(
      T(0, 0), T(0, 1), T(0, 2),
      T(1, 0), T(1, 1), T(1, 2),
      T(2, 0), T(2, 1), T(2, 2));
    const cv::Matx33d Rt = R.t();
    const cv::Vec3d t(T(0, 3), T(1, 3), T(2, 3));
    const cv::Vec3d tinv = -rotateVec(Rt, t);

    cv::Matx44d Tinv = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) Tinv(r, c) = Rt(r, c);
    }
    Tinv(0, 3) = tinv[0]; Tinv(1, 3) = tinv[1]; Tinv(2, 3) = tinv[2];
    return Tinv;
  }

  static cv::Vec3d applyTransform(const cv::Matx44d & T, const cv::Vec3d & v)
  {
    const cv::Vec4d h(v[0], v[1], v[2], 1.0);
    const cv::Vec4d r = T * h;
    return cv::Vec3d(r[0], r[1], r[2]);
  }

  // ---- Validation ----------------------------------------------------------

  void validateParameters() const
  {
    if (marker_length_ <= 0.0) throw std::runtime_error("marker_length must be > 0");
    if (processing_period_ms_ <= 0) throw std::runtime_error("processing_period_ms must be > 0");
    if (debug_max_save_rate_hz_ < 0.0)
      throw std::runtime_error("debug_max_save_rate_hz must be >= 0");
    if (max_odom_time_diff_sec_ < 0.0 || max_track_age_sec_ < 0.0)
      throw std::runtime_error("max_odom_time_diff_sec and max_track_age_sec must be >= 0");
    if (odom_buffer_size_ <= 1)
      throw std::runtime_error("odom_buffer_size must be > 1");
    if (max_reprojection_error_px_ <= 0.0)
      throw std::runtime_error("max_reprojection_error_px must be > 0");
    if (kf_process_pos_rate_sigma_ <= 0.0 || kf_measurement_noise_floor_m_ <= 0.0)
      throw std::runtime_error(
        "kf_process_pos_rate_sigma and kf_measurement_noise_floor_m must be > 0");
    if (kf_depth_to_lateral_ratio_min_ < 1.0)
      throw std::runtime_error("kf_depth_to_lateral_ratio_min must be >= 1");
    if (aruco_error_correction_rate_ < 0.0 || aruco_error_correction_rate_ > 1.0)
      throw std::runtime_error("aruco_error_correction_rate must be in [0, 1]");
    if (preprocess_gamma_ <= 0.0) throw std::runtime_error("preprocess_gamma must be > 0");
    if (preprocess_clahe_clip_limit_ <= 0.0)
      throw std::runtime_error("preprocess_clahe_clip_limit must be > 0");
    if (preprocess_clahe_tile_grid_size_ <= 0)
      throw std::runtime_error("preprocess_clahe_tile_grid_size must be > 0");
    if (preprocess_median_blur_ksize_ <= 0 || (preprocess_median_blur_ksize_ % 2) == 0)
      throw std::runtime_error("preprocess_median_blur_ksize must be a positive odd integer");
    if (preprocess_adaptive_block_size_ <= 1 || (preprocess_adaptive_block_size_ % 2) == 0)
      throw std::runtime_error("preprocess_adaptive_block_size must be an odd integer > 1");
  }

  // ---- Camera registration -------------------------------------------------

  void addCamera(
    const std::string & topic, const cv::Matx44d & T_drone_camera,
    const cv::Mat & camera_matrix, const cv::Mat & dist_coeffs)
  {
    auto camera = std::make_shared<CameraState>();
    camera->T_drone_camera = T_drone_camera;
    camera->calibration.camera_matrix = camera_matrix.clone();
    camera->calibration.dist_coeffs = dist_coeffs.clone();
    camera->debug_topic_tag = sanitizeTopicName(topic);
    cameras_[topic] = camera;
  }

  // ---- Odometry buffer & interpolation -------------------------------------

  void slamOdomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    SlamOdomSample s;
    s.t_sec = timeMsgToSec(msg->header.stamp);
    s.p_world_body = cv::Vec3d(
      msg->pose.pose.position.x,
      msg->pose.pose.position.y,
      msg->pose.pose.position.z);
    s.q_world_body = normalizeQuaternion({
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z,
      msg->pose.pose.orientation.w});
    s.v_body = cv::Vec3d(
      msg->twist.twist.linear.x,
      msg->twist.twist.linear.y,
      msg->twist.twist.linear.z);

    std::lock_guard<std::mutex> lock(odom_buffer_mutex_);
    if (odom_buffer_.empty() || s.t_sec >= odom_buffer_.back().t_sec) {
      odom_buffer_.push_back(s);
    } else {
      auto it = std::lower_bound(
        odom_buffer_.begin(), odom_buffer_.end(), s.t_sec,
        [](const SlamOdomSample & a, double b) { return a.t_sec < b; });
      odom_buffer_.insert(it, s);
    }
    while (static_cast<int>(odom_buffer_.size()) > odom_buffer_size_) {
      odom_buffer_.pop_front();
    }
  }

  InterpolatedOdom interpolateOdomAt(double t_query) const
  {
    InterpolatedOdom out;
    std::lock_guard<std::mutex> lock(odom_buffer_mutex_);
    if (odom_buffer_.empty()) return out;

    const auto & front = odom_buffer_.front();
    const auto & back = odom_buffer_.back();

    if (t_query <= front.t_sec) {
      if (front.t_sec - t_query > max_odom_time_diff_sec_) return out;
      out.valid = true;
      out.t_sec = front.t_sec;
      out.p_world_body = front.p_world_body;
      out.q_world_body = front.q_world_body;
      out.v_body = front.v_body;
      return out;
    }
    if (t_query >= back.t_sec) {
      if (t_query - back.t_sec > max_odom_time_diff_sec_) return out;
      out.valid = true;
      out.t_sec = back.t_sec;
      out.p_world_body = back.p_world_body;
      out.q_world_body = back.q_world_body;
      out.v_body = back.v_body;
      return out;
    }

    auto it = std::upper_bound(
      odom_buffer_.begin(), odom_buffer_.end(), t_query,
      [](double v, const SlamOdomSample & s) { return v < s.t_sec; });
    if (it == odom_buffer_.begin() || it == odom_buffer_.end()) return out;
    const auto & b = *it;
    const auto & a = *(it - 1);

    const double dt = b.t_sec - a.t_sec;
    if (dt <= 1e-9) {
      out.valid = true;
      out.t_sec = a.t_sec;
      out.p_world_body = a.p_world_body;
      out.q_world_body = a.q_world_body;
      out.v_body = a.v_body;
      return out;
    }
    const double u = std::clamp((t_query - a.t_sec) / dt, 0.0, 1.0);

    out.valid = true;
    out.t_sec = t_query;
    out.p_world_body = a.p_world_body + u * (b.p_world_body - a.p_world_body);
    out.q_world_body = slerpQuaternion(a.q_world_body, b.q_world_body, u);
    out.v_body = a.v_body + u * (b.v_body - a.v_body);
    return out;
  }

  static double computeSpeedNorm(const cv::Vec3d & v)
  {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  }

  // ---- Image plumbing ------------------------------------------------------

  void storeLatestImage(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg, const std::string & topic_name)
  {
    const auto it = cameras_.find(topic_name);
    if (it == cameras_.end()) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Received image for unknown topic: %s", topic_name.c_str());
      return;
    }
    std::lock_guard<std::mutex> lock(it->second->latest_msg_mutex);
    it->second->latest_msg = msg;
  }

  void onTimer()
  {
    cycle_measurement_candidates_.clear();
    processLatestFrames();
    publishCycleMeasurementCandidates();
    publishPredictedTracks();
    pruneStaleTracks();
  }

  void processLatestFrames()
  {
    for (auto & [topic_name, camera] : cameras_) {
      sensor_msgs::msg::Image::ConstSharedPtr msg;
      {
        std::lock_guard<std::mutex> lock(camera->latest_msg_mutex);
        if (!camera->latest_msg) continue;
        msg = camera->latest_msg;
        camera->latest_msg.reset();
      }
      processImage(msg, topic_name, *camera);
    }
  }

  // ---- Detection preprocessing & selection ---------------------------------

  void buildDetectionCandidates(
    const cv::Mat & gray, std::vector<DetectionCandidateImage> & candidates) const
  {
    candidates.clear();
    candidates.reserve(5);

    if (!preprocess_enable_) {
      candidates.push_back({"gray_original", gray});
      return;
    }
    if (preprocess_try_original_first_) {
      candidates.push_back({"gray_original", gray});
    }

    cv::Mat enhanced = gray;
    bool have_nontrivial_enhancement = false;
    if (preprocess_use_gamma_) {
      cv::Mat tmp; cv::LUT(enhanced, gamma_lut_, tmp); enhanced = tmp;
      have_nontrivial_enhancement = true;
    }
    if (preprocess_use_clahe_ && clahe_) {
      cv::Mat tmp; clahe_->apply(enhanced, tmp); enhanced = tmp;
      have_nontrivial_enhancement = true;
    }
    if (preprocess_use_median_blur_) {
      cv::Mat tmp; cv::medianBlur(enhanced, tmp, preprocess_median_blur_ksize_); enhanced = tmp;
      have_nontrivial_enhancement = true;
    }
    if (have_nontrivial_enhancement || !preprocess_try_original_first_) {
      candidates.push_back({"gray_enhanced", enhanced});
    }
    if (preprocess_use_adaptive_threshold_) {
      cv::Mat adaptive;
      cv::adaptiveThreshold(
        enhanced, adaptive, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
        preprocess_adaptive_block_size_, preprocess_adaptive_c_);
      candidates.push_back({"adaptive_threshold", adaptive});
    }
    if (preprocess_use_otsu_threshold_) {
      cv::Mat otsu;
      cv::threshold(enhanced, otsu, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
      candidates.push_back({"otsu_threshold", otsu});
    }
    if (candidates.empty()) {
      candidates.push_back({"gray_original", gray});
    }
  }

  void selectBestDetection(const cv::Mat & gray, DetectionSelection & best)
  {
    best = DetectionSelection{};

    std::vector<DetectionCandidateImage> candidates;
    buildDetectionCandidates(gray, candidates);

    for (const auto & candidate : candidates) {
      std::vector<std::vector<cv::Point2f>> corners;
      std::vector<int> ids;
      detector_->detectMarkers(candidate.detect_image, corners, ids);

      int relevant_count = 0;
      double area_score = 0.0;
      for (size_t i = 0; i < ids.size(); ++i) {
        if (marker_labels_.find(ids[i]) != marker_labels_.end()) ++relevant_count;
        area_score += quadrilateralArea(corners[i]);
      }
      const int total_count = static_cast<int>(ids.size());

      bool replace = false;
      if (!best.valid && total_count > 0) replace = true;
      else if (relevant_count > best.relevant_count) replace = true;
      else if (relevant_count == best.relevant_count && total_count > best.total_count) replace = true;
      else if (relevant_count == best.relevant_count && total_count == best.total_count &&
               area_score > best.area_score) replace = true;

      if (replace) {
        best.valid = true;
        best.candidate_name = candidate.name;
        best.corners = std::move(corners);
        best.ids = std::move(ids);
        best.relevant_count = relevant_count;
        best.total_count = total_count;
        best.area_score = area_score;
      }
    }

    if (!best.valid && !candidates.empty()) {
      best.candidate_name = candidates.front().name;
    }
  }

  bool shouldSaveDebugImage(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    CameraState & camera, bool has_valid_detections, std::string & output_path)
  {
    output_path.clear();
    if (!save_debug_images_) return false;
    if (debug_save_only_with_valid_detections_ && !has_valid_detections) return false;

    const double stamp_sec = timeMsgToSec(msg->header.stamp);
    std::lock_guard<std::mutex> lock(camera.debug_state_mutex);
    if (debug_max_save_rate_hz_ > 0.0 && camera.last_debug_save_time_sec >= 0.0) {
      const double min_dt = 1.0 / debug_max_save_rate_hz_;
      if ((stamp_sec - camera.last_debug_save_time_sec) < min_dt) return false;
    }
    camera.last_debug_save_time_sec = stamp_sec;
    const uint64_t seq = camera.debug_image_sequence++;
    std::ostringstream oss;
    oss << debug_output_dir_ << "/"
        << camera.debug_topic_tag << "_"
        << static_cast<long long>(msg->header.stamp.sec) << "_"
        << std::setw(9) << std::setfill('0') << msg->header.stamp.nanosec << "_"
        << std::setw(6) << std::setfill('0') << seq << "_"
        << (has_valid_detections ? "detected" : "processed") << ".png";
    output_path = oss.str();
    return true;
  }

  // ---- Pose estimation core ------------------------------------------------

  // Object points in the marker frame: x-right, y-up, z-out-of-marker.
  // Ordering matches aruco_nano's returned corner order.
  static std::vector<cv::Point3f> makeMarkerObjectPoints(double marker_length)
  {
    const float s = static_cast<float>(marker_length / 2.0);
    return {
      cv::Point3f(-s,  s, 0.0f),   // 0: top-left
      cv::Point3f( s,  s, 0.0f),   // 1: top-right
      cv::Point3f( s, -s, 0.0f),   // 2: bottom-right
      cv::Point3f(-s, -s, 0.0f)};  // 3: bottom-left
  }

  static double computeReprojectionError(
    const std::vector<cv::Point3f> & object_points,
    const std::vector<cv::Point2f> & image_corners,
    const cv::Vec3d & rvec, const cv::Vec3d & tvec,
    const cv::Mat & K, const cv::Mat & D)
  {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec, K, D, projected);
    double sq = 0.0;
    for (size_t i = 0; i < image_corners.size(); ++i) {
      const cv::Point2f d = projected[i] - image_corners[i];
      sq += static_cast<double>(d.x * d.x + d.y * d.y);
    }
    return std::sqrt(sq / static_cast<double>(image_corners.size()));
  }

  // Estimate marker pose with IPPE_SQUARE (returns up to 2 solutions for
  // planar markers), then disambiguate. We need to disambiguate even though
  // we only output position, because the two solutions differ in translation
  // too - picking the wrong one gives the wrong position.
  //
  // Optional ITERATIVE refinement runs only from the chosen seed and is
  // rejected if it would flip the marker normal or drift far from the prior.
  //
  // prior_position_camera: if non-null, the predicted marker centre in the
  // current camera frame (from the KF transformed through current odom).
  // This is the most reliable disambiguator.
  // NOTE: deliberately NOT a const member function. RCLCPP_WARN_THROTTLE in
  // ROS 2 Humble expands to clock.now(), which is non-const on
  // rclcpp::Clock; calling it from a const method (which gets a const Clock
  // via *this->get_clock()) fails to compile. The function reads members
  // only, so dropping const is safe and surgical.
  bool estimateMarkerPoseRefined(
    const std::vector<cv::Point2f> & image_corners,
    const CameraState & camera,
    const cv::Vec3d * prior_position_camera,
    cv::Vec3d & rvec_out, cv::Vec3d & tvec_out,
    double & reprojection_error_px)
  {
    const auto object_points = makeMarkerObjectPoints(marker_length_);
    const cv::Mat & K = camera.calibration.camera_matrix;
    const cv::Mat & D = camera.calibration.dist_coeffs;

    std::vector<cv::Mat> rvecs_all, tvecs_all;
    std::vector<double> reproj_errs;
    int n_sol = 0;
    try {
      n_sol = cv::solvePnPGeneric(
        object_points, image_corners, K, D,
        rvecs_all, tvecs_all,
        false, cv::SOLVEPNP_IPPE_SQUARE,
        cv::noArray(), cv::noArray(), reproj_errs);
    } catch (const cv::Exception & e) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "solvePnPGeneric(IPPE_SQUARE) failed: %s", e.what());
      return false;
    }
    if (n_sol < 1) return false;

    struct Candidate { cv::Vec3d r, t; double err{0.0}; bool normal_ok{false}; double prior_dist{0.0}; };
    std::vector<Candidate> cands;
    cands.reserve(static_cast<size_t>(n_sol));
    for (int i = 0; i < n_sol; ++i) {
      Candidate c;
      c.r = cv::Vec3d(rvecs_all[i].at<double>(0), rvecs_all[i].at<double>(1), rvecs_all[i].at<double>(2));
      c.t = cv::Vec3d(tvecs_all[i].at<double>(0), tvecs_all[i].at<double>(1), tvecs_all[i].at<double>(2));
      if (c.t[2] <= 0.0) continue;
      c.err = (i < static_cast<int>(reproj_errs.size())) ? reproj_errs[i]
              : computeReprojectionError(object_points, image_corners, c.r, c.t, K, D);
      cv::Mat R; cv::Rodrigues(c.r, R);
      // Marker normal in camera frame is the third column of R. For the
      // marker's front to face the camera (camera looks along +z), the
      // normal's z-component should be negative.
      c.normal_ok = (R.at<double>(2, 2) < 0.0);
      if (prior_position_camera != nullptr) {
        c.prior_dist = cv::norm(c.t - *prior_position_camera);
      }
      cands.push_back(c);
    }
    if (cands.empty()) return false;

    int best = 0;
    if (prior_position_camera != nullptr) {
      double best_score = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < cands.size(); ++i) {
        if (cands[i].prior_dist < best_score) {
          best_score = cands[i].prior_dist; best = static_cast<int>(i);
        }
      }
    } else {
      bool found_normal_ok = false;
      for (const auto & c : cands) if (c.normal_ok) { found_normal_ok = true; break; }
      double best_score = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < cands.size(); ++i) {
        if (found_normal_ok && !cands[i].normal_ok) continue;
        if (cands[i].err < best_score) { best_score = cands[i].err; best = static_cast<int>(i); }
      }
    }

    rvec_out = cands[best].r;
    tvec_out = cands[best].t;

    if (use_iterative_pnp_) {
      cv::Vec3d r_ref = rvec_out, t_ref = tvec_out;
      bool ok = false;
      try {
        ok = cv::solvePnP(
          object_points, image_corners, K, D,
          r_ref, t_ref, true, cv::SOLVEPNP_ITERATIVE);
      } catch (const cv::Exception &) { ok = false; }
      if (ok && t_ref[2] > 0.0) {
        cv::Mat R; cv::Rodrigues(r_ref, R);
        const bool normal_ok = (R.at<double>(2, 2) < 0.0);
        if (prior_position_camera != nullptr) {
          const double moved = cv::norm(t_ref - *prior_position_camera);
          if (moved <= cands[best].prior_dist * 1.5 + 0.05) {
            rvec_out = r_ref; tvec_out = t_ref;
          }
        } else if (normal_ok) {
          rvec_out = r_ref; tvec_out = t_ref;
        }
      }
    }

    reprojection_error_px = computeReprojectionError(
      object_points, image_corners, rvec_out, tvec_out, K, D);
    return reprojection_error_px <= max_reprojection_error_px_;
  }

  // ---- KF helpers ----------------------------------------------------------

  void initializeTrack(
    MarkerTrack & track, int marker_id, const std::string & label,
    const cv::Vec3d & p_world_marker, double t_sec)
  {
    track.initialized = true;
    track.marker_id = marker_id;
    track.label = label;
    track.last_prediction_t_sec = t_sec;
    track.last_measurement_t_sec = t_sec;

    track.kf.statePost = cv::Mat::zeros(3, 1, CV_64F);
    track.kf.statePost.at<double>(0, 0) = p_world_marker[0];
    track.kf.statePost.at<double>(1, 0) = p_world_marker[1];
    track.kf.statePost.at<double>(2, 0) = p_world_marker[2];
    track.kf.statePre = track.kf.statePost.clone();

    cv::setIdentity(track.kf.errorCovPost, cv::Scalar(1.0));
    cv::setIdentity(track.kf.measurementNoiseCov, cv::Scalar(0.01));
    cv::setIdentity(track.kf.processNoiseCov, cv::Scalar(1e-4));
  }

  void predictTrackTo(MarkerTrack & track, double target_t_sec, double speed_norm)
  {
    if (!track.initialized) return;
    const double dt = target_t_sec - track.last_prediction_t_sec;
    if (dt <= 1e-6) return;  // also catches dt < 0 (out-of-order updates)

    track.kf.transitionMatrix = cv::Mat::eye(3, 3, CV_64F);
    const double sigma_pos_rate =
      kf_process_pos_rate_sigma_ * (1.0 + kf_speed_noise_scale_ * speed_norm);
    const double q = sigma_pos_rate * sigma_pos_rate * dt;
    track.kf.processNoiseCov = cv::Mat::eye(3, 3, CV_64F) * q;
    track.kf.predict();
    track.last_prediction_t_sec = target_t_sec;
  }

  // Build a 3x3 measurement covariance for the marker centre in the world
  // frame. We model per-corner pixel noise as the reprojection RMS and
  // propagate it to (lateral, depth) sigmas in the camera frame:
  //   sigma_lat   ~ (depth * sigma_px) / focal
  //   sigma_depth ~ (depth^2 * sigma_px) / (focal * marker_size_eff)
  // Then rotate this diagonal camera-frame covariance into the world frame.
  cv::Mat buildMeasurementCovWorld(
    const CameraState & camera, const cv::Matx44d & T_world_camera,
    double depth_m, double reproj_rms_px) const
  {
    const double fx = camera.calibration.camera_matrix.at<double>(0, 0);
    const double fy = camera.calibration.camera_matrix.at<double>(1, 1);
    const double focal_avg = std::max(1.0, 0.5 * (fx + fy));
    const double sigma_px = std::max(0.25, reproj_rms_px);
    const double depth = std::max(0.05, depth_m);
    const double marker_eff = std::max(0.05, marker_length_);

    double sigma_lat = depth * sigma_px / focal_avg;
    double sigma_dep = depth * depth * sigma_px / (focal_avg * marker_eff);

    sigma_lat = std::max(kf_measurement_noise_floor_m_, sigma_lat);
    sigma_dep = std::max(sigma_lat * kf_depth_to_lateral_ratio_min_, sigma_dep);

    cv::Matx33d Sigma_cam(
      sigma_lat * sigma_lat, 0.0, 0.0,
      0.0, sigma_lat * sigma_lat, 0.0,
      0.0, 0.0, sigma_dep * sigma_dep);

    // R = camera->world rotation (top-left 3x3 of T_world_camera).
    cv::Matx33d R(
      T_world_camera(0, 0), T_world_camera(0, 1), T_world_camera(0, 2),
      T_world_camera(1, 0), T_world_camera(1, 1), T_world_camera(1, 2),
      T_world_camera(2, 0), T_world_camera(2, 1), T_world_camera(2, 2));
    cv::Matx33d Sigma_w = R * Sigma_cam * R.t();

    cv::Mat cov(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) cov.at<double>(r, c) = Sigma_w(r, c);
    }
    return cov;
  }

  void correctTrack(
    MarkerTrack & track, const cv::Vec3d & p_world_marker_meas,
    const cv::Mat & measurement_cov_world, double t_sec)
  {
    if (!track.initialized) {
      initializeTrack(track, track.marker_id, track.label, p_world_marker_meas, t_sec);
      return;
    }
    measurement_cov_world.copyTo(track.kf.measurementNoiseCov);

    cv::Mat measurement(3, 1, CV_64F);
    measurement.at<double>(0, 0) = p_world_marker_meas[0];
    measurement.at<double>(1, 0) = p_world_marker_meas[1];
    measurement.at<double>(2, 0) = p_world_marker_meas[2];
    track.kf.correct(measurement);

    track.last_measurement_t_sec = t_sec;
    track.last_prediction_t_sec = t_sec;
  }

  // ---- Publishing & lifecycle ----------------------------------------------

  // Build a PoseStamped with the given position in the drone frame.
  // Orientation is set to identity (the contract is: only position is
  // meaningful; downstream consumers must ignore orientation).
  geometry_msgs::msg::PoseStamped positionToPoseStamped(
    const cv::Vec3d & p_drone, const builtin_interfaces::msg::Time & stamp) const
  {
    geometry_msgs::msg::PoseStamped p;
    p.header.stamp = stamp;
    p.header.frame_id = drone_frame_id_;
    p.pose.position.x = p_drone[0];
    p.pose.position.y = p_drone[1];
    p.pose.position.z = p_drone[2];
    p.pose.orientation.x = 0.0;
    p.pose.orientation.y = 0.0;
    p.pose.orientation.z = 0.0;
    p.pose.orientation.w = 1.0;
    return p;
  }

  static bool isFilteredMeasurementKind(const std::string & kind)
  {
    return kind == "filtered_measurement";
  }

  void registerMeasurementCandidate(
    int marker_id, const std::string & label,
    const geometry_msgs::msg::PoseStamped & pose_msg,
    const std::string & source_topic, const std::string & output_kind,
    double reprojection_error_px)
  {
    auto it = cycle_measurement_candidates_.find(marker_id);
    if (it == cycle_measurement_candidates_.end()) {
      CycleMeasurementCandidate c;
      c.valid = true; c.marker_id = marker_id; c.label = label;
      c.pose_msg = pose_msg; c.source_topic = source_topic;
      c.output_kind = output_kind; c.reprojection_error_px = reprojection_error_px;
      cycle_measurement_candidates_[marker_id] = c;
      return;
    }
    const bool new_is_filtered = isFilteredMeasurementKind(output_kind);
    const bool old_is_filtered = isFilteredMeasurementKind(it->second.output_kind);
    bool replace = false;
    if (new_is_filtered && !old_is_filtered) replace = true;
    else if (new_is_filtered == old_is_filtered &&
             reprojection_error_px < it->second.reprojection_error_px) replace = true;
    if (replace) {
      it->second.valid = true; it->second.marker_id = marker_id; it->second.label = label;
      it->second.pose_msg = pose_msg; it->second.source_topic = source_topic;
      it->second.output_kind = output_kind;
      it->second.reprojection_error_px = reprojection_error_px;
    }
  }

  void publishPoseAndInfo(
    int marker_id, const std::string & label,
    const geometry_msgs::msg::PoseStamped & pose_msg,
    const std::string & source_topic, const std::string & output_kind)
  {
    pose_pub_->publish(pose_msg);
    if (type_pub_->get_subscription_count() > 0U) {
      std_msgs::msg::String m; m.data = label; type_pub_->publish(m);
    }
    if (msg_pub_->get_subscription_count() > 0U) {
      std_msgs::msg::String info;
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3)
          << "Marker ID: " << marker_id
          << ", Type: " << label
          << ", Source: " << source_topic
          << ", Detector: aruco_nano"
          << ", Output: " << output_kind
          << ", Frame: " << drone_frame_id_
          << ", Position: ("
          << pose_msg.pose.position.x << ", "
          << pose_msg.pose.position.y << ", "
          << pose_msg.pose.position.z << ")";
      info.data = oss.str();
      msg_pub_->publish(info);
    }
  }

  void publishCycleMeasurementCandidates()
  {
    for (const auto & [marker_id, candidate] : cycle_measurement_candidates_) {
      if (!candidate.valid) continue;
      publishPoseAndInfo(marker_id, candidate.label, candidate.pose_msg,
                         candidate.source_topic, candidate.output_kind);
    }
  }

  void publishPredictedTracks()
  {
    if (!use_track_filter_) return;
    InterpolatedOdom odom;
    {
      std::lock_guard<std::mutex> lock(odom_buffer_mutex_);
      if (odom_buffer_.empty()) return;
      const auto & back = odom_buffer_.back();
      odom.valid = true;
      odom.t_sec = back.t_sec;
      odom.p_world_body = back.p_world_body;
      odom.q_world_body = back.q_world_body;
      odom.v_body = back.v_body;
    }

    const cv::Matx44d T_world_body = matrixFromPose(odom.p_world_body, odom.q_world_body);
    const cv::Matx44d T_body_world = invertTransform(T_world_body);
    const double speed_norm = computeSpeedNorm(odom.v_body);

    for (auto & [marker_id, track] : tracks_) {
      if (!track.initialized) continue;
      if (cycle_measurement_candidates_.find(marker_id) != cycle_measurement_candidates_.end()) continue;
      if (std::abs(odom.t_sec - track.last_measurement_t_sec) > max_track_age_sec_) continue;

      predictTrackTo(track, odom.t_sec, speed_norm);
      const cv::Vec3d p_world_marker(
        track.kf.statePre.at<double>(0, 0),
        track.kf.statePre.at<double>(1, 0),
        track.kf.statePre.at<double>(2, 0));

      const cv::Vec3d p_body_marker = applyTransform(T_body_world, p_world_marker);
      auto pose_msg = positionToPoseStamped(p_body_marker, secToTimeMsg(odom.t_sec));
      publishPoseAndInfo(marker_id, track.label, pose_msg, "track_prediction", "track_prediction");
    }
  }

  void pruneStaleTracks()
  {
    double latest_t = 0.0;
    {
      std::lock_guard<std::mutex> lock(odom_buffer_mutex_);
      if (odom_buffer_.empty()) return;
      latest_t = odom_buffer_.back().t_sec;
    }
    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
      const auto & track = it->second;
      if (!track.initialized ||
          std::abs(latest_t - track.last_measurement_t_sec) > max_track_age_sec_) {
        it = tracks_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // ---- Main per-image processing -------------------------------------------

  void processImage(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const std::string & topic_name, CameraState & camera)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "cv_bridge failed for %s: %s", topic_name.c_str(), e.what());
      return;
    }

    cv::Mat debug_bgr, gray;
    const bool need_debug_image = save_debug_images_;
    try {
      const auto & enc = msg->encoding;
      if (enc == sensor_msgs::image_encodings::MONO8) {
        gray = cv_ptr->image;
        if (need_debug_image) cv::cvtColor(cv_ptr->image, debug_bgr, cv::COLOR_GRAY2BGR);
      } else if (enc == sensor_msgs::image_encodings::BGR8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
        if (need_debug_image) debug_bgr = cv_ptr->image.clone();
      } else if (enc == sensor_msgs::image_encodings::RGB8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_RGB2GRAY);
        if (need_debug_image) cv::cvtColor(cv_ptr->image, debug_bgr, cv::COLOR_RGB2BGR);
      } else if (enc == sensor_msgs::image_encodings::BGRA8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGRA2GRAY);
        if (need_debug_image) cv::cvtColor(cv_ptr->image, debug_bgr, cv::COLOR_BGRA2BGR);
      } else if (enc == sensor_msgs::image_encodings::RGBA8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_RGBA2GRAY);
        if (need_debug_image) cv::cvtColor(cv_ptr->image, debug_bgr, cv::COLOR_RGBA2BGR);
      } else {
        auto bgr_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::cvtColor(bgr_ptr->image, gray, cv::COLOR_BGR2GRAY);
        if (need_debug_image) debug_bgr = bgr_ptr->image.clone();
      }
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Image conversion failed for %s: %s", topic_name.c_str(), e.what());
      return;
    }

    DetectionSelection sel;
    try {
      selectBestDetection(gray, sel);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Aruco Nano detection failed for %s: %s", topic_name.c_str(), e.what());
      return;
    }

    std::vector<std::vector<cv::Point2f>> corners = sel.corners;
    std::vector<int> ids = sel.ids;

    if (ids.empty()) {
      std::string debug_output_path;
      if (shouldSaveDebugImage(msg, camera, false, debug_output_path)) {
        saveDebugImage(debug_bgr, debug_output_path, topic_name, sel.candidate_name,
                       {}, {}, {}, {}, {}, camera);
      }
      return;
    }
    // No second cornerSubPix here: aruco_nano refines internally on raw gray.

    const double img_t = timeMsgToSec(msg->header.stamp);
    InterpolatedOdom odom = interpolateOdomAt(img_t);

    cv::Matx44d T_world_body = cv::Matx44d::eye();
    cv::Matx44d T_body_world = cv::Matx44d::eye();
    cv::Matx44d T_world_camera = cv::Matx44d::eye();
    cv::Matx44d T_camera_world = cv::Matx44d::eye();
    double speed_norm = 0.0;
    if (odom.valid) {
      T_world_body = matrixFromPose(odom.p_world_body, odom.q_world_body);
      T_body_world = invertTransform(T_world_body);
      T_world_camera = T_world_body * camera.T_drone_camera;
      T_camera_world = invertTransform(T_world_camera);
      speed_norm = computeSpeedNorm(odom.v_body);
    }

    std::vector<std::vector<cv::Point2f>> valid_corners;
    std::vector<int> valid_ids;
    std::vector<cv::Vec3d> rvecs, tvecs;
    std::vector<bool> filtered_flags;
    valid_corners.reserve(ids.size());
    valid_ids.reserve(ids.size());
    rvecs.reserve(ids.size());
    tvecs.reserve(ids.size());
    filtered_flags.reserve(ids.size());

    for (size_t i = 0; i < ids.size(); ++i) {
      const auto label_it = marker_labels_.find(ids[i]);
      if (label_it == marker_labels_.end()) continue;

      cv::Vec3d prior_cam(0.0, 0.0, 0.0);
      bool have_prior = false;
      auto track_it = tracks_.find(ids[i]);
      if (use_track_filter_ && odom.valid &&
          track_it != tracks_.end() && track_it->second.initialized) {
        const cv::Vec3d p_world(
          track_it->second.kf.statePost.at<double>(0, 0),
          track_it->second.kf.statePost.at<double>(1, 0),
          track_it->second.kf.statePost.at<double>(2, 0));
        prior_cam = applyTransform(T_camera_world, p_world);
        have_prior = (prior_cam[2] > 0.0);  // only useful if in front of camera
      }

      cv::Vec3d rvec, tvec;
      double reprojection_error_px = 0.0;
      if (!estimateMarkerPoseRefined(
            corners[i], camera, have_prior ? &prior_cam : nullptr,
            rvec, tvec, reprojection_error_px)) {
        continue;
      }

      // Marker position in the drone frame for publishing.
      // p_drone = T_drone_camera * (R_cm * 0 + tvec) = T_drone_camera * tvec.
      // We could equivalently compose T_drone_marker_raw and pick the
      // translation, but applying the transform directly to tvec is clearer
      // and avoids constructing a full marker-frame transform we never need.
      const cv::Vec3d p_drone_marker_raw = applyTransform(camera.T_drone_camera, tvec);
      auto raw_pose_msg = positionToPoseStamped(p_drone_marker_raw, msg->header.stamp);
      if (publish_raw_pose_) raw_pose_pub_->publish(raw_pose_msg);

      geometry_msgs::msg::PoseStamped out_pose_msg = raw_pose_msg;
      std::string output_kind = "raw_measurement";
      bool used_filtered_measurement = false;

      if (use_track_filter_ && odom.valid) {
        // Marker position in world coords: lift drone-frame position into
        // world using the (interpolated) body pose at the image timestamp.
        const cv::Vec3d p_world_marker_meas = applyTransform(T_world_body, p_drone_marker_raw);

        auto & track = tracks_[ids[i]];
        if (!track.initialized) {
          track.marker_id = ids[i];
          track.label = label_it->second;
          initializeTrack(track, ids[i], label_it->second, p_world_marker_meas, img_t);
        } else {
          predictTrackTo(track, img_t, speed_norm);
          const cv::Mat cov_world = buildMeasurementCovWorld(
            camera, T_world_camera, tvec[2], reprojection_error_px);
          correctTrack(track, p_world_marker_meas, cov_world, img_t);
        }

        const cv::Vec3d p_world_marker_filt(
          track.kf.statePost.at<double>(0, 0),
          track.kf.statePost.at<double>(1, 0),
          track.kf.statePost.at<double>(2, 0));
        const cv::Vec3d p_drone_marker_filt = applyTransform(T_body_world, p_world_marker_filt);

        out_pose_msg = positionToPoseStamped(p_drone_marker_filt, msg->header.stamp);
        output_kind = "filtered_measurement";
        used_filtered_measurement = true;
      }

      registerMeasurementCandidate(ids[i], label_it->second, out_pose_msg,
                                   topic_name, output_kind, reprojection_error_px);

      valid_corners.push_back(corners[i]);
      valid_ids.push_back(ids[i]);
      rvecs.push_back(rvec);
      tvecs.push_back(tvec);
      filtered_flags.push_back(used_filtered_measurement);
    }

    std::string debug_output_path;
    if (shouldSaveDebugImage(msg, camera, !valid_ids.empty(), debug_output_path)) {
      saveDebugImage(debug_bgr, debug_output_path, topic_name, sel.candidate_name,
                     valid_corners, valid_ids, rvecs, tvecs, filtered_flags, camera);
    }
  }

  // ---- Debug image saving --------------------------------------------------

  void saveDebugImage(
    const cv::Mat & debug_bgr, const std::string & output_path,
    const std::string & topic_name, const std::string & preprocessing_variant_name,
    const std::vector<std::vector<cv::Point2f>> & valid_corners,
    const std::vector<int> & valid_ids,
    const std::vector<cv::Vec3d> & rvecs, const std::vector<cv::Vec3d> & tvecs,
    const std::vector<bool> & filtered_flags, const CameraState & camera)
  {
    if (debug_bgr.empty()) return;
    cv::Mat annotated = debug_bgr.clone();
    if (!valid_ids.empty()) {
      cv::aruco::drawDetectedMarkers(annotated, valid_corners, valid_ids);
      for (size_t i = 0; i < valid_ids.size() && i < rvecs.size() && i < tvecs.size(); ++i) {
        // Draw frame axes for visual debugging only (orientation is NOT
        // published; this is just for sanity-checking pose recovery).
        cv::drawFrameAxes(
          annotated, camera.calibration.camera_matrix, camera.calibration.dist_coeffs,
          rvecs[i], tvecs[i], static_cast<float>(0.5 * marker_length_));

        const cv::Vec3d p_drone = applyTransform(camera.T_drone_camera, tvecs[i]);
        std::ostringstream line;
        line << "ID " << valid_ids[i] << " D=("
             << std::fixed << std::setprecision(2)
             << p_drone[0] << ", " << p_drone[1] << ", " << p_drone[2] << ")"
             << (filtered_flags[i] ? " FILT" : " RAW");
        const auto & corner = valid_corners[i][0];
        cv::putText(
          annotated, line.str(),
          cv::Point(static_cast<int>(corner.x),
                    std::max(20, static_cast<int>(corner.y) - 10)),
          cv::FONT_HERSHEY_SIMPLEX, 0.5,
          filtered_flags[i] ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 255, 255),
          1, cv::LINE_AA);
      }
    }
    std::ostringstream header;
    header << "Aruco Nano | Topic: " << topic_name
           << " | Valid markers: " << valid_ids.size()
           << " | Frame: " << drone_frame_id_
           << " | Prep: " << preprocessing_variant_name;
    cv::putText(annotated, header.str(), cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    try {
      const bool ok = cv::imwrite(output_path, annotated);
      if (!ok) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
          "Failed to write debug image: %s", output_path.c_str());
      }
    } catch (const cv::Exception & e) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "OpenCV failed to write debug image %s: %s", output_path.c_str(), e.what());
    }
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoNanoPoseEstimator>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
