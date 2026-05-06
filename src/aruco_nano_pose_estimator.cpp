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

    subpix_window_size_ = this->declare_parameter<int>("subpix_window_size", 5);
    subpix_zero_zone_ = this->declare_parameter<int>("subpix_zero_zone", -1);
    subpix_max_iterations_ = this->declare_parameter<int>("subpix_max_iterations", 30);
    subpix_epsilon_ = this->declare_parameter<double>("subpix_epsilon", 1e-3);

    use_iterative_pnp_ = this->declare_parameter<bool>("use_iterative_pnp", true);
    max_reprojection_error_px_ = this->declare_parameter<double>("max_reprojection_error_px", 4.0);

    // Static-landmark world-position random-walk sigma rate [m/sqrt(s)].
    kf_process_accel_sigma_ = this->declare_parameter<double>("kf_process_accel_sigma", 0.05);
    kf_measurement_noise_floor_m_ = this->declare_parameter<double>("kf_measurement_noise_floor_m", 0.03);
    kf_speed_noise_scale_ = this->declare_parameter<double>("kf_speed_noise_scale", 0.3);

    // Detection preprocessing
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

    if (marker_length_ <= 0.0) {
      throw std::runtime_error("marker_length must be > 0");
    }
    if (processing_period_ms_ <= 0) {
      throw std::runtime_error("processing_period_ms must be > 0");
    }
    if (debug_max_save_rate_hz_ < 0.0) {
      throw std::runtime_error("debug_max_save_rate_hz must be >= 0");
    }
    if (max_odom_time_diff_sec_ < 0.0 || max_track_age_sec_ < 0.0) {
      throw std::runtime_error("max_odom_time_diff_sec and max_track_age_sec must be >= 0");
    }
    if (subpix_window_size_ <= 0) {
      throw std::runtime_error("subpix_window_size must be > 0");
    }
    if (subpix_max_iterations_ <= 0 || subpix_epsilon_ <= 0.0) {
      throw std::runtime_error("subpix_max_iterations and subpix_epsilon must be > 0");
    }
    if (max_reprojection_error_px_ <= 0.0) {
      throw std::runtime_error("max_reprojection_error_px must be > 0");
    }
    if (kf_process_accel_sigma_ <= 0.0 || kf_measurement_noise_floor_m_ <= 0.0) {
      throw std::runtime_error("kf_process_accel_sigma and kf_measurement_noise_floor_m must be > 0");
    }

    if (preprocess_gamma_ <= 0.0) {
      throw std::runtime_error("preprocess_gamma must be > 0");
    }
    if (preprocess_clahe_clip_limit_ <= 0.0) {
      throw std::runtime_error("preprocess_clahe_clip_limit must be > 0");
    }
    if (preprocess_clahe_tile_grid_size_ <= 0) {
      throw std::runtime_error("preprocess_clahe_tile_grid_size must be > 0");
    }
    if (preprocess_median_blur_ksize_ <= 0 || (preprocess_median_blur_ksize_ % 2) == 0) {
      throw std::runtime_error("preprocess_median_blur_ksize must be a positive odd integer");
    }
    if (preprocess_adaptive_block_size_ <= 1 || (preprocess_adaptive_block_size_ % 2) == 0) {
      throw std::runtime_error("preprocess_adaptive_block_size must be an odd integer > 1");
    }

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

    dictionary_ = cv::aruco::getPredefinedDictionary(dictionaryNameToEnum(dictionary_name_));
    detector_ = std::make_unique<aruco_nano::ArucoDetector>(dictionary_);

    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/aruco/pose", rclcpp::QoS(10));
    raw_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/aruco/raw_pose", rclcpp::QoS(10));
    type_pub_ = this->create_publisher<std_msgs::msg::String>("/aruco/landing_place_type", rclcpp::QoS(10));
    msg_pub_  = this->create_publisher<std_msgs::msg::String>("/aruco/message", rclcpp::QoS(10));

    slam_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      slam_odom_topic_,
      rclcpp::QoS(20),
      std::bind(&ArucoNanoPoseEstimator::slamOdomCallback, this, std::placeholders::_1));

    addCamera(
      "/camera/front/image_raw",
      cv::Matx44d(
         0.0,  -0.9659258,  0.2588120,  0.0,
         1.0,   0.0,        0.0,        0.0,
         0.0,   0.2588120,  0.9659258,  0.0,
         0.0,   0.0,        0.0,        1.0
      ),
      (cv::Mat_<double>(3, 3) <<
        1103.60401999, 0.0, 946.547127394,
        0.0, 1100.45084936, 598.302984461,
        0.0, 0.0, 1.0),
      (cv::Mat_<double>(1, 5) <<
        -0.00344874724739, -0.0394044158268, -0.00372898986636, 0.00277661293568, 0.0172371744751)
    );

    addCamera(
      "/camera/down/image_raw",
      cv::Matx44d(
         0.0, -1.0, 0.0, 0.0,
         1.0,  0.0, 0.0, 0.0,
         0.0,  0.0, 1.0, 0.0,
         0.0,  0.0, 0.0, 1.0
      ),
      (cv::Mat_<double>(3, 3) <<
        1103.60401999, 0.0, 946.547127394,
        0.0, 1100.45084936, 598.302984461,
        0.0, 0.0, 1.0),
      (cv::Mat_<double>(1, 5) <<
        -0.00344874724739, -0.0394044158268, -0.00372898986636, 0.00277661293568, 0.0172371744751)
    );

    const auto image_qos = rclcpp::SensorDataQoS().keep_last(1);

    for (auto & [topic, camera] : cameras_) {
      camera->sub = this->create_subscription<sensor_msgs::msg::Image>(
        topic,
        image_qos,
        [this, topic](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          this->storeLatestImage(msg, topic);
        }
      );
    }

    processing_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(processing_period_ms_),
      [this]() { this->onTimer(); });

    RCLCPP_INFO(
      this->get_logger(),
      "ArucoNanoPoseEstimator started. marker_length=%.6f, dictionary=%s, frame_id=%s, "
      "period=%d ms, preprocess_enable=%s, save_debug_images=%s, slam_odom_topic=%s, "
      "use_track_filter=%s",
      marker_length_,
      dictionary_name_.c_str(),
      drone_frame_id_.c_str(),
      processing_period_ms_,
      preprocess_enable_ ? "true" : "false",
      save_debug_images_ ? "true" : "false",
      slam_odom_topic_.c_str(),
      use_track_filter_ ? "true" : "false");
  }

private:
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

  struct SlamOdomState
  {
    bool valid{false};
    builtin_interfaces::msg::Time stamp;
    std::string frame_id;
    std::string child_frame_id;
    cv::Vec3d p_world_body{0.0, 0.0, 0.0};
    std::array<double, 4> q_world_body{0.0, 0.0, 0.0, 1.0};
    cv::Vec3d v_body{0.0, 0.0, 0.0};
  };

  struct MarkerTrack
  {
    bool initialized{false};
    int marker_id{-1};
    std::string label;
    cv::KalmanFilter kf;
    builtin_interfaces::msg::Time last_prediction_stamp;
    builtin_interfaces::msg::Time last_measurement_stamp;
    std::array<double, 4> q_world_marker{0.0, 0.0, 0.0, 1.0};
    bool have_orientation{false};

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
    cv::Mat refine_image;
  };

  struct DetectionSelection
  {
    bool valid{false};
    std::string candidate_name{"none"};
    cv::Mat refine_image;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    int relevant_count{0};
    int total_count{0};
    double area_score{0.0};
  };

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

  mutable std::mutex slam_odom_mutex_;
  SlamOdomState latest_slam_odom_;

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

  int subpix_window_size_{5};
  int subpix_zero_zone_{-1};
  int subpix_max_iterations_{30};
  double subpix_epsilon_{1e-3};

  bool use_iterative_pnp_{true};
  double max_reprojection_error_px_{4.0};

  double kf_process_accel_sigma_{0.05};
  double kf_measurement_noise_floor_m_{0.03};
  double kf_speed_noise_scale_{0.3};

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
      {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL}
    };

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
      if (std::isalnum(static_cast<unsigned char>(ch))) {
        sanitized.push_back(ch);
      } else {
        sanitized.push_back('_');
      }
    }

    while (!sanitized.empty() && sanitized.front() == '_') {
      sanitized.erase(sanitized.begin());
    }
    if (sanitized.empty()) {
      sanitized = "camera";
    }
    return sanitized;
  }

  static double timeMsgToSec(const builtin_interfaces::msg::Time & t)
  {
    return static_cast<double>(t.sec) + 1e-9 * static_cast<double>(t.nanosec);
  }

  static double timeDiffSec(
    const builtin_interfaces::msg::Time & a,
    const builtin_interfaces::msg::Time & b)
  {
    return timeMsgToSec(a) - timeMsgToSec(b);
  }

  static std::array<double, 4> normalizeQuaternion(const std::array<double, 4> & q)
  {
    const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (norm <= 1e-12) {
      return {0.0, 0.0, 0.0, 1.0};
    }
    return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
  }

  static cv::Mat makeGammaLut(double gamma)
  {
    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
      const double normalized = static_cast<double>(i) / 255.0;
      const double corrected = std::pow(normalized, gamma);
      lut.at<unsigned char>(0, i) = static_cast<unsigned char>(
        std::round(std::clamp(corrected, 0.0, 1.0) * 255.0));
    }
    return lut;
  }

  static cv::Matx33d quaternionToRotationMatrix(const std::array<double, 4> & q_in)
  {
    const auto q = normalizeQuaternion(q_in);
    const double x = q[0];
    const double y = q[1];
    const double z = q[2];
    const double w = q[3];

    return cv::Matx33d(
      1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
      2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
      2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)
    );
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
    if (corners.size() < 4U) {
      return 0.0;
    }
    return std::fabs(cv::contourArea(corners));
  }

  void addCamera(
    const std::string & topic,
    const cv::Matx44d & T_drone_camera,
    const cv::Mat & camera_matrix,
    const cv::Mat & dist_coeffs)
  {
    auto camera = std::make_shared<CameraState>();
    camera->T_drone_camera = T_drone_camera;
    camera->calibration.camera_matrix = camera_matrix.clone();
    camera->calibration.dist_coeffs = dist_coeffs.clone();
    camera->debug_topic_tag = sanitizeTopicName(topic);
    cameras_[topic] = camera;
  }

  void slamOdomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    SlamOdomState state;
    state.valid = true;
    state.stamp = msg->header.stamp;
    state.frame_id = msg->header.frame_id;
    state.child_frame_id = msg->child_frame_id;
    state.p_world_body = cv::Vec3d(
      msg->pose.pose.position.x,
      msg->pose.pose.position.y,
      msg->pose.pose.position.z);
    state.q_world_body = normalizeQuaternion({
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z,
      msg->pose.pose.orientation.w
    });
    state.v_body = cv::Vec3d(
      msg->twist.twist.linear.x,
      msg->twist.twist.linear.y,
      msg->twist.twist.linear.z);

    std::lock_guard<std::mutex> lock(slam_odom_mutex_);
    latest_slam_odom_ = state;
  }

  SlamOdomState getLatestSlamOdom() const
  {
    std::lock_guard<std::mutex> lock(slam_odom_mutex_);
    return latest_slam_odom_;
  }

  void storeLatestImage(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const std::string & topic_name)
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
        if (!camera->latest_msg) {
          continue;
        }
        msg = camera->latest_msg;
        camera->latest_msg.reset();
      }

      processImage(msg, topic_name, *camera);
    }
  }

  void buildDetectionCandidates(
    const cv::Mat & gray,
    std::vector<DetectionCandidateImage> & candidates) const
  {
    candidates.clear();
    candidates.reserve(5);

    if (!preprocess_enable_) {
      candidates.push_back({"gray_original", gray, gray});
      return;
    }

    if (preprocess_try_original_first_) {
      candidates.push_back({"gray_original", gray, gray});
    }

    cv::Mat enhanced = gray;
    bool have_nontrivial_enhancement = false;

    if (preprocess_use_gamma_) {
      cv::LUT(enhanced, gamma_lut_, enhanced);
      have_nontrivial_enhancement = true;
    }

    if (preprocess_use_clahe_ && clahe_) {
      cv::Mat clahe_out;
      clahe_->apply(enhanced, clahe_out);
      enhanced = clahe_out;
      have_nontrivial_enhancement = true;
    }

    if (preprocess_use_median_blur_) {
      cv::Mat median_out;
      cv::medianBlur(enhanced, median_out, preprocess_median_blur_ksize_);
      enhanced = median_out;
      have_nontrivial_enhancement = true;
    }

    if (have_nontrivial_enhancement || !preprocess_try_original_first_) {
      candidates.push_back({"gray_enhanced", enhanced, enhanced});
    }

    if (preprocess_use_adaptive_threshold_) {
      cv::Mat adaptive;
      cv::adaptiveThreshold(
        enhanced,
        adaptive,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY,
        preprocess_adaptive_block_size_,
        preprocess_adaptive_c_);
      candidates.push_back({"adaptive_threshold", adaptive, enhanced});
    }

    if (preprocess_use_otsu_threshold_) {
      cv::Mat otsu;
      cv::threshold(
        enhanced,
        otsu,
        0.0,
        255.0,
        cv::THRESH_BINARY | cv::THRESH_OTSU);
      candidates.push_back({"otsu_threshold", otsu, enhanced});
    }

    if (candidates.empty()) {
      candidates.push_back({"gray_original", gray, gray});
    }
  }

  void selectBestDetection(
    const cv::Mat & gray,
    DetectionSelection & best_selection)
  {
    best_selection = DetectionSelection{};
    best_selection.refine_image = gray;

    std::vector<DetectionCandidateImage> candidates;
    buildDetectionCandidates(gray, candidates);

    for (const auto & candidate : candidates) {
      std::vector<std::vector<cv::Point2f>> corners;
      std::vector<int> ids;

      detector_->detectMarkers(candidate.detect_image, corners, ids);

      int relevant_count = 0;
      double area_score = 0.0;

      for (size_t i = 0; i < ids.size(); ++i) {
        if (marker_labels_.find(ids[i]) != marker_labels_.end()) {
          ++relevant_count;
        }
        area_score += quadrilateralArea(corners[i]);
      }

      const int total_count = static_cast<int>(ids.size());

      bool replace = false;
      if (!best_selection.valid && total_count > 0) {
        replace = true;
      } else if (relevant_count > best_selection.relevant_count) {
        replace = true;
      } else if (
        relevant_count == best_selection.relevant_count &&
        total_count > best_selection.total_count)
      {
        replace = true;
      } else if (
        relevant_count == best_selection.relevant_count &&
        total_count == best_selection.total_count &&
        area_score > best_selection.area_score)
      {
        replace = true;
      }

      if (replace) {
        best_selection.valid = true;
        best_selection.candidate_name = candidate.name;
        best_selection.refine_image = candidate.refine_image;
        best_selection.corners = std::move(corners);
        best_selection.ids = std::move(ids);
        best_selection.relevant_count = relevant_count;
        best_selection.total_count = total_count;
        best_selection.area_score = area_score;
      }
    }

    if (!best_selection.valid && !candidates.empty()) {
      best_selection.candidate_name = candidates.front().name;
      best_selection.refine_image = candidates.front().refine_image;
    }
  }

  bool shouldSaveDebugImage(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    CameraState & camera,
    bool has_valid_detections,
    std::string & output_path)
  {
    output_path.clear();

    if (!save_debug_images_) {
      return false;
    }
    if (debug_save_only_with_valid_detections_ && !has_valid_detections) {
      return false;
    }

    const double stamp_sec =
      static_cast<double>(msg->header.stamp.sec) +
      1e-9 * static_cast<double>(msg->header.stamp.nanosec);

    std::lock_guard<std::mutex> lock(camera.debug_state_mutex);

    if (debug_max_save_rate_hz_ > 0.0 && camera.last_debug_save_time_sec >= 0.0) {
      const double min_dt = 1.0 / debug_max_save_rate_hz_;
      if ((stamp_sec - camera.last_debug_save_time_sec) < min_dt) {
        return false;
      }
    }

    camera.last_debug_save_time_sec = stamp_sec;
    const uint64_t seq = camera.debug_image_sequence++;

    std::ostringstream oss;
    oss << debug_output_dir_ << "/"
        << camera.debug_topic_tag << "_"
        << static_cast<long long>(msg->header.stamp.sec) << "_"
        << std::setw(9) << std::setfill('0') << msg->header.stamp.nanosec << "_"
        << std::setw(6) << std::setfill('0') << seq << "_"
        << (has_valid_detections ? "detected" : "processed")
        << ".png";

    output_path = oss.str();
    return true;
  }

  static cv::Matx44d matrixFromPose(
    const cv::Vec3d & p,
    const std::array<double, 4> & q)
  {
    cv::Matx44d T = cv::Matx44d::eye();
    const cv::Matx33d R = quaternionToRotationMatrix(q);

    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        T(r, c) = R(r, c);
      }
    }

    T(0, 3) = p[0];
    T(1, 3) = p[1];
    T(2, 3) = p[2];
    return T;
  }

  static cv::Matx44d invertTransform(const cv::Matx44d & T)
  {
    cv::Matx33d R(
      T(0, 0), T(0, 1), T(0, 2),
      T(1, 0), T(1, 1), T(1, 2),
      T(2, 0), T(2, 1), T(2, 2)
    );
    const cv::Matx33d Rt = R.t();
    const cv::Vec3d t(T(0, 3), T(1, 3), T(2, 3));
    const cv::Vec3d tinv = -rotateVec(Rt, t);

    cv::Matx44d Tinv = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        Tinv(r, c) = Rt(r, c);
      }
    }
    Tinv(0, 3) = tinv[0];
    Tinv(1, 3) = tinv[1];
    Tinv(2, 3) = tinv[2];
    return Tinv;
  }

  static double computeSpeedNorm(const SlamOdomState & odom)
  {
    return std::sqrt(
      odom.v_body[0] * odom.v_body[0] +
      odom.v_body[1] * odom.v_body[1] +
      odom.v_body[2] * odom.v_body[2]);
  }

  void initializeTrack(
    MarkerTrack & track,
    int marker_id,
    const std::string & label,
    const cv::Vec3d & p_world_marker,
    const std::array<double, 4> & q_world_marker,
    const builtin_interfaces::msg::Time & stamp)
  {
    track.initialized = true;
    track.marker_id = marker_id;
    track.label = label;
    track.have_orientation = true;
    track.q_world_marker = q_world_marker;
    track.last_prediction_stamp = stamp;
    track.last_measurement_stamp = stamp;

    track.kf.statePost = cv::Mat::zeros(3, 1, CV_64F);
    track.kf.statePost.at<double>(0, 0) = p_world_marker[0];
    track.kf.statePost.at<double>(1, 0) = p_world_marker[1];
    track.kf.statePost.at<double>(2, 0) = p_world_marker[2];
    track.kf.statePre = track.kf.statePost.clone();

    cv::setIdentity(track.kf.errorCovPost, cv::Scalar(1.0));
    cv::setIdentity(track.kf.measurementNoiseCov, cv::Scalar(0.01));
    cv::setIdentity(track.kf.processNoiseCov, cv::Scalar(1e-4));
  }

  void predictTrackTo(
    MarkerTrack & track,
    const builtin_interfaces::msg::Time & target_stamp,
    double speed_norm)
  {
    if (!track.initialized) {
      return;
    }

    const double dt = timeDiffSec(target_stamp, track.last_prediction_stamp);
    if (dt <= 1e-6) {
      return;
    }
    if (dt < 0.0) {
      return;
    }

    track.kf.transitionMatrix = cv::Mat::eye(3, 3, CV_64F);

    const double sigma_pos_rate =
      kf_process_accel_sigma_ * (1.0 + kf_speed_noise_scale_ * speed_norm);
    const double q = sigma_pos_rate * sigma_pos_rate * dt;

    track.kf.processNoiseCov = cv::Mat::eye(3, 3, CV_64F) * q;
    track.kf.predict();
    track.last_prediction_stamp = target_stamp;
  }

  void correctTrack(
    MarkerTrack & track,
    const cv::Vec3d & p_world_marker_meas,
    const std::array<double, 4> & q_world_marker_meas,
    double measurement_sigma_m,
    const builtin_interfaces::msg::Time & stamp)
  {
    if (!track.initialized) {
      initializeTrack(track, track.marker_id, track.label, p_world_marker_meas, q_world_marker_meas, stamp);
      return;
    }

    track.kf.measurementNoiseCov =
      cv::Mat::eye(3, 3, CV_64F) * (measurement_sigma_m * measurement_sigma_m);

    cv::Mat measurement(3, 1, CV_64F);
    measurement.at<double>(0, 0) = p_world_marker_meas[0];
    measurement.at<double>(1, 0) = p_world_marker_meas[1];
    measurement.at<double>(2, 0) = p_world_marker_meas[2];

    track.kf.correct(measurement);
    track.q_world_marker = q_world_marker_meas;
    track.have_orientation = true;
    track.last_measurement_stamp = stamp;
    track.last_prediction_stamp = stamp;
  }

  static std::vector<cv::Point3f> makeMarkerObjectPoints(double marker_length)
  {
    const float s = static_cast<float>(marker_length / 2.0);
    return {
      cv::Point3f(-s,  s, 0.0f),
      cv::Point3f( s,  s, 0.0f),
      cv::Point3f( s, -s, 0.0f),
      cv::Point3f(-s, -s, 0.0f)
    };
  }

  void refineCornersSubpixel(
    const cv::Mat & gray,
    std::vector<std::vector<cv::Point2f>> & corners) const
  {
    const cv::Size win(subpix_window_size_, subpix_window_size_);
    const cv::Size zero(subpix_zero_zone_, subpix_zero_zone_);
    const cv::TermCriteria criteria(
      cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
      subpix_max_iterations_,
      subpix_epsilon_);

    for (auto & c : corners) {
      if (c.size() == 4) {
        cv::cornerSubPix(gray, c, win, zero, criteria);
      }
    }
  }

  bool estimateMarkerPoseRefined(
    const std::vector<cv::Point2f> & image_corners,
    const CameraState & camera,
    cv::Vec3d & rvec,
    cv::Vec3d & tvec,
    double & reprojection_error_px) const
  {
    const auto object_points = makeMarkerObjectPoints(marker_length_);

    cv::Vec3d rvec_init, tvec_init;
    const bool ok_ippe = cv::solvePnP(
      object_points,
      image_corners,
      camera.calibration.camera_matrix,
      camera.calibration.dist_coeffs,
      rvec_init,
      tvec_init,
      false,
      cv::SOLVEPNP_IPPE_SQUARE);

    if (!ok_ippe) {
      return false;
    }

    if (use_iterative_pnp_) {
      rvec = rvec_init;
      tvec = tvec_init;
      const bool ok_iter = cv::solvePnP(
        object_points,
        image_corners,
        camera.calibration.camera_matrix,
        camera.calibration.dist_coeffs,
        rvec,
        tvec,
        true,
        cv::SOLVEPNP_ITERATIVE);

      if (!ok_iter) {
        return false;
      }
    } else {
      rvec = rvec_init;
      tvec = tvec_init;
    }

    if (tvec[2] <= 0.0) {
      return false;
    }

    std::vector<cv::Point2f> projected;
    cv::projectPoints(
      object_points,
      rvec,
      tvec,
      camera.calibration.camera_matrix,
      camera.calibration.dist_coeffs,
      projected);

    double sq_err = 0.0;
    for (size_t i = 0; i < image_corners.size(); ++i) {
      const cv::Point2f d = projected[i] - image_corners[i];
      sq_err += static_cast<double>(d.x * d.x + d.y * d.y);
    }
    reprojection_error_px = std::sqrt(sq_err / static_cast<double>(image_corners.size()));
    return reprojection_error_px <= max_reprojection_error_px_;
  }

  static bool isFilteredMeasurementKind(const std::string & output_kind)
  {
    return output_kind == "filtered_measurement";
  }

  void registerMeasurementCandidate(
    int marker_id,
    const std::string & label,
    const geometry_msgs::msg::PoseStamped & pose_msg,
    const std::string & source_topic,
    const std::string & output_kind,
    double reprojection_error_px)
  {
    auto it = cycle_measurement_candidates_.find(marker_id);

    if (it == cycle_measurement_candidates_.end()) {
      CycleMeasurementCandidate candidate;
      candidate.valid = true;
      candidate.marker_id = marker_id;
      candidate.label = label;
      candidate.pose_msg = pose_msg;
      candidate.source_topic = source_topic;
      candidate.output_kind = output_kind;
      candidate.reprojection_error_px = reprojection_error_px;
      cycle_measurement_candidates_[marker_id] = candidate;
      return;
    }

    const bool new_is_filtered = isFilteredMeasurementKind(output_kind);
    const bool old_is_filtered = isFilteredMeasurementKind(it->second.output_kind);

    bool replace = false;
    if (new_is_filtered && !old_is_filtered) {
      replace = true;
    } else if (new_is_filtered == old_is_filtered &&
               reprojection_error_px < it->second.reprojection_error_px) {
      replace = true;
    }

    if (replace) {
      it->second.valid = true;
      it->second.marker_id = marker_id;
      it->second.label = label;
      it->second.pose_msg = pose_msg;
      it->second.source_topic = source_topic;
      it->second.output_kind = output_kind;
      it->second.reprojection_error_px = reprojection_error_px;
    }
  }

  void saveDebugImage(
    const cv::Mat & debug_bgr,
    const std::string & output_path,
    const std::string & topic_name,
    const std::string & preprocessing_variant_name,
    const std::vector<std::vector<cv::Point2f>> & valid_corners,
    const std::vector<int> & valid_ids,
    const std::vector<cv::Vec3d> & rvecs,
    const std::vector<cv::Vec3d> & tvecs,
    const std::vector<bool> & filtered_flags,
    const CameraState & camera)
  {
    if (debug_bgr.empty()) {
      return;
    }

    cv::Mat annotated = debug_bgr.clone();

    if (!valid_ids.empty()) {
      cv::aruco::drawDetectedMarkers(annotated, valid_corners, valid_ids);

      for (size_t i = 0; i < valid_ids.size() && i < rvecs.size() && i < tvecs.size(); ++i) {
        cv::drawFrameAxes(
          annotated,
          camera.calibration.camera_matrix,
          camera.calibration.dist_coeffs,
          rvecs[i],
          tvecs[i],
          static_cast<float>(0.5 * marker_length_));

        const cv::Matx44d T_camera_marker = rvecTvecToHomogeneous(rvecs[i], tvecs[i]);
        const cv::Matx44d T_drone_marker = camera.T_drone_camera * T_camera_marker;

        std::ostringstream line;
        line << "ID " << valid_ids[i]
             << " D=("
             << std::fixed << std::setprecision(2)
             << T_drone_marker(0, 3) << ", "
             << T_drone_marker(1, 3) << ", "
             << T_drone_marker(2, 3) << ")"
             << (filtered_flags[i] ? " FILT" : " RAW");

        const auto & corner = valid_corners[i][0];
        cv::putText(
          annotated,
          line.str(),
          cv::Point(static_cast<int>(corner.x), std::max(20, static_cast<int>(corner.y) - 10)),
          cv::FONT_HERSHEY_SIMPLEX,
          0.5,
          filtered_flags[i] ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 255, 255),
          1,
          cv::LINE_AA);
      }
    }

    std::ostringstream header_line;
    header_line << "Aruco Nano | Topic: " << topic_name
                << " | Valid markers: " << valid_ids.size()
                << " | Frame: " << drone_frame_id_
                << " | Prep: " << preprocessing_variant_name;

    cv::putText(
      annotated,
      header_line.str(),
      cv::Point(20, 30),
      cv::FONT_HERSHEY_SIMPLEX,
      0.6,
      cv::Scalar(0, 255, 0),
      2,
      cv::LINE_AA);

    try {
      const bool ok = cv::imwrite(output_path, annotated);
      if (!ok) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Failed to write debug image: %s", output_path.c_str());
      }
    } catch (const cv::Exception & e) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "OpenCV failed to write debug image %s: %s",
        output_path.c_str(), e.what());
    }
  }

  void publishPoseAndInfo(
    int marker_id,
    const std::string & label,
    const geometry_msgs::msg::PoseStamped & pose_msg,
    const std::string & source_topic,
    const std::string & output_kind)
  {
    pose_pub_->publish(pose_msg);

    if (type_pub_->get_subscription_count() > 0U) {
      std_msgs::msg::String type_msg;
      type_msg.data = label;
      type_pub_->publish(type_msg);
    }

    if (msg_pub_->get_subscription_count() > 0U) {
      std_msgs::msg::String info_msg;
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
      info_msg.data = oss.str();
      msg_pub_->publish(info_msg);
    }
  }

  void publishCycleMeasurementCandidates()
  {
    for (const auto & [marker_id, candidate] : cycle_measurement_candidates_) {
      if (!candidate.valid) {
        continue;
      }

      publishPoseAndInfo(
        marker_id,
        candidate.label,
        candidate.pose_msg,
        candidate.source_topic,
        candidate.output_kind);
    }
  }

  void publishPredictedTracks()
  {
    if (!use_track_filter_) {
      return;
    }

    const SlamOdomState odom = getLatestSlamOdom();
    if (!odom.valid) {
      return;
    }

    const cv::Matx44d T_world_body = matrixFromPose(odom.p_world_body, odom.q_world_body);
    const cv::Matx44d T_body_world = invertTransform(T_world_body);
    const double speed_norm = computeSpeedNorm(odom);

    for (auto & [marker_id, track] : tracks_) {
      if (!track.initialized) {
        continue;
      }

      if (cycle_measurement_candidates_.find(marker_id) != cycle_measurement_candidates_.end()) {
        continue;
      }

      if (std::abs(timeDiffSec(odom.stamp, track.last_measurement_stamp)) > max_track_age_sec_) {
        continue;
      }

      predictTrackTo(track, odom.stamp, speed_norm);

      const cv::Vec3d p_world_marker(
        track.kf.statePre.at<double>(0, 0),
        track.kf.statePre.at<double>(1, 0),
        track.kf.statePre.at<double>(2, 0));

      if (!track.have_orientation) {
        continue;
      }

      const cv::Matx44d T_world_marker = matrixFromPose(p_world_marker, track.q_world_marker);
      const cv::Matx44d T_body_marker = T_body_world * T_world_marker;

      auto pose_msg = matrixToPoseStamped(T_body_marker, odom.stamp);
      publishPoseAndInfo(marker_id, track.label, pose_msg, "track_prediction", "track_prediction");
    }
  }

  void pruneStaleTracks()
  {
    const SlamOdomState odom = getLatestSlamOdom();
    if (!odom.valid) {
      return;
    }

    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
      const auto & track = it->second;
      if (!track.initialized ||
          std::abs(timeDiffSec(odom.stamp, track.last_measurement_stamp)) > max_track_age_sec_) {
        it = tracks_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void processImage(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const std::string & topic_name,
    CameraState & camera)
  {
    cv_bridge::CvImageConstPtr cv_ptr;

    try {
      cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "cv_bridge failed for %s: %s", topic_name.c_str(), e.what());
      return;
    }

    cv::Mat debug_bgr;
    cv::Mat gray;
    const bool need_debug_image = save_debug_images_;

    try {
      const auto & enc = msg->encoding;

      if (enc == sensor_msgs::image_encodings::MONO8) {
        gray = cv_ptr->image;
        if (need_debug_image) {
          cv::cvtColor(cv_ptr->image, debug_bgr, cv::COLOR_GRAY2BGR);
        }
      } else if (enc == sensor_msgs::image_encodings::BGR8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
        if (need_debug_image) {
          debug_bgr = cv_ptr->image.clone();
        }
      } else if (enc == sensor_msgs::image_encodings::RGB8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_RGB2GRAY);
        if (need_debug_image) {
          cv::cvtColor(cv_ptr->image, debug_bgr, cv::COLOR_RGB2BGR);
        }
      } else if (enc == sensor_msgs::image_encodings::BGRA8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGRA2GRAY);
        if (need_debug_image) {
          cv::cvtColor(cv_ptr->image, debug_bgr, cv::COLOR_BGRA2BGR);
        }
      } else if (enc == sensor_msgs::image_encodings::RGBA8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_RGBA2GRAY);
        if (need_debug_image) {
          cv::cvtColor(cv_ptr->image, debug_bgr, cv::COLOR_RGBA2BGR);
        }
      } else {
        auto bgr_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::cvtColor(bgr_ptr->image, gray, cv::COLOR_BGR2GRAY);
        if (need_debug_image) {
          debug_bgr = bgr_ptr->image.clone();
        }
      }
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Image conversion failed for %s: %s", topic_name.c_str(), e.what());
      return;
    }

    DetectionSelection detection_selection;
    try {
      selectBestDetection(gray, detection_selection);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Aruco Nano detection failed for %s: %s", topic_name.c_str(), e.what());
      return;
    }

    std::vector<std::vector<cv::Point2f>> corners = detection_selection.corners;
    std::vector<int> ids = detection_selection.ids;

    if (ids.empty()) {
      std::string debug_output_path;
      if (shouldSaveDebugImage(msg, camera, false, debug_output_path)) {
        saveDebugImage(
          debug_bgr,
          debug_output_path,
          topic_name,
          detection_selection.candidate_name,
          {},
          {},
          {},
          {},
          {},
          camera);
      }
      return;
    }

    refineCornersSubpixel(detection_selection.refine_image, corners);

    std::vector<std::vector<cv::Point2f>> valid_corners;
    std::vector<int> valid_ids;
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;
    std::vector<bool> filtered_flags;

    valid_corners.reserve(ids.size());
    valid_ids.reserve(ids.size());
    rvecs.reserve(ids.size());
    tvecs.reserve(ids.size());
    filtered_flags.reserve(ids.size());

    const SlamOdomState odom = getLatestSlamOdom();
    const bool odom_valid_for_image =
      odom.valid &&
      std::abs(timeDiffSec(odom.stamp, msg->header.stamp)) <= max_odom_time_diff_sec_;

    const cv::Matx44d T_world_body =
      odom_valid_for_image ? matrixFromPose(odom.p_world_body, odom.q_world_body) : cv::Matx44d::eye();
    const cv::Matx44d T_body_world =
      odom_valid_for_image ? invertTransform(T_world_body) : cv::Matx44d::eye();
    const double speed_norm = computeSpeedNorm(odom);

    for (size_t i = 0; i < ids.size(); ++i) {
      const auto label_it = marker_labels_.find(ids[i]);
      if (label_it == marker_labels_.end()) {
        continue;
      }

      cv::Vec3d rvec, tvec;
      double reprojection_error_px = 0.0;
      if (!estimateMarkerPoseRefined(corners[i], camera, rvec, tvec, reprojection_error_px)) {
        continue;
      }

      const cv::Matx44d T_camera_marker = rvecTvecToHomogeneous(rvec, tvec);
      const cv::Matx44d T_drone_marker_raw = camera.T_drone_camera * T_camera_marker;

      auto raw_pose_msg = matrixToPoseStamped(T_drone_marker_raw, msg->header.stamp);
      if (publish_raw_pose_) {
        raw_pose_pub_->publish(raw_pose_msg);
      }

      geometry_msgs::msg::PoseStamped out_pose_msg = raw_pose_msg;
      std::string output_kind = "raw_measurement";
      bool used_filtered_measurement = false;

      if (use_track_filter_ && odom_valid_for_image) {
        const cv::Matx44d T_world_marker_meas = T_world_body * T_drone_marker_raw;

        const cv::Vec3d p_world_marker_meas(
          T_world_marker_meas(0, 3),
          T_world_marker_meas(1, 3),
          T_world_marker_meas(2, 3));

        cv::Matx33d R_world_marker(
          T_world_marker_meas(0, 0), T_world_marker_meas(0, 1), T_world_marker_meas(0, 2),
          T_world_marker_meas(1, 0), T_world_marker_meas(1, 1), T_world_marker_meas(1, 2),
          T_world_marker_meas(2, 0), T_world_marker_meas(2, 1), T_world_marker_meas(2, 2)
        );
        const auto q_world_marker_meas = rotationMatrixToQuaternion(R_world_marker);

        auto & track = tracks_[ids[i]];
        if (!track.initialized) {
          track.marker_id = ids[i];
          track.label = label_it->second;
          initializeTrack(track, ids[i], label_it->second, p_world_marker_meas, q_world_marker_meas, odom.stamp);
        } else {
          predictTrackTo(track, odom.stamp, speed_norm);

          const double focal_avg =
            0.5 * (camera.calibration.camera_matrix.at<double>(0, 0) +
                   camera.calibration.camera_matrix.at<double>(1, 1));
          const double depth = std::max(0.05, tvec[2]);
          const double sigma_from_px = depth * reprojection_error_px / std::max(1.0, focal_avg);
          const double measurement_sigma_m = std::max(kf_measurement_noise_floor_m_, sigma_from_px);

          correctTrack(track, p_world_marker_meas, q_world_marker_meas, measurement_sigma_m, odom.stamp);
        }

        const cv::Vec3d p_world_marker_filt(
          track.kf.statePost.at<double>(0, 0),
          track.kf.statePost.at<double>(1, 0),
          track.kf.statePost.at<double>(2, 0));

        const cv::Matx44d T_world_marker_filt = matrixFromPose(p_world_marker_filt, track.q_world_marker);
        const cv::Matx44d T_drone_marker_filt = T_body_world * T_world_marker_filt;

        out_pose_msg = matrixToPoseStamped(T_drone_marker_filt, msg->header.stamp);
        output_kind = "filtered_measurement";
        used_filtered_measurement = true;
      }

      registerMeasurementCandidate(
        ids[i],
        label_it->second,
        out_pose_msg,
        topic_name,
        output_kind,
        reprojection_error_px);

      valid_corners.push_back(corners[i]);
      valid_ids.push_back(ids[i]);
      rvecs.push_back(rvec);
      tvecs.push_back(tvec);
      filtered_flags.push_back(used_filtered_measurement);
    }

    std::string debug_output_path;
    if (shouldSaveDebugImage(msg, camera, !valid_ids.empty(), debug_output_path)) {
      saveDebugImage(
        debug_bgr,
        debug_output_path,
        topic_name,
        detection_selection.candidate_name,
        valid_corners,
        valid_ids,
        rvecs,
        tvecs,
        filtered_flags,
        camera);
    }
  }

  cv::Matx44d rvecTvecToHomogeneous(const cv::Vec3d & rvec, const cv::Vec3d & tvec) const
  {
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);

    cv::Matx44d T = cv::Matx44d::eye();

    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        T(r, c) = R_cv.at<double>(r, c);
      }
    }

    T(0, 3) = tvec[0];
    T(1, 3) = tvec[1];
    T(2, 3) = tvec[2];

    return T;
  }

  geometry_msgs::msg::PoseStamped matrixToPoseStamped(
    const cv::Matx44d & T,
    const builtin_interfaces::msg::Time & stamp) const
  {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = drone_frame_id_;

    pose_msg.pose.position.x = T(0, 3);
    pose_msg.pose.position.y = T(1, 3);
    pose_msg.pose.position.z = T(2, 3);

    cv::Matx33d R(
      T(0, 0), T(0, 1), T(0, 2),
      T(1, 0), T(1, 1), T(1, 2),
      T(2, 0), T(2, 1), T(2, 2)
    );

    const auto q = rotationMatrixToQuaternion(R);

    pose_msg.pose.orientation.x = q[0];
    pose_msg.pose.orientation.y = q[1];
    pose_msg.pose.orientation.z = q[2];
    pose_msg.pose.orientation.w = q[3];

    return pose_msg;
  }

  std::array<double, 4> rotationMatrixToQuaternion(const cv::Matx33d & R) const
  {
    std::array<double, 4> q{};
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);

    if (tr > 0.0) {
      const double S = std::sqrt(tr + 1.0) * 2.0;
      q[3] = 0.25 * S;
      q[0] = (R(2, 1) - R(1, 2)) / S;
      q[1] = (R(0, 2) - R(2, 0)) / S;
      q[2] = (R(1, 0) - R(0, 1)) / S;
    } else if ((R(0, 0) > R(1, 1)) && (R(0, 0) > R(2, 2))) {
      const double S = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
      q[3] = (R(2, 1) - R(1, 2)) / S;
      q[0] = 0.25 * S;
      q[1] = (R(0, 1) + R(1, 0)) / S;
      q[2] = (R(0, 2) + R(2, 0)) / S;
    } else if (R(1, 1) > R(2, 2)) {
      const double S = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
      q[3] = (R(0, 2) - R(2, 0)) / S;
      q[0] = (R(0, 1) + R(1, 0)) / S;
      q[1] = 0.25 * S;
      q[2] = (R(1, 2) + R(2, 1)) / S;
    } else {
      const double S = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
      q[3] = (R(1, 0) - R(0, 1)) / S;
      q[0] = (R(0, 2) + R(2, 0)) / S;
      q[1] = (R(1, 2) + R(2, 1)) / S;
      q[2] = 0.25 * S;
    }

    return normalizeQuaternion(q);
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
