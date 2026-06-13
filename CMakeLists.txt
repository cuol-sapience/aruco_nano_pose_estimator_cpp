cmake_minimum_required(VERSION 3.8)
project(aruco_nano_pose_estimator_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV 4.9 REQUIRED COMPONENTS core imgproc imgcodecs aruco calib3d video)

# GStreamer (pipeline management + appsink pull API)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)

# Jetson NvBufSurface (NVMM direct-map, zero DMA copy from ISP)
find_path(NVBUFSURFACE_INCLUDE_DIR nvbufsurface.h
  PATHS
    /usr/src/jetson_multimedia_api/include
    /usr/local/cuda/include
    /usr/include
  DOC "Path to nvbufsurface.h"
)
find_library(NVBUFSURFACE_LIB
  NAMES nvbufsurface
  PATHS
    /usr/lib/aarch64-linux-gnu/tegra
    /usr/lib/aarch64-linux-gnu
  DOC "NvBufSurface library"
)

if(NOT NVBUFSURFACE_INCLUDE_DIR OR NOT NVBUFSURFACE_LIB)
  message(WARNING "nvbufsurface not found — NVMM path will fall back to system-RAM copy")
  set(NVBUFSURFACE_INCLUDE_DIR "")
  set(NVBUFSURFACE_LIB "")
endif()

add_executable(aruco_nano_pose_estimator
  src/aruco_nano_pose_estimator.cpp
)

ament_target_dependencies(aruco_nano_pose_estimator
  rclcpp
  geometry_msgs
  std_msgs
  nav_msgs
)

target_include_directories(aruco_nano_pose_estimator PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${GSTREAMER_APP_INCLUDE_DIRS}
  ${NVBUFSURFACE_INCLUDE_DIR}
)

target_link_libraries(aruco_nano_pose_estimator
  ${OpenCV_LIBS}
  ${GSTREAMER_LIBRARIES}
  ${GSTREAMER_APP_LIBRARIES}
  ${NVBUFSURFACE_LIB}
)

install(TARGETS
  aruco_nano_pose_estimator
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
