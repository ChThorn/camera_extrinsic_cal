cmake_minimum_required(VERSION 3.10)
project(camera_extrinsic_cal)

# Find required packages
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${realsense2_INCLUDE_DIR}
)

# Create executable
add_executable(DialogExtrinsic DialogExtrinsic.cpp)

# Link libraries
target_link_libraries(DialogExtrinsic 
    ${realsense2_LIBRARY}
    ${OpenCV_LIBS}
    Eigen3::Eigen
)