/*hand-eye calibration using TSAI method with pre-calibrated intrinsics*/
#include <stdlib.h>
#include <iostream>
#include <fstream>

// Eigen headers must come first
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>

// OpenCV headers after Eigen
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#define PI 3.1415926

using namespace std;
using namespace cv;
using namespace Eigen;

// Configuration parameters
int num_of_all_images = 10;  // Number of images to process
cv::Size board_size = cv::Size(9, 6);  // Chessboard internal corners
cv::Size2f square_size = cv::Size2f(25, 25);  // Square size in mm

// Function to load camera calibration data from YAML
void loadCalibrationData(const string& filename, cv::Mat& camera_matrix, cv::Mat& dist_coeffs)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        cerr << "Failed to open " << filename << endl;
        exit(-1);
    }

    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;
    
    // Optional: Read image dimensions if needed
    int image_width = (int)fs["image_width"];
    int image_height = (int)fs["image_height"];
    
    fs.release();

    // Verify camera matrix format
    if(camera_matrix.rows != 3 || camera_matrix.cols != 3)
    {
        cerr << "Invalid camera matrix format" << endl;
        exit(-1);
    }

    // Verify distortion coefficients format
    if(dist_coeffs.cols != 5 || dist_coeffs.rows != 1)
    {
        cerr << "Invalid distortion coefficients format" << endl;
        exit(-1);
    }
}

void loadRobotPoses(const string& filename, std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& bHg)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        cerr << "Failed to open " << filename << endl;
        exit(-1);
    }

    int num_poses = (int)fs["num_poses"];
    cv::FileNode poses = fs["poses"];
    
    for(const auto& pose_entry : poses)
    {
        try {
            // Access position and orientation arrays
            std::vector<double> position, orientation;
            
            // First node contains the position and orientation
            cv::FileNode pose_data = pose_entry.begin().operator*();
            
            // Read arrays directly
            pose_data["position"] >> position;
            pose_data["orientation"] >> orientation;

            if(position.size() != 3 || orientation.size() != 3) {
                cerr << "Invalid position or orientation data in pose file" << endl;
                continue;
            }

            // Debug output
            cout << "Loading pose with position: [" 
                 << position[0] << ", " << position[1] << ", " << position[2] << "] "
                 << "orientation: [" 
                 << orientation[0] << ", " << orientation[1] << ", " << orientation[2] << "]" 
                 << endl;

            // Convert to transformation matrix
            Eigen::AngleAxisd rollAngle(orientation[0] * PI/180, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd pitchAngle(orientation[1] * PI/180, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd yawAngle(orientation[2] * PI/180, Eigen::Vector3d::UnitZ());

            Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
            transform.block<3,3>(0,0) = (yawAngle * pitchAngle * rollAngle).matrix();
            transform.block<3,1>(0,3) = Eigen::Vector3d(position[0], position[1], position[2]);

            bHg.push_back(transform.inverse());
        }
        catch (const std::exception& e) {
            cerr << "Error processing pose: " << e.what() << endl;
            continue;
        }
    }

    fs.release();

    if(bHg.size() != num_poses) {
        cerr << "Warning: Number of loaded poses (" << bHg.size() 
             << ") doesn't match expected number (" << num_poses << ")" << endl;
    }
    else {
        cout << "Successfully loaded " << bHg.size() << " poses" << endl;
    }
}

// Helper function to create skew matrix
Eigen::Matrix3d skew(Eigen::Vector3d V) {
    Eigen::Matrix3d S;
    S << 0, -V(2), V(1),
         V(2), 0, -V(0),
         -V(1), V(0), 0;
    return S;
}

// Convert quaternion to rotation matrix
Eigen::Matrix4d quat2rot(Eigen::Vector3d q) {
    double p = q.transpose() * q;
    if (p > 1)
        std::cout << "Warning: quat2rot: quaternion greater than 1";
    double w = sqrt(1 - p);
    Eigen::Matrix4d R = Eigen::MatrixXd::Identity(4, 4);
    Eigen::Matrix3d res;
    res = 2 * (q*q.transpose()) + 2 * w*skew(q);
    res = res + Eigen::MatrixXd::Identity(3, 3) - 2 * p*Eigen::MatrixXd::Identity(3, 3);
    R.topLeftCorner(3, 3) = res;
    return R;
}

// Convert rotation matrix to quaternion
Eigen::Vector3d rot2quat(Eigen::MatrixXd R) {
    double w4 = 2 * sqrt(1 + R.topLeftCorner(3, 3).trace());
    Eigen::Vector3d q;
    q << (R(2, 1) - R(1, 2)) / w4,
         (R(0, 2) - R(2, 0)) / w4,
         (R(1, 0) - R(0, 1)) / w4;
    return q;
}

// Create transformation matrix from translation vector
Eigen::Matrix4d transl(Eigen::Vector3d x) {
    Eigen::Matrix4d r = Eigen::MatrixXd::Identity(4, 4);
    r.topRightCorner(3, 1) = x;
    return r;
}

// ------Addition consideration in case the offset Gripper has added (gripper L shape)--
Eigen::Matrix4d createTcpToGripperTransform(bool use_gripper_offset,
                                            double x_offset = 0.0,
                                            double y_offset = 0.0,
                                            double z_offset = 0.0,
                                            double roll = 0.0,
                                            double pitch = 0.0,
                                            double yaw = 0.0)
{
    Eigen::Matrix4d gHe = Eigen::Matrix4d::Identity();

    if(!use_gripper_offset)
    {
        return gHe; // Return identity matrix if no offset needed
    }

    // Convert angles to radians
    roll *= PI/180.0;
    pitch *= PI/180.0;
    yaw *= PI/180.0;

    // Create rotation matrix
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    // Set rotation
    gHe.block<3,3>(0, 0) = (yawAngle * pitchAngle * rollAngle).matrix();

    // Set translation
    gHe.block<3, 1> (0, 3) = Eigen::Vector3d(x_offset, y_offset, z_offset);

    return gHe;
}
//-----------------------end----

// Hand-eye calibration using all possible pairs
Eigen::Matrix4d handEye(std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> bHg,
                       std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> cHw) {
    int M = bHg.size();
    int K = (M*M - M) / 2;
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * K, 3);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3 * K, 1);
    int k = 0;

    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Hg = bHg;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Hc = cHw;

    // Calculate rotation component
    for (int i = 0; i < M; i++) {
        for (int j = i + 1; j < M; j++) {
            Eigen::Matrix4d Hgij = Hg.at(j).lu().solve(Hg.at(i));
            Eigen::Vector3d Pgij = 2 * rot2quat(Hgij);

            Eigen::Matrix4d Hcij = Hc.at(j) * Hc.at(i).inverse();
            Eigen::Vector3d Pcij = 2 * rot2quat(Hcij);

            k = k + 1;
            A.block(3 * k - 3, 0, 3, 3) = skew(Pgij + Pcij);
            B.block(3 * k - 3, 0, 3, 1) = Pcij - Pgij;
        }
    }

    Eigen::Vector3d Pcg_ = A.colPivHouseholderQr().solve(B);
    Eigen::Vector3d Pcg = 2 * Pcg_ / sqrt(1 + (double)(Pcg_.transpose()*Pcg_));
    Eigen::Matrix4d Rcg = quat2rot(Pcg / 2);

    // Calculate translation component
    k = 0;
    A = Eigen::MatrixXd::Zero(3 * K, 3);
    B = Eigen::MatrixXd::Zero(3 * K, 1);
    
    for (int i = 0; i < M; i++) {
        for (int j = i + 1; j < M; j++) {
            Eigen::Matrix4d Hgij = Hg.at(j).lu().solve(Hg.at(i));
            Eigen::Matrix4d Hcij = Hc.at(j) * Hc.at(i).inverse();

            k = k + 1;
            A.block(3 * k - 3, 0, 3, 3) = Hgij.topLeftCorner(3, 3) - Eigen::MatrixXd::Identity(3, 3);
            B.block(3 * k - 3, 0, 3, 1) = Rcg.topLeftCorner(3, 3)*Hcij.block(0, 3, 3, 1) - Hgij.block(0, 3, 3, 1);
        }
    }

    Eigen::Vector3d Tcg = A.colPivHouseholderQr().solve(B);
    return transl(Tcg) * Rcg;
}

// Main hand-eye calibration function
int handEye_calib(Eigen::Matrix4d &gHc, std::string path, bool use_gripper_offset = false) {
    ofstream ofs(path + "/output.txt");
    std::vector<cv::Mat> images;
    
    // Load camera calibration data
    cv::Mat camera_matrix, dist_coeffs;
    loadCalibrationData(path + "/intrinsic_calibrated.yaml", camera_matrix, dist_coeffs);
    
    // Read images
    std::cout << "Reading images..." << std::endl;
    for (int i = 0; i < num_of_all_images; i++) {
        std::string image_path = path + "/" + std::to_string(i) + ".png";
        cv::Mat image = cv::imread(image_path, 0);
        if (!image.empty())
            images.push_back(image);
        else {
            std::cout << "Cannot find " << image_path << std::endl;
            return -1;
        }
    }

    // Extract chessboard corners
    std::cout << "Extracting chessboard corners..." << std::endl;
    std::vector<std::vector<cv::Point2f>> image_points_seq;
    std::vector<cv::Point3f> object_point_template;
    
    // Create template for object points
    for (int i = 0; i < board_size.height; i++) {
        for (int j = 0; j < board_size.width; j++) {
            object_point_template.push_back(
                cv::Point3f(j * square_size.width, i * square_size.height, 0));
        }
    }

    // // Find corners in each image
    // for (size_t i = 0; i < images.size(); i++) {
    //     std::vector<cv::Point2f> corners;
    //     bool found = cv::findChessboardCorners(images[i], board_size, corners);
        
    //     if (found) {
    //         cv::cornerSubPix(images[i], corners, cv::Size(11, 11), cv::Size(-1, -1),
    //             cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER::MAX_ITER, 30, 0.1));
    //         image_points_seq.push_back(corners);

    //         // Optional: Draw and save corners
    //         cv::Mat colored_img;
    //         cv::cvtColor(images[i], colored_img, COLOR_GRAY2BGR);
    //         cv::drawChessboardCorners(colored_img, board_size, corners, found);
    //         cv::imwrite(path + "/corners_" + std::to_string(i) + ".png", colored_img);
    //     }
    // }
    
    // Find corners in each image
    for (size_t i = 0; i < images.size(); i++) {
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(images[i], board_size, corners);
        
        if (found) {
            cv::cornerSubPix(images[i], corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            image_points_seq.push_back(corners);

            // Optional: Draw and save corners
            cv::Mat colored_img;
            cv::cvtColor(images[i], colored_img, cv::COLOR_GRAY2BGR);
            cv::drawChessboardCorners(colored_img, board_size, corners, found);
            cv::imwrite(path + "/corners_" + std::to_string(i) + ".png", colored_img);
        }
    }

    // Calculate extrinsic parameters for each image
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> cHw;
    for (size_t i = 0; i < image_points_seq.size(); i++) {
        cv::Mat rvec, tvec;
        std::vector<cv::Point3f> object_points = object_point_template;
        
        cv::solvePnP(object_points, image_points_seq[i], camera_matrix, 
                     dist_coeffs, rvec, tvec);

        // Convert to transformation matrix
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        // cv::cv2eigen(R, transform.block<3,3>(0,0));
        Eigen::Matrix3d R_eigen;
        cv::cv2eigen(R, R_eigen);
        transform.block<3,3>(0, 0) = R_eigen;
        transform.block<3,1>(0,3) = Eigen::Vector3d(tvec.at<double>(0),
                                                   tvec.at<double>(1),
                                                   tvec.at<double>(2));
        cHw.push_back(transform);
    }

    // Read robot poses from YAML file
    std::cout << "Reading robot poses..." << std::endl;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> bHg;
    try {
        loadRobotPoses(path + "/robot_poses.yaml", bHg);
    } catch (const std::exception& e) {
        std::cout << "Error loading robot poses: " << e.what() << std::endl;
        return -1;
    }

    // Perform hand-eye calibration
    if (bHg.size() == cHw.size() && !bHg.empty()) {
        std::cout << "Performing hand-eye calibration..." << std::endl;
        Eigen::Matrix4d eHc = handEye(bHg, cHw); // Get TCP to camera transform

        // For example gripper offset (modify these values based on your gripper)
        double x_offset = 0.0;      // mm
        double y_offset = 0.0;      // mm
        double z_offset = 100.0;    // mm (example: 100mm extension)
        double roll = 0.0;          // degrees
        double pitch = 45.0;        // degrees (example: 45 degree bend)
        double yaw = 0.0;           // degrees

        // Create gripper transform
        Eigen::Matrix4d gHe = createTcpToGripperTransform(use_gripper_offset,
                                                          x_offset, y_offset, z_offset,
                                                          roll, pitch, yaw);
        
        // Calculate final transformation
        gHc = gHe * eHc;
        // gHc = handEye(bHg, cHw);
        
        // Save results
        ofs << "Hand-eye calibration result (gHc):" << std::endl;
        ofs << gHc << std::endl;

        if(use_gripper_offset)
        {
            ofs << "\nGripper offset transform (gHe - Gripper to TCP): "<<std::endl;
            ofs << gHe << std::endl;

            ofs << "\nFinal tranform with gripper offset (gHc - Gripper to Camera): "<<std::endl;
            ofs << gHc << std::endl;
        }
        
        std::cout << "Calibration complete. Results saved to output.txt" << std::endl;
        return 0;
    }
    
    std::cout << "Error: Mismatched number of poses and images" << std::endl;
    return -1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <data_path> [user_gripper_offset]" << std::endl;
        return -1;
    }

    Eigen::Matrix4d gHc;
    // return handEye_calib(gHc, argv[1]);
    bool user_gripper_offset = (argc > 2) ? std::atoi(argv[2]) : false;
    return handEye_calib(gHc, argv[1], user_gripper_offset);
}