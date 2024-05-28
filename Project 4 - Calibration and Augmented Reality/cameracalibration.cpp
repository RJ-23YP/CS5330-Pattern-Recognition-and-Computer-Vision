/*
  Ruchik Jani (NUID - 002825482)
  Spring 2024
  CS 5330 Computer Vision

  This program is the camera calibration code where we are extracting target corners and saving calibration images to calculate the calibration parameters which are saved to an XML file. 
*/


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// Define global variables for storing calibration data
std::vector<cv::Vec3f> point_set; // Set of 3D world points corresponding to detected corners
std::vector<std::vector<cv::Vec3f>> point_list; // List of sets of 3D world points for each calibration image
std::vector<std::vector<cv::Point2f>> corner_list; // List of sets of 2D image points (corner coordinates) for each calibration image
std::vector<cv::Point2f> corner_set; // Detected corner coordinates for a single frame
cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F); // Intrinsic camera matrix
cv::Mat distortion_coefficients; // Distortion coefficients
double reprojection_error; // Reprojection error after camera calibration


/**
 * brief description: Detects and extracts target corners in a given frame.
 * 
 * parameter: frame - The input frame containing the target.
 */
void detectAndExtractCorners(cv::Mat frame) {
    // Convert the frame to grayscale
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

    // Find chessboard corners
    bool found = cv::findChessboardCorners(grayFrame, cv::Size(9, 6), corner_set);

    if (found) {
        // Refine corner locations
        cv::cornerSubPix(grayFrame, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
        // Draw corners on the frame
        cv::drawChessboardCorners(frame, cv::Size(9, 6), corner_set, found);
        //std::cout << "Number of corners found: " << corner_set.size() << std::endl;
        //std::cout << "Coordinates of the first corner: " << corner_set[0] << std::endl;
    }
}

/**
 * brief description: Saves the corner locations and corresponding 3D world points for calibration.
 * 
 * parameter: frame - The frame containing the detected corners.
 * parameter: calibrationImagesDir - The directory to save calibration images.
 */
void saveCalibrationData(cv::Mat frame, const std::string& calibrationImagesDir) {
    if (!corner_set.empty()) {
        // Generate 3D world points corresponding to the corners
        point_set.clear();
        for (int i = 0; i < corner_set.size(); ++i) {
            point_set.push_back(cv::Vec3f(i / 9, i % 9, 0));
        }
        // Save corner locations and 3D points
        corner_list.push_back(corner_set);
        point_list.push_back(point_set);
        // Save the frame as a calibration image
        std::string filename = calibrationImagesDir + "/image_" + std::to_string(point_list.size()) + ".jpg";
        cv::imwrite(filename, frame);
    }
}

/**
 * brief description: Calibrates the camera using the saved calibration data.
 * 
 * parameter: xmlFile - The XML file to store intrinsic parameters.
 */
void calibrateCamera(const std::string& xmlFile) {
    if (point_list.size() >= 5) {
        // Initialize variables
        std::vector<cv::Mat> rotations, translations;
        distortion_coefficients = cv::Mat::zeros(1, 5, CV_64F);
        std::vector<double> per_view_errors;

        // Size of the calibration images
        cv::Size imageSize(640, 480);

        // Print calibration results
        std::cout << "Camera matrix before calibration:\n" << camera_matrix << std::endl;
        std::cout << "Distortion coefficients before calibration:\n" << distortion_coefficients << std::endl;

        // Perform camera calibration
        reprojection_error = cv::calibrateCamera(point_list, corner_list, imageSize, camera_matrix,
                                                 distortion_coefficients, rotations, translations,
                                                 cv::CALIB_FIX_ASPECT_RATIO);

        // Print calibration results
        std::cout << "Camera matrix after calibration:\n" << camera_matrix << std::endl;
        std::cout << "Distortion coefficients after calibration:\n" << distortion_coefficients << std::endl;
        std::cout << "Reprojection error: " << reprojection_error << std::endl;

        // Write intrinsic parameters to XML file
        cv::FileStorage fs(xmlFile, cv::FileStorage::WRITE);
        fs << "camera_matrix" << camera_matrix;
        fs << "distortion_coefficients" << distortion_coefficients;
        fs.release();
    } else {
        std::cerr << "Not enough calibration frames selected (need at least 5)." << std::endl;
    }
}


int main() {
    // Path to XML file to save intrinsic parameters
    std::string xmlFile = "/home/rj/Project-4/data/intrinsicParameters.xml";
    // Directory to store captured calibration images
    std::string calibrationImagesDir = "/home/rj/Project-4/data/captured";


     // Ask the user for input: webcam or video feed from URL
    std::cout << "Select input source:\n";
    std::cout << "1. Webcam\n";
    std::cout << "2. Video feed from phone\n";
    std::cout << "Enter your choice (1 or 2): ";
    int choice;
    std::cin >> choice;

    cv::VideoCapture cap;
    if (choice == 1) {
        // Use webcam
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open webcam" << std::endl;
            return -1;
        }
    } else if (choice == 2) {
        //Use Video streaming from phone
        std::string streamURL = "http://10.110.39.157:4747/video"; // Enter the IP address of your webcam stream
        cap.open(streamURL);
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open video feed from phone" << std::endl;
            return -1;
        }
    } else {
        std::cerr << "Error: Invalid choice" << std::endl;
        return -1;
    }

    cv::Mat frame; // Container for each captured frame

    // Main loop to capture frames and perform tasks
    while (true) {
        // Capture a frame from the camera
        cap >> frame;
        // Check if the frame is empty
        if (frame.empty()) {
            // Display error message if frame cannot be read
            std::cerr << "Error: Could not read frame from the camera" << std::endl;
            break;
        }

        detectAndExtractCorners(frame); // Task 1: Detect and extract target corners

        cv::imshow("Frame", frame); // Display the frame

        char key = cv::waitKey(1); // Wait for key press
        // Check if the 'q' key is pressed to quit the program
        if (key == 'q') {
            break; // Quit the program
        } else if (key == 's') {
            saveCalibrationData(frame, calibrationImagesDir); // Task 2: Save calibration data
            calibrateCamera(xmlFile); // Task 3: Calibrate the camera
        }
    }
    
    cap.release(); // Release the video capture device
    cv::destroyAllWindows(); // Close all OpenCV windows
    return 0; // Exit the program
}




