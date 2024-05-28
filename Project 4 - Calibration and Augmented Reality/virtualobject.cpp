/*
  Ruchik Jani (NUID - 002825482)
  Spring 2024
  CS 5330 Computer Vision

  This program is the code reads the camera calibration parameters from the XML file, and uses it for projecting 3D axes, overlaying virtual objects and Harris corner detection. 
*/


#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // Load camera calibration parameters from XML file
    cv::FileStorage fs("/home/rj/Project-4/data/intrinsicParameters.xml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open calibration file" << std::endl;
        return -1;
    }

    cv::Mat camera_matrix, distortion_coefficients;
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> distortion_coefficients;
    fs.release(); // Release file

    // Ask the user for input: webcam or video feed from URL
    std::cout << "Select input source:\n";
    std::cout << "1. Video feed from Phone\n";
    std::cout << "2. Pre-recorded video\n";
    std::cout << "Enter your choice (1 or 2): ";
    int choice;
    std::cin >> choice;
    
    cv::VideoCapture cap;
    if (choice == 1) {
        // Use video stream from phone
        std::string streamURL = "http://10.110.39.157:4747/video"; // Enter the IP address of your webcam stream
        cap.open(streamURL);
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open video feed from phone" << std::endl;
            return -1;
        }
    } else if (choice == 2) {
        // Use pre-recorded video
        std::string video = "/home/rj/Project-4/data/WhatsApp Video 2024-03-21 at 8.42.23 PM.mp4"; // Enter the file location of your pre-recorded video
        cap.open(video);
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open pre-recorded video" << std::endl;
            return -1;
        }
    } else {
        std::cerr << "Error: Invalid choice" << std::endl;
        return -1;
    }

    // Define checkerboard parameters
    cv::Size boardSize(9, 6);
    float squareSize = 0.1; // size of one square in meters

    // Create vectors to store detected corners and object points
    std::vector<cv::Point2f> corners;
    std::vector<cv::Point3f> objPoints;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objPoints.push_back(cv::Point3f((j - boardSize.width / 2) * squareSize, (i - boardSize.height / 2) * squareSize, 0));
        }
    }

    // Define 3D points for the virtual object (a simple pyramid)
    std::vector<cv::Point3f> objectPoints;
    objectPoints.push_back(cv::Point3f(-0.05, -0.05, 0));
    objectPoints.push_back(cv::Point3f(0.05, -0.05, 0));
    objectPoints.push_back(cv::Point3f(0.05, 0.05, 0));
    objectPoints.push_back(cv::Point3f(-0.05, 0.05, 0));
    objectPoints.push_back(cv::Point3f(0, 0, 0.1)); // Apex

    bool displayAxis = false;
    bool displayPyramid = false;
    bool detectHarris = false; // Flag to indicate whether to detect Harris corners

    // Parameters for Harris corner detection
    int blockSize = 5;
    int apertureSize = 7;
    double k = 0.06;
    double threshold = 125;

    cv::Mat frame;
    while (cap.read(frame)) {
        // Find chessboard corners
        bool found = cv::findChessboardCorners(frame, boardSize, corners);

        if (found) {
            // Refine corner positions
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

            // Draw the detected corners on the original frame
            cv::drawChessboardCorners(frame, boardSize, corners, found);

            // Draw lines on a copy of the frame
            cv::Mat frameWithLines = frame.clone();

            // Estimate pose using solvePnP
            cv::Mat rvec, tvec;
            cv::solvePnP(objPoints, corners, camera_matrix, distortion_coefficients, rvec, tvec);

            // Print rotation and translation data
            //std::cout << "Rotation vector:\n" << rvec << std::endl;
            //std::cout << "Translation vector:\n" << tvec << std::endl;

            if (displayAxis) {
                // Draw axes on the checkerboard manually
                cv::Point3f origin(0, 0, 0);
                cv::Point3f xAxis(0.1, 0, 0);
                cv::Point3f yAxis(0, 0.1, 0);
                cv::Point3f zAxis(0, 0, -0.1); // Adjusted z-axis to be visible
                std::vector<cv::Point3f> axesPoints = {origin, xAxis, origin, yAxis, origin, zAxis};
                std::vector<cv::Point2f> imagePoints;
                cv::projectPoints(axesPoints, rvec, tvec, camera_matrix, distortion_coefficients, imagePoints);

                // Draw lines connecting the axes
                cv::line(frameWithLines, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 5); // X-axis (red)
                cv::line(frameWithLines, imagePoints[0], imagePoints[3], cv::Scalar(0, 255, 0), 5); // Y-axis (green)
                cv::line(frameWithLines, imagePoints[0], imagePoints[5], cv::Scalar(255, 0, 0), 5); // Z-axis (blue)
            }

            if (displayPyramid) {
            // Define 3D points for the larger virtual object (a simple pyramid)
            std::vector<cv::Point3f> largeObjectPoints;
            // Increase the size of the pyramid
            float pyramidHeight = 0.5; // Adjust the height of the pyramid as needed
            largeObjectPoints.push_back(cv::Point3f(-0.1, -0.1, 0));
            largeObjectPoints.push_back(cv::Point3f(0.1, -0.1, 0));
            largeObjectPoints.push_back(cv::Point3f(0.1, 0.1, 0));
            largeObjectPoints.push_back(cv::Point3f(-0.1, 0.1, 0));
            largeObjectPoints.push_back(cv::Point3f(0, 0, -pyramidHeight)); // Apex with negative z-coordinate

            // Project 3D object points to image space
            std::vector<cv::Point2f> imagePointsObj;
            cv::projectPoints(largeObjectPoints, rvec, tvec, camera_matrix, distortion_coefficients, imagePointsObj);

            // Draw lines between transformed points for the object
            // Base of the pyramid
            cv::line(frameWithLines, imagePointsObj[0], imagePointsObj[1], cv::Scalar(255, 0, 0), 5);
            cv::line(frameWithLines, imagePointsObj[1], imagePointsObj[2], cv::Scalar(0, 255, 0), 5);
            cv::line(frameWithLines, imagePointsObj[2], imagePointsObj[3], cv::Scalar(0, 0, 255), 5);
            cv::line(frameWithLines, imagePointsObj[3], imagePointsObj[0], cv::Scalar(255, 255, 0), 5);
            // Sides of the pyramid
            cv::line(frameWithLines, imagePointsObj[0], imagePointsObj[4], cv::Scalar(255, 255, 255), 5);
            cv::line(frameWithLines, imagePointsObj[1], imagePointsObj[4], cv::Scalar(255, 255, 255), 5);
            cv::line(frameWithLines, imagePointsObj[2], imagePointsObj[4], cv::Scalar(255, 255, 255), 5);
            cv::line(frameWithLines, imagePointsObj[3], imagePointsObj[4], cv::Scalar(255, 255, 255), 5);
            }


            if (detectHarris) {
                // Perform Harris corner detection
                cv::cornerHarris(gray, gray, blockSize, apertureSize, k);

                // Normalize and apply threshold to obtain corner map
                cv::Mat cornerMap;
                cv::normalize(gray, cornerMap, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::threshold(cornerMap, cornerMap, threshold, 255, cv::THRESH_BINARY);

                // Find corners
                corners.clear();
                for (int i = 0; i < cornerMap.rows; ++i) {
                        for (int j = 0; j < cornerMap.cols; ++j) {
                        if (static_cast<int>(cornerMap.at<uchar>(i, j)) > 0) {
                                corners.push_back(cv::Point(j, i));
                        }
                        }
                }

                // Draw detected corners on original frame
                for (const cv::Point& corner : corners) {
                        cv::circle(frame, corner, 5, cv::Scalar(0, 0, 255), 2);
                }
                }


            // Display frame with lines
            cv::imshow("Frame with Lines", frameWithLines);
        }

        // Display original frame
        cv::imshow("Original Frame", frame);

        // Keyboard input
        char key = cv::waitKey(1);
        if (key == 'q') {
            break; // Exit if 'q' is pressed
        } else if (key == 'x') {
            displayAxis = !displayAxis;
            displayPyramid = false;
             // Toggle display of axis on/off
        } else if (key == 'p') {
            displayPyramid = !displayPyramid; 
            displayAxis = false;// Toggle display of pyramid on/off
        }
        else if (key == 'h') {
            detectHarris = !detectHarris; // Toggle Harris corner detection on/off
        }
    }

    // Release video capture
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
