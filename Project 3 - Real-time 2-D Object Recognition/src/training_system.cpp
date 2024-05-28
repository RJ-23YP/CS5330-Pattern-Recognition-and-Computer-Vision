/*
  Ruchik Jani (NUID - 002825482)
  Anuj Patel (NUID - 002874710)
  Spring 2024
  CS 5330 Computer Vision

  This program is a seperate training system code for task-5 to compute feature vectors and store them in a database along with labels inputted by the user. 
*/



#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/ml.hpp> // Include OpenCV machine learning header for k-means
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem; // Alias for the filesystem namespace



/***********************************************************TASK-5****************************************************************************************/


// Dynamic Thresholding Using Kmeans
void dynamicThresholdUsingKmeans(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src.clone();

    cv::Mat data;
    gray.convertTo(data, CV_32F);
    data = data.reshape(1, src.total());

    int K = 2;
    cv::Mat labels, centers;
    cv::kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    double thresholdValue = (centers.at<float>(0, 0) + centers.at<float>(1, 0)) / 2;
    cout << "Threshold Value Used: " << thresholdValue << endl;

    cv::threshold(gray, dst, thresholdValue, 255, cv::THRESH_BINARY_INV);
}

// Function to perform erosion
void customErode(const Mat &input, Mat &output, const Mat &kernel) {
    output = Mat::zeros(input.size(), input.type());
    Point anchor = Point(kernel.rows / 2, kernel.cols / 2);

    for (int x = anchor.x; x < input.rows - anchor.x; ++x) {
        for (int y = anchor.y; y < input.cols - anchor.y; ++y) {
            Rect roi = Rect(y - anchor.y, x - anchor.x, kernel.cols, kernel.rows);
            Mat imageROI = input(roi);
            bool shouldErode = true;

            for (int i = 0; i < kernel.rows && shouldErode; ++i) {
                for (int j = 0; j < kernel.cols && shouldErode; ++j) {
                    if (kernel.at<uchar>(i, j) == 255 && imageROI.at<uchar>(i, j) != 255) {
                        shouldErode = false;
                    }
                }
            }

            if (shouldErode) {
                output.at<uchar>(x, y) = 255;
            } else {
                output.at<uchar>(x, y) = 0;
            }
        }
    }
}


// Struct representing information about a region.
struct RegionInfo {
    Point2d centroid;
    Vec3b color;

    RegionInfo(Point2d c, Vec3b col) : centroid(c), color(col) {}
};

// Vector storing information about previously detected regions.
vector<RegionInfo> previousRegions;

const double SOME_DISTANCE_THRESHOLD = 50.0;

// Segment the input image based on connected components and draw region features.
void segmentImage(const Mat &input, Mat &output, int numberOfLargestRegions, int minSizeThreshold = 5000) {
    Mat labels, stats, centroidsMat;
    Mat thresholded; 
    threshold(input, thresholded , 0, 255, THRESH_BINARY | THRESH_OTSU);
    int nLabels = connectedComponentsWithStats(input, labels, stats, centroidsMat);

    // Convert centroids to a more usable form
    vector<Point2d> centroids;
    for(int i = 0; i < centroidsMat.rows; i++) {
        centroids.push_back(Point2d(centroidsMat.at<double>(i, 0), centroidsMat.at<double>(i, 1)));
    }

    // Assign colors based on matching with previous frame's regions
    vector<Vec3b> colors(nLabels, Vec3b(0,0,0)); // Default to black
    for (size_t i = 1; i < centroids.size(); ++i) {
        double minDistance = std::numeric_limits<double>::max();
        int matchedIndex = -1;

        for (size_t j = 0; j < previousRegions.size(); ++j) {
            double distance = norm(centroids[i] - previousRegions[j].centroid);
            if (distance < minDistance) {
                minDistance = distance;
                matchedIndex = j;
            }
        }

        if (matchedIndex != -1 && minDistance < SOME_DISTANCE_THRESHOLD) {
            // If a match is found and is within a reasonable distance,
            // use the matched region's color
            colors[i] = previousRegions[matchedIndex].color;
        } else {
            // Otherwise, assign a new random color
            colors[i] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
        }
    }

    // Update the output image and previousRegions for the next frame
    output = Mat::zeros(input.rows, input.cols, CV_8UC3);
    previousRegions.clear();
    for (int r = 0; r < input.rows; r++) {
        for (int c = 0; c < input.cols; c++) {
            int label = labels.at<int>(r, c);
            if (label > 0) { // Ignore background
                output.at<Vec3b>(r, c) = colors[label];
                if (std::find_if(previousRegions.begin(), previousRegions.end(), 
                    [&](const RegionInfo& ri){ return ri.centroid == centroids[label]; }) == previousRegions.end()) {
                    previousRegions.push_back(RegionInfo(centroids[label], colors[label]));
                }
            }
        }
    }
}


// Compute and draw region features for a specific region.

void computeAndDrawRegionFeatures(Mat& output, const Mat& labels, const Mat& stats, const Mat& centroids, int regionID, const Vec3b& color) {
    // Directly extract the region using the region ID
    Mat region = labels == regionID;

    // Calculate moments
    Moments m = moments(region, true);

    // Calculate the area, centroid, and other features
    double area = m.m00;
    Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);

    // Calculate the oriented bounding box
    vector<Point> points;
    findNonZero(region, points);
    RotatedRect obb = minAreaRect(points);

    // Calculate percent filled and bounding box height/width ratio
    Rect bbox = boundingRect(points);
    double percentFilled = area / (bbox.width * bbox.height);
    double bboxRatio = static_cast<double>(bbox.height) / bbox.width;

    // Calculate the angle of orientation (axis of least central moment)
    double mu11 = m.mu11;
    double mu20 = m.mu20;
    double mu02 = m.mu02;
    double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);

    // Draw the oriented bounding box
    Point2f vertices[4];
    obb.points(vertices);
    for (int i = 0; i < 4; i++)
        line(output, vertices[i], vertices[(i+1)%4], Scalar(color), 2);

    // Draw the axis of least central moment
    Point2f end_point(centroid.x + cos(theta) * 50, centroid.y + sin(theta) * 50); // Adjust length of axis
    line(output, centroid, end_point, Scalar(color), 2);

    // Display features as text
    string featureText = format("ID: %d, Fill: %.2f, Ratio: %.2f", regionID, percentFilled, bboxRatio);
    putText(output, featureText, centroid, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(color), 1);
}



// Define a struct to store region features
struct RegionFeatures {
    //int regionID;
    double percentFilled;
    double bboxRatio;
    double orientationAngle;
    double area;
    string label; // New member to store the label
};



// Function to store region features in a CSV file
    // Arguments:
    // - filename: The name of the CSV file to store the features.
    // - features: A vector containing the region features to be stored.
    // Return: None.
void storeRegionFeatures(const string& filename, vector<RegionFeatures>& features) {
   

    ofstream outputFile(filename, ios::app); // Open file in append mode

    // Write header
    //outputFile << "Area, PercentFilled, BoundingBoxRatio, OrientationAngle, Label " << endl;

    // Write data
    for (const auto& feature : features) {
        outputFile << feature.area << ", " << feature.percentFilled << ", "
                   << feature.bboxRatio << ",     " << feature.orientationAngle << ",     "
                   << feature.label << endl; 
    }

    outputFile.close();
}

// Function to extract region features from connected components and user input
    // Arguments:
    // - finalOutput: The final output image with region features drawn.
    // - labels: The labeled image containing connected components.
    // - stats: Statistics of connected components.
    // - centroids: Centroids of connected components.
    // - nLabels: The number of labels.
    // Return: A vector containing the extracted region features.
vector<RegionFeatures> extractRegionFeatures(Mat& finalOutput, const Mat& labels, const Mat& stats, const Mat& centroids, int nLabels) {
   

    vector<RegionFeatures> regionFeatures;

    // Find the index of the largest region
    int largestRegionIndex = -1;
    double maxArea = -1.0;
    for (int i = 1; i < nLabels; i++) { // Start from 1 to skip the background
        double area = stats.at<int>(i, CC_STAT_AREA);
        if (area > maxArea) {
            maxArea = area;
            largestRegionIndex = i;
        }
    }

    // If no regions are found, return empty vector
    if (largestRegionIndex == -1) {
        return regionFeatures;
    }

    // Extract features of the largest region
    Vec3b color(rand() & 255, rand() & 255, rand() & 255); // Random color for drawing
    RegionFeatures features;

    // Call computeAndDrawRegionFeatures() to extract features
    computeAndDrawRegionFeatures(finalOutput, labels, stats, centroids, largestRegionIndex, color);

    // Extract features from the finalOutput image
    Mat region = labels == largestRegionIndex;
    Moments m = moments(region, true);
    double area = m.m00;
    features.area = area; 
    Rect bbox = boundingRect(region);
    features.percentFilled = area / (bbox.width * bbox.height);
    features.bboxRatio = static_cast<double>(bbox.height) / bbox.width;
    double mu11 = m.mu11;
    double mu20 = m.mu20;
    double mu02 = m.mu02;
    features.orientationAngle = 0.5 * atan2(2 * mu11, mu20 - mu02);

    // Prompt user for label
    cout << "Enter label for the largest region: ";
    string label;
    cin >> label;
    features.label = label;

    // Store the extracted features
    regionFeatures.push_back(features);

    return regionFeatures;
}

// Function to display the contents of a CSV file
    // Arguments:
    // - filename: The name of the CSV file to be displayed.
    // Return: None.
void displayCSVContents(const string& filename) {
   

    ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return;
    }

    string line;
    while (getline(inputFile, line)) {
        stringstream ss(line);
        string field;

        while (getline(ss, field, ',')) {
            cout << field << "\t";
        }
        cout << endl;
    }

    inputFile.close();
}


int main() {
    
    const string objectDBFile = "/home/rj/Project3 (OR)/data/prof_data/training_set/objectDB.csv";
    const string trainingSetFolder = "/home/rj/Project3 (OR)/data/prof_data/training_set";
    
    cout << "Press 'N' to add a new object to the database. Press 'Q' to quit." << endl;

    ofstream outputFile(objectDBFile, ios::app); // Open file in append mode

    // Write header
    outputFile << "Area, PercentFilled, BoundingBoxRatio, OrientationAngle, Label " << endl;

    outputFile.close();

    cout << "Contents of " << objectDBFile << ":" << endl;
    displayCSVContents(objectDBFile);

    namedWindow("Original Frame", WINDOW_AUTOSIZE);
    //namedWindow("Processed Frame", WINDOW_AUTOSIZE);
    //namedWindow("Eroded Frame", WINDOW_AUTOSIZE);
    //namedWindow("Segmented Frame", WINDOW_AUTOSIZE); // Window for displaying segmented regions.
    //namedWindow("Final Output", WINDOW_AUTOSIZE); 
    
    // Resize windows to fit within the screen
    resizeWindow("Original Frame", 800, 600);
  /*resizeWindow("Processed Frame", 800, 600);
    resizeWindow("Eroded Frame", 800, 600);
    resizeWindow("Segmented Frame", 800, 600);
    resizeWindow("Final Output", 800, 600);*/

    Mat image, resizedImage, processedFrame, erodedFrame, dilatedFrame, segmentedFrame, finalOutput;
    int kernelSize = 5;
    Mat kernel = Mat::ones(Size(kernelSize, kernelSize), CV_8U) * 255;

    int numberOfLargestRegions = 1; // Example: limit to the 3 largest regions. Set to 0 to include all significant regions.
    int minSizeThreshold; // Minimum size for regions to be considered significant.
    const double SOME_DISTANCE_THRESHOLD = 50.0; // Example distance threshold for matching centroids

    fs::path trainingSetPath(trainingSetFolder);

    for (const auto& entry : fs::directory_iterator(trainingSetPath)) {
        string imagePath = entry.path().string();
        string imageName = entry.path().filename().string();

        cout << "Processing image: " << imageName << endl;

        image = imread(imagePath);
        if (image.empty()) {
            cerr << "Could not read the image from: " << imagePath << endl;
            continue;
        }

        // Display the image
        //namedWindow("Image", WINDOW_NORMAL);
        resize(image, resizedImage, Size(800, 600));
        imshow("Original Frame", resizedImage);
        waitKey(0);
        destroyAllWindows();

        // Apply dynamic thresholding and erosion to the captured frame.
        dynamicThresholdUsingKmeans(resizedImage, processedFrame);
        customErode(processedFrame, erodedFrame, kernel);
        //customDilate(erodedFrame, dilatedFrame, kernel);

        // Segment the dilated frame, focusing on the largest N significant regions,
        // and attempt to maintain color consistency across frames.
        segmentImage(erodedFrame, segmentedFrame, numberOfLargestRegions, minSizeThreshold);

        // For Task 4, we first need to find the connected components in the segmented frame
        Mat labels, stats, centroids;
        // Ensure segmentedFrame is suitable for connectedComponentsWithStats (might need conversion to grayscale)
        Mat graySegmented;
        if (segmentedFrame.channels() == 3) {
            cvtColor(segmentedFrame, graySegmented, COLOR_BGR2GRAY);
        } else {
            graySegmented = segmentedFrame;
        }
        threshold(graySegmented, graySegmented, 0, 255, THRESH_BINARY | THRESH_OTSU);
        int nLabels = connectedComponentsWithStats(graySegmented, labels, stats, centroids, 8, CV_32S);

        // Copy segmentedFrame to finalOutput for drawing
        finalOutput = segmentedFrame.clone();

        // Task 4: Compute and draw region features for each identified region
        for (int i = 1; i < nLabels; i++) { // Start from 1 to skip the background
            Vec3b color(rand() & 255, rand() & 255, rand() & 255); // Random color for drawing
            computeAndDrawRegionFeatures(finalOutput, labels, stats, centroids, i, color);
        }

        //imshow("Processed Frame", processedFrame);
        //imshow("Eroded Frame", erodedFrame);
        //imshow("Segmented Frame", segmentedFrame); // Display the segmented image.
        //imshow("Final Output", finalOutput);

        waitKey(0);
        destroyAllWindows();

        //Implementing the code to extract the above features and store them in a csv file. 
        // Extract region features
        vector<RegionFeatures> regionFeatures = extractRegionFeatures(finalOutput, labels, stats, centroids, nLabels);

        // Store the region features in a CSV file
        storeRegionFeatures(objectDBFile, regionFeatures);

        cout << "Object data stored successfully." << endl;
    }
    
    cout << "Finished processing all images in the training set folder." << endl;
    return 0;
}


/***********************************************************TASK-5****************************************************************************************/
