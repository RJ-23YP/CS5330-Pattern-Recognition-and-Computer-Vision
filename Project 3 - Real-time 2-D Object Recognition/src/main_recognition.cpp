/*
  Ruchik Jani (NUID - 002825482)
  Anuj Patel (NUID - 002874710)
  Spring 2024
  CS 5330 Computer Vision

  This program is the main source code where we are implementing object recognition using different classifiers as per user selection. 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <segmentation.h>
#include <classification.h>

using namespace cv;
using namespace std;


/**
 * brief: Segment the input image based on connected components and draw region features.
 * 
 * This function segments the input image based on connected components, computes region features, and draws them on the output image.
 * 
 * param input The input image to be segmented.
 * param output The segmented output image with region features drawn.
 * param numberOfLargestRegions The number of largest regions to consider.
 * param minSizeThreshold The minimum size threshold for regions to be considered significant.
 * return None.
 */


void segmentImage(const Mat &input, Mat &output, int numberOfLargestRegions, int minSizeThreshold = 100)
{
    // Perform Otsu thresholding on the input image
    Mat labels, stats, centroidsMat;
    Mat thresholded;
    threshold(input, thresholded, 0, 255, THRESH_BINARY | THRESH_OTSU);
    // Find connected components in the thresholded image
    int nLabels = connectedComponentsWithStats(thresholded, labels, stats, centroidsMat);

    // Convert centroidsMat to a vector of Point2d
    vector<Point2d> centroids;
    for (int i = 0; i < centroidsMat.rows; i++)
    {
        centroids.push_back(Point2d(centroidsMat.at<double>(i, 0), centroidsMat.at<double>(i, 1)));
    }

    // Calculate current dynamic threshold value based on centroids
    double currentDynamicThreshold = calculateDynamicThreshold(centroids);

    // Initialize vector to store colors for each label
    vector<Vec3b> colors(nLabels, Vec3b(0, 0, 0));
    // Initialize vector to store new regions
    vector<RegionInfo> newRegions;

    // Iterate over each label (region)
    for (size_t i = 1; i < centroids.size(); ++i)
    {
        double minDistance = std::numeric_limits<double>::max();
        int matchedIndex = -1;

        // Iterate over each previous region to find matching region
        for (size_t j = 0; j < previousRegions.size(); ++j)
        {
            // Calculate Euclidean distance between centroids
            double distance = norm(centroids[i] - previousRegions[j].centroid);
            // Update minimum distance and matched index if necessary
            if (distance < minDistance)
            {
                minDistance = distance;
                matchedIndex = j;
            }
        }

        // Check if a matching region is found and distance is within dynamic threshold
        if (matchedIndex != -1 && minDistance < currentDynamicThreshold)
        {
            // Use color of matched region
            colors[i] = previousRegions[matchedIndex].color;
            // Update centroid of matched region and add to newRegions
            previousRegions[matchedIndex].centroid = centroids[i];
            newRegions.push_back(previousRegions[matchedIndex]);
        }
        else
        {
            // Generate new color for unmatched region
            Vec3b newColor = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
            colors[i] = newColor;
            // Add new region to newRegions
            newRegions.emplace_back(centroids[i], newColor);
        }
    }

    // Initialize output image with zeros
    output = Mat::zeros(input.size(), CV_8UC3);
    // Draw regions on output image using assigned colors
    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            int label = labels.at<int>(i, j);
            if (label > 0)
            {
                output.at<Vec3b>(i, j) = colors[label];
            }
        }
    }

    // Update previousRegions with new regions
    previousRegions = std::move(newRegions);
}


/***********************************************************MAIN FUNCTION****************************************************************************************/
int main()
{

    string streamURL = "http://10.110.10.186:4747/video"; // Enter the IP address of your webcam stream

    VideoCapture cap(streamURL);

    if (!cap.isOpened())
    {
        cerr << "Error: Failed to open video capture device or file" << endl;
        return -1;
    }

    char choice;
    cout << "Choose approach: (K)NN or (N)earest Neighbor: ";
    cin >> choice;

    int k = 1;
    if (choice == 'K' || choice == 'k')
    {
        cout << "Enter the value of K for KNN: ";
        cin >> k;
    }

    Mat frame, processedFrame, erodedFrame, segmentedFrame, output, finalOutput, regionFrame;
    int kernelSize = 5;
    Mat kernel = Mat::ones(Size(kernelSize, kernelSize), CV_8U) * 255;

    int numberOfLargestRegions = 1;
    int minSizeThreshold;
    const double SOME_DISTANCE_THRESHOLD = 50.0;

    while (true)
    {
        // Capture frame from the video stream
        Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            cerr << "Error: Failed to capture frame" << endl;
            break;
        }

        // Preprocess the frame
        dynamicThresholdUsingKmeans(frame, processedFrame);
        customErode(processedFrame, erodedFrame, kernel);

        // Segment the frame
        segmentImage(erodedFrame, segmentedFrame, numberOfLargestRegions, minSizeThreshold);
        segmentImage(erodedFrame, segmentedFrame, minSizeThreshold);

        // Perform connected component analysis on the segmented frame
        Mat labels, stats, centroids;
        Mat graySegmented;
        if (segmentedFrame.channels() == 3)
        {
            cvtColor(segmentedFrame, graySegmented, COLOR_BGR2GRAY);
        }
        else
        {
            graySegmented = segmentedFrame;
        }
        threshold(graySegmented, graySegmented, 0, 255, THRESH_BINARY | THRESH_OTSU);
        int nLabels = connectedComponentsWithStats(graySegmented, labels, stats, centroids, 8, CV_32S);

        // Read known object features from the database
        vector<ObjectFeature> knownObjects = readObjectFeatures("/home/rj/Project3 (OR)/data/prof_data/training_set/objectDB.csv");

        // Draw bounding boxes and classify objects
        output = segmentedFrame.clone();
        for (int i = 1; i < nLabels; i++)
        {
            // Random color for each object
            Vec3b color(rand() & 255, rand() & 255, rand() & 255);

            // Compute region features and draw bounding box
            double percentFilled, bboxRatio, orientationAngle;
            int areaInt = stats.at<int>(i, CC_STAT_AREA);
            double area = static_cast<double>(areaInt);
            if (area > 500)
            {
                computeAndDrawRegionFeatures(output, labels, i, color, percentFilled, bboxRatio, orientationAngle, area);
                
                // Create new object feature and classify using chosen approach
                ObjectFeature newObj = {area, percentFilled, bboxRatio, orientationAngle, ""};
                string label;
                if (choice == 'K' || choice == 'k')
                {
                    label = classifyObjectKNN(newObj, knownObjects, k);
                }
                else
                {
                    label = classifyObject(newObj, knownObjects);
                }

                // Draw label
                Point2f centroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
                Point2f labelPosition(centroid.x + 50, centroid.y + 50);
                putText(output, label, labelPosition, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(color), 1);
            }
        }

        // Display frames
        imshow("Original Frame", frame);
        imshow("Processed Frame", processedFrame);
        imshow("Eroded Frame", erodedFrame);
        imshow("Segmented Frame", segmentedFrame);
        imshow("Classified Object", output);

        // Exit loop if 'Esc' is pressed
        if (waitKey(1) == 27)
        {
            break;
        }
    }

    // Release video capture and close windows
    cap.release();
    destroyAllWindows();

    return 0;
}

/***********************************************************MAIN FUNCTION****************************************************************************************/
