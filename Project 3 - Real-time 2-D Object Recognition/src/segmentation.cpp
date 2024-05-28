/*
  Ruchik Jani (NUID - 002825482)
  Anuj Patel (NUID - 002874710)
  Spring 2024
  CS 5330 Computer Vision

  This program consists of function definitions for thresholding, morphological, segmentation and region feature computing functions.  
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

using namespace cv;
using namespace std;


/***********************************************************TASK-1****************************************************************************************/

/**
 * brief: Perform dynamic thresholding using k-means clustering.
 * 
 * This function takes an input image and dynamically thresholds it using k-means clustering.
 * 
 * param src The source input image.
 * param dst The output thresholded image.
 * return None.
 */

void dynamicThresholdUsingKmeans(const Mat &src, Mat &dst) {
    // First, apply Gaussian blur to the source image
    Mat blurred;
    GaussianBlur(src, blurred, Size(5, 5), 0);

    Mat gray;
    if (blurred.channels() == 3) {
        // Convert the blurred image to grayscale
        gray = Mat(blurred.rows, blurred.cols, CV_8UC1);
        for (int i = 0; i < blurred.rows; ++i) {
            for (int j = 0; j < blurred.cols; ++j) {
                Vec3b intensity = blurred.at<Vec3b>(i, j);
                gray.at<uchar>(i, j) = static_cast<uchar>((intensity[0] + intensity[1] + intensity[2]) / 3);
            }
        }
    } else {
        gray = blurred.clone();
    }

    // Proceed with the rest of your function to dynamically threshold the image using k-means
    Mat data;
    gray.convertTo(data, CV_32F);
    data = data.reshape(1, src.total());

    int K = 2;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0);
    Mat bestLabels, centers;
    kmeans(data, K, bestLabels, criteria, 3, KMEANS_PP_CENTERS, centers);

    double thresholdValue = (centers.at<float>(0, 0) + centers.at<float>(1, 0)) / 2;

    dst = Mat(gray.size(), CV_8UC1);
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            if (gray.at<uchar>(i, j) > thresholdValue)
                dst.at<uchar>(i, j) = 0;
            else
                dst.at<uchar>(i, j) = 255;
        }
    }

    cout << "Threshold Value Used: " << thresholdValue << endl;
}

/***********************************************************TASK-1****************************************************************************************/




/***********************************************************TASK-2****************************************************************************************/

/**
 * brief: Custom erosion function.
 * 
 * This function performs custom erosion operation on the input image using the provided kernel.
 * 
 * param input The input image to be eroded.
 * param output The output eroded image.
 * param kernel The kernel used for erosion.
 * return None.
 */
void customErode(const Mat &input, Mat &output, const Mat &kernel)

{
    // Initialize the output image with zeros
    output = Mat::zeros(input.size(), input.type());

    // Compute the anchor point for the kernel
    Point anchor = Point(kernel.rows / 2, kernel.cols / 2);

    // Iterate over each pixel in the input image
    for (int x = anchor.x; x < input.rows - anchor.x; ++x)
    {
        for (int y = anchor.y; y < input.cols - anchor.y; ++y)
        {
            // Define the region of interest (ROI) for the current pixel
            Rect roi = Rect(y - anchor.y, x - anchor.x, kernel.cols, kernel.rows);
            Mat imageROI = input(roi);
            bool shouldErode = true;

            // Iterate over the kernel and check if erosion should be applied
            for (int i = 0; i < kernel.rows && shouldErode; ++i)
            {
                for (int j = 0; j < kernel.cols && shouldErode; ++j)
                {
                    // Check if the kernel element is foreground and input pixel is background
                    if (kernel.at<uchar>(i, j) == 255 && imageROI.at<uchar>(i, j) != 255)
                    {
                        shouldErode = false;
                    }
                }
            }

            // Update the output pixel based on erosion condition
            if (shouldErode)
            {
                output.at<uchar>(x, y) = 255;
            }
            else
            {
                output.at<uchar>(x, y) = 0;
            }
        }
    }
}

/***********************************************************TASK-2****************************************************************************************/




/***********************************************************TASK-3****************************************************************************************/

// Vector storing information about previously detected regions.
vector<RegionInfo> previousRegions;
double dynamicThreshold = 115;

/**
 * brief: Calculate the dynamic threshold value based on the centroids of the regions.
 * 
 * This function calculates the dynamic threshold value for segmenting regions based on the centroids of the regions and previous region centroids.
 * 
 * param currentCentroids The centroids of the current regions.
 * return The calculated dynamic threshold value.
 */

double calculateDynamicThreshold(const vector<Point2d> &currentCentroids)
{
    // Check if previous regions or current centroids are empty, return the default dynamic threshold value
    if (previousRegions.empty() || currentCentroids.empty())
        return dynamicThreshold;

    // Initialize a vector to store distances between current and previous centroids
    vector<double> distances;

    // Iterate over each current centroid
    for (const auto &currCentroid : currentCentroids)
    {
        // Initialize minimum distance to maximum possible value
        double minDistance = numeric_limits<double>::max();

        // Iterate over each previous region
        for (const auto &prevRegion : previousRegions)
        {
            // Calculate Euclidean distance between current centroid and previous region centroid
            double distance = norm(currCentroid - prevRegion.centroid);
            // Update minimum distance if necessary
            if (distance < minDistance)
            {
                minDistance = distance;
            }
        }
        // Store the minimum distance for the current centroid
        distances.push_back(minDistance);
    }

    // Calculate average distance between current and previous centroids
    if (!distances.empty())
    {
        double avgDistance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
        // Update dynamic threshold as the maximum of average distance and a minimum threshold
        dynamicThreshold = max(150.00, avgDistance);
    }

    // Return the calculated dynamic threshold value
    return dynamicThreshold;
}

/***********************************************************TASK-3****************************************************************************************/




/***********************************************************TASK-4****************************************************************************************/

/**
 * brief: Compute and draw region features for a specific region.
 * 
 * This function computes and draws region features such as centroid, orientation, bounding box, etc., for a specific region.
 * 
 * param output The output image where the region features will be drawn.
 * param labels The label image containing the region labels.
 * param regionID The ID of the region for which features are computed.
 * param color The color used to draw the region features.
 * param percentFilled The percentage of area filled by the region.
 * param bboxRatio The aspect ratio of the bounding box enclosing the region.
 * param orientationAngle The orientation angle of the region.
 * param area The area of the region.
 * return None.
 */
void computeAndDrawRegionFeatures(Mat &output, const Mat &labels, int regionID, const Vec3b &color, double &percentFilled, double &bboxRatio, double &orientationAngle, double &area)
{
    // Extract the binary mask of the region based on its label ID
    Mat region = labels == regionID;
    // Compute moments of the region
    Moments m = moments(region, true);

    // Compute area and centroid of the region
    area = m.m00;
    Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);

    // Extract non-zero points within the region
    vector<Point> points;
    findNonZero(region, points);
    // Compute the minimum area bounding box (oriented bounding box) of the region
    RotatedRect obb = minAreaRect(points);

    // Compute the bounding box of the region
    Rect bbox = boundingRect(points);

    // Compute the percentage of area filled by the region
    percentFilled = area / static_cast<double>(bbox.width * bbox.height);

    // Compute the aspect ratio of the bounding box
    bboxRatio = static_cast<double>(bbox.height) / bbox.width;

    // Compute the orientation angle of the region
    double mu11 = m.mu11;
    double mu20 = m.mu20;
    double mu02 = m.mu02;
    orientationAngle = 0.5 * atan2(2 * mu11, mu20 - mu02);

    // Draw the rotated bounding box
    Point2f vertices[4];
    obb.points(vertices);
    for (int i = 0; i < 4; i++)
        line(output, vertices[i], vertices[(i + 1) % 4], Scalar(color), 2);

    // Draw the orientation line
    Point2f end_point(centroid.x + cos(orientationAngle) * 50, centroid.y + sin(orientationAngle) * 50);
    line(output, centroid, end_point, Scalar(color), 2);

    // Display region ID, fill percentage, and aspect ratio as text
    string featureText = format("ID: %d, Fill: %.2f, Ratio: %.2f", regionID, percentFilled, bboxRatio);
    putText(output, featureText, centroid, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(color), 1);
}

/***********************************************************TASK-4****************************************************************************************/
