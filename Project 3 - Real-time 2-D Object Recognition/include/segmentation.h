/*
  Anuj Patel (NUID - 002874710)
  Ruchik Jani (NUID - 002825482)
  Spring 2024
  CS 5330 Computer Vision

  This header file consists declarations of different functions for thresholding, morphological, segmentation and computing region features.  

*/


#ifndef SEGMENTATION_H
#define SEGMENTATION_H


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

using namespace cv;
using namespace std;


/***********************************************************TASK-1****************************************************************************************/

// This function takes an input image and dynamically thresholds it using k-means clustering.
void dynamicThresholdUsingKmeans(const Mat &src, Mat &dst); 

/***********************************************************TASK-1****************************************************************************************/




/***********************************************************TASK-2****************************************************************************************/

//This function performs custom erosion operation on the input image using the provided kernel.
void customErode(const Mat &input, Mat &output, const Mat &kernel); 

/***********************************************************TASK-2****************************************************************************************/




/***********************************************************TASK-3****************************************************************************************/
// Struct representing information about a region.
struct RegionInfo
{
    Point2d centroid;
    Vec3b color;

    RegionInfo(Point2d c, Vec3b col) : centroid(c), color(col) {}
};

extern std::vector<RegionInfo> previousRegions;

// Calculate the dynamic threshold value based on the centroids of the regions.
double calculateDynamicThreshold(const vector<Point2d> &currentCentroids);

/***********************************************************TASK-3****************************************************************************************/

/***********************************************************TASK-4****************************************************************************************/

// Compute and draw region features for a specific region.
void computeAndDrawRegionFeatures(Mat &output, const Mat &labels, int regionID, const Vec3b &color, double &percentFilled, double &bboxRatio, double &orientationAngle, double &area);


/***********************************************************TASK-4****************************************************************************************/


#endif // SEGMENTATION_H