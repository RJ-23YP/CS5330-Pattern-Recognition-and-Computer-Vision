/*
  Anuj Patel (NUID - 002874710)
  Ruchik Jani (NUID - 002825482)
  Spring 2024
  CS 5330 Computer Vision

  This header file consists declarations of different functions which contain different image matching methods that will be applied when called from image_retrieval.cpp & computes.cpp file. 
*/

#ifndef COMPUTES_H
#define COMPUTES_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <map>
#include <algorithm>

using namespace cv;
using namespace std;
namespace fs = filesystem;


/***********************************************************TASK-1****************************************************************************************/
// Function: computeBaselineFeatures
Mat computeBaselineFeatures(const Mat &image);

// Function: computeDistance
double computeDistance(const Mat &feature1, const Mat &feature2);

/***********************************************************TASK-1****************************************************************************************/




/***********************************************************TASK-2****************************************************************************************/

// Structure: ImageDistance_2
// Summary: Represents an image distance entity containing the image path (filename and path) and the distance.
struct ImageDistance_2
{
    string imagePath; // Using imagePath for both filename and image path
    double distance;
};

// Function: computeRGBHistogram
cv::Mat computeRGBHistogram(const cv::Mat &image);

// Function: histogramDistance
double histogramDistance(const cv::Mat &hist1, const cv::Mat &hist2);

/***********************************************************TASK-2****************************************************************************************/




/***********************************************************TASK-3****************************************************************************************/

// Structure: ImageDistance_3
// Summary: Represents an image distance entity containing the image path and the distance.
struct ImageDistance_3
{
    string imagePath; // Path of the image
    double distance;       // Distance of the image from a target image

    // Constructor
    ImageDistance_3(string path = "", double dist = 0.0) : imagePath(move(path)), distance(dist) {}
};

// Function: computeHistogram_3
cv::Mat computeHistogram_3(const cv::Mat &image, int bins = 8, const cv::Rect &roi = cv::Rect());

// Function: histogramIntersection
double histogramIntersection(const cv::Mat &hist1, const cv::Mat &hist2);

// Function: findSimilarImages_3
vector<ImageDistance_3> findSimilarImages_3(const string &targetImagePath, const string &directoryPath, int topN);

/***********************************************************TASK-3****************************************************************************************/




/***********************************************************TASK-4****************************************************************************************/
// Structure to hold image path and distance FOR TASK 4
struct ImageDistance
{
    string imagePath; // Path of the image
    double distance;       // Distance of the image from a target image
};

// Function to compute the color histogram TASK 4
cv::Mat computeColorHistogram(const cv::Mat &image);

// Function to compute texture features (magnitude and orientation histograms) TASK 4
pair<cv::Mat, cv::Mat> computeTextureFeatures(const cv::Mat &image);

// Function to compare histogram TASK 4
double compareHistograms(const cv::Mat &hist1, const cv::Mat &hist2);

// Function: findSimilarImages
vector<ImageDistance> findSimilarImages(const string &targetImagePath, const string &directoryPath, int N);

/***********************************************************TASK-4****************************************************************************************/




/***********************************************************TASK-5****************************************************************************************/
// Function: split
vector<string> split(const string &s, char delimiter);

// Function: loadFeatures
map<string, Eigen::VectorXd> loadFeatures(const string &filename);

// Function: cosineDistance
double cosineDistance(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2);

// Function: findSimilarImagesDNN
vector<pair<string, double>> findSimilarImagesDNN(const map<string, Eigen::VectorXd> &features, const string &targetImagePath, int N);

/***********************************************************TASK-5****************************************************************************************/




/***********************************************************TASK-6****************************************************************************************/
// Function: computeHistogram
cv::Mat computeHistogram(const cv::Mat &image);

// Function: computeLBP
cv::Mat computeLBP(const cv::Mat &src);

// Function: featureDistance
double featureDistance(const cv::Mat &hist1, const cv::Mat &hist2);

// Function: findSimilarImagesClassic
vector<pair<string, double>> findSimilarImagesClassic(const string &targetImagePath, const string &directoryPath, int N = 4);
/***********************************************************TASK-6****************************************************************************************/



/***********************************************************TASK-7****************************************************************************************/
// Structure to hold image features including color histogram, texture, and shape
struct ImageFeature_7 {
    string filePath;       // Path to the image file
    cv::Mat histogram;     // Color histogram feature
    cv::Mat texture;       // Texture feature (LBP)
    cv::Mat shape;         // Shape feature (Hu Moments)
};

// Function to calculate color histogram of an image
cv::Mat calculateHistogram(const cv::Mat& image);

// Function to calculate Euclidean distance between two histograms
double calculateEuclideanDistance(const cv::Mat& hist1, const cv::Mat& hist2);

// Function to calculate Local Binary Patterns (LBP) of an image
cv::Mat calculateLBP(const cv::Mat& src);

// Function to calculate Hu Moments of an image
cv::Mat calculateHuMoments(const cv::Mat& src);

// Function to compute features for all images in a directory
vector<ImageFeature_7> computeFeaturesForDirectory(const string& directoryPath);

// Function to calculate weighted distance between two image features
double calculateWeightedDistance(const ImageFeature_7& f1, const ImageFeature_7& f2);

// Function to find similar images to a target image
vector<pair<string, double>> findSimilarImages(const string& targetImagePath, const vector<ImageFeature_7>& features);

/***********************************************************TASK-7****************************************************************************************/


#endif // COMPUTES_H
