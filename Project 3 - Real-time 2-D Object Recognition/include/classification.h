/*
  Anuj Patel (NUID - 002874710)
  Ruchik Jani (NUID - 002825482)
  Spring 2024
  CS 5330 Computer Vision

  This header file consists declarations of different functions for Nearest Neighbour and K-Nearest Neighbour classifiers.
*/


#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

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



/***********************************************************TASK-6****************************************************************************************/


struct ObjectFeature  // Structure defining features of an object.
{
    double area;
    double percentFilled;
    double bboxRatio;
    double orientationAngle;
    string label;
};

vector<ObjectFeature> readObjectFeatures(const string &filename); // Read object features from a file.

double calculateDistance(const ObjectFeature &obj1, const ObjectFeature &obj2); // Calculate Euclidean distance between two objects.

string classifyObject(const ObjectFeature &newObj, const vector<ObjectFeature> &database); // Classify object using nearest neighbor.

/***********************************************************TASK-6****************************************************************************************/



/***********************************************************TASK-9****************************************************************************************/

// Function declaration for classifying objects using the K-Nearest Neighbors (KNN) approach.

string classifyObjectKNN(const ObjectFeature &newObj, const vector<ObjectFeature> &database, int k); // Classify object using KNN.

/***********************************************************TASK-9****************************************************************************************/


#endif // CLASSIFICATION_H