/*
  Ruchik Jani (NUID - 002825482)
  Anuj Patel (NUID - 002874710)
  Spring 2024
  CS 5330 Computer Vision

  This program consists of function definitions for NN, KNN classifier functions.  
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
#include <classification.h>


using namespace cv;
using namespace std;


/***********************************************************TASK-6****************************************************************************************/

/**
 * Read object features from a CSV file.
 * 
 * param filename The path to the CSV file containing object features.
 * return A vector of ObjectFeature objects read from the file.
 */
vector<ObjectFeature> readObjectFeatures(const string &filename)
{
    vector<ObjectFeature> features;
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return features;
    }

    // Read features from each line of the file
    string line;
    getline(file, line); // Skip header line
    while (getline(file, line))
    {
        stringstream ss(line);
        ObjectFeature feature;

        // Parse each feature from the comma-separated values
        getline(ss, line, ',');
        feature.area = stod(line);
        getline(ss, line, ',');
        feature.percentFilled = stod(line);
        getline(ss, line, ',');
        feature.bboxRatio = stod(line);
        getline(ss, line, ',');
        feature.orientationAngle = stod(line);
        getline(ss, feature.label, ',');
        features.push_back(feature);
    }
    file.close();
    return features;
}

/**
 * Calculate the Euclidean distance between two ObjectFeature objects.
 * 
 * param obj1 The first ObjectFeature object.
 * param obj2 The second ObjectFeature object.
 * return The Euclidean distance between the two objects.
 */
double calculateDistance(const ObjectFeature &obj1, const ObjectFeature &obj2)
{
    return sqrt(pow(obj1.percentFilled - obj2.percentFilled, 2) +
                pow(obj1.bboxRatio - obj2.bboxRatio, 2) +
                pow(obj1.orientationAngle - obj2.orientationAngle, 2) +
                pow(obj1.area - obj2.area, 2));
}

/**
 * Classify a new object using the nearest neighbor approach.
 * 
 * param newObj The ObjectFeature of the new object to classify.
 * param database A vector of ObjectFeature objects representing the database of known objects.
 * return The label of the closest object in the database.
 */
string classifyObject(const ObjectFeature &newObj, const vector<ObjectFeature> &database)
{
    double minDistance = numeric_limits<double>::max();
    string closestLabel = "Unknown";
    // Find the closest object in the database
    for (const auto &obj : database)
    {
        double distance = calculateDistance(newObj, obj);
        if (distance < minDistance)
        {
            minDistance = distance;
            closestLabel = obj.label;
        }
    }
    return closestLabel;
}

/***********************************************************TASK-6****************************************************************************************/



/***********************************************************TASK-9****************************************************************************************/

/**
 * Classify a new object using the K-Nearest Neighbors (KNN) approach.
 * 
 * param newObj The ObjectFeature of the new object to classify.
 * param database A vector of ObjectFeature objects representing the database of known objects.
 * param k The number of nearest neighbors to consider.
 * return The label assigned to the new object based on the KNN classification.
 */
string classifyObjectKNN(const ObjectFeature &newObj, const vector<ObjectFeature> &database, int k)
{
    // Limit k to the size of the database
    k = min(k, static_cast<int>(database.size()));

    // Calculate distances between the new object and objects in the database
    vector<pair<double, string>> distances;
    for (const auto &obj : database)
    {
        double distance = calculateDistance(newObj, obj);
        distances.push_back({distance, obj.label});
    }

    // Sort distances in ascending order
    sort(distances.begin(), distances.end());

    // Count the labels of the k nearest neighbors
    unordered_map<string, int> labelCount;
    for (int i = 0; i < k; ++i)
    {
        labelCount[distances[i].second]++;
    }

    // Find the label with the maximum count
    int maxCount = 0;
    string closestLabel;
    for (const auto &count : labelCount)
    {
        if (count.second > maxCount)
        {
            maxCount = count.second;
            closestLabel = count.first;
        }
    }

    return closestLabel;
}

/***********************************************************TASK-9****************************************************************************************/
