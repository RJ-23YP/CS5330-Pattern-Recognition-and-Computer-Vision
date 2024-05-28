/*
  Anuj Patel (NUID - 002874710)
  Ruchik Jani (NUID - 002825482)
  Spring 2024
  CS 5330 Computer Vision

  This program consists definitions of different functions which contain different image matching methods that will be applied when called from image_retrieval.cpp file. 
*/


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
#include <computes.h>

using namespace cv;
using namespace std;
namespace fs = filesystem;


/***********************************************************TASK-1****************************************************************************************/
// Function: computeBaselineFeatures
// Summary: Computes baseline features from the input image.
// Parameters:
//    - image: Input image for feature extraction.
// Returns: 
//    Mat: Feature vector extracted from the input image.
Mat computeBaselineFeatures(const Mat &image)
{
    // Define the region of interest (ROI) in the center of the image.
    Rect roi(image.cols / 2 - 3, image.rows / 2 - 3, 7, 7);
    
    // Extract the ROI from the image and reshape it into a feature vector.
    Mat feature = image(roi).clone().reshape(1, 1);
    
    return feature;
}


// Function: computeDistance
// Summary: Computes the Euclidean distance between two feature vectors.
// Parameters:
//    - feature1: First feature vector.
//    - feature2: Second feature vector.
// Returns:
//    double: Euclidean distance between the two feature vectors.
double computeDistance(const Mat &feature1, const Mat &feature2)
{
    // Assert that both feature vectors have the same type and dimensions.
    CV_Assert(feature1.type() == CV_8UC1 && feature2.type() == CV_8UC1);
    CV_Assert(feature1.rows == feature2.rows && feature1.cols == feature2.cols);

    // Compute the sum of squared differences between corresponding elements.
    double sum = 0.0;
    for (int i = 0; i < feature1.total(); ++i)
    {
        sum += pow(feature1.at<uchar>(i) - feature2.at<uchar>(i), 2);
    }
    
    // Calculate the square root of the sum to obtain the Euclidean distance.
    return sqrt(sum);
}
/***********************************************************TASK-1****************************************************************************************/




/***********************************************************TASK-2****************************************************************************************/

// Function: computeRGBHistogram
// Summary: Computes the RGB histogram of the input image.
// Parameters:
//    - image: Input image for which RGB histogram is to be computed.
// Returns:
//    cv::Mat: Computed RGB histogram.
cv::Mat computeRGBHistogram(const cv::Mat &image)
{
    int histSize[] = {8, 8, 8}; // 8 bins for each channel
    float range[] = {0, 256};
    const float *histRange[] = {range, range, range};
    cv::Mat hist;
    int channels[] = {0, 1, 2};
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, histSize, histRange, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

// Function: histogramDistance
// Summary: Computes the distance between two RGB histograms.
// Parameters:
//    - hist1: First RGB histogram.
//    - hist2: Second RGB histogram.
// Returns:
//    double: Distance between the two RGB histograms.
double histogramDistance(const cv::Mat &hist1, const cv::Mat &hist2)
{
    Mat diff;
    absdiff(hist1, hist2, diff);              // Compute absolute difference
    diff.convertTo(diff, CV_64F);             // Convert to double
    double distance = sum(diff.mul(diff))[0]; // Compute sum of squared differences
    return sqrt(distance);
}

/***********************************************************TASK-2****************************************************************************************/




/***********************************************************TASK-3****************************************************************************************/



// Function: computeHistogram_3
// Summary: Computes the histogram of an image, optionally focusing on a specified region of interest (ROI).
// Parameters:
//    - image: Input image for which the histogram is to be computed.
//    - bins: Number of bins for the histogram (default is 8).
//    - roi: Region of interest (default is an empty rectangle, meaning the entire image is considered).
// Returns:
//    cv::Mat: Computed histogram of the input image.
cv::Mat computeHistogram_3(const cv::Mat &image, int bins, const cv::Rect &roi)
{
    cv::Mat hist;
    cv::Mat imgRegion = roi.area() > 0 ? image(roi) : image; // Extract ROI if provided, otherwise use the whole image
    int histSize[] = {bins, bins, bins};
    float range[] = {0, 256};
    const float *ranges[] = {range, range, range};
    int channels[] = {0, 1, 2};
    cv::calcHist(&imgRegion, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

// Function: histogramIntersection
// Summary: Computes the histogram intersection between two histograms.
// Parameters:
//    - hist1: First histogram.
//    - hist2: Second histogram.
// Returns:
//    double: Histogram intersection value between the two histograms.
double histogramIntersection(const cv::Mat &hist1, const cv::Mat &hist2)
{
    return cv::compareHist(hist1, hist2, cv::HISTCMP_INTERSECT);
}

// Function: findSimilarImages_3
// Summary: Finds similar images in a directory based on histogram comparison with a target image.
// Parameters:
//    - targetImagePath: Path of the target image for comparison.
//    - directoryPath: Path of the directory containing images to compare with.
//    - topN: Number of top similar images to return.
// Returns:
//    vector<ImageDistance_3>: Vector containing ImageDistance_3 objects representing similar images.
vector<ImageDistance_3> findSimilarImages_3(const string &targetImagePath, const string &directoryPath, int topN)
{
    cv::Mat targetImage = cv::imread(targetImagePath);
    if (targetImage.empty())
    {
        cerr << "Error loading target image.\n";
        return {};
    }

    // Calculate histograms for target image and a central region of interest
    cv::Mat targetHistWhole = computeHistogram_3(targetImage, 8); // Histogram for whole image
    cv::Rect centralRegion(targetImage.cols / 4, targetImage.rows / 4, targetImage.cols / 2, targetImage.rows / 2);
    cv::Mat targetHistCenter = computeHistogram_3(targetImage, 8, centralRegion); // Histogram for central region

    vector<ImageDistance_3> distances;
    for (const auto &entry : fs::directory_iterator(directoryPath))
    {
        if (!entry.is_regular_file())
            continue; // Skip if not a file
        string filePath = entry.path().string();
        cv::Mat image = cv::imread(filePath);
        if (image.empty())
            continue;

        // Calculate histograms for the current image and a central region of interest
        cv::Mat imageHistWhole = computeHistogram_3(image, 8); // Histogram for whole image
        cv::Mat imageHistCenter = computeHistogram_3(image, 8, centralRegion); // Histogram for central region

        // Compute histogram intersection distances for whole image and central region
        double distanceWhole = histogramIntersection(targetHistWhole, imageHistWhole);
        double distanceCenter = histogramIntersection(targetHistCenter, imageHistCenter);
        double combinedDistance = (distanceWhole + distanceCenter) / 2; // Average distance

        distances.emplace_back(filePath, combinedDistance); // Store distance with file path
    }

    // Sort distances in descending order of similarity
    sort(distances.begin(), distances.end(), [](const ImageDistance_3 &a, const ImageDistance_3 &b)
              {
                  return a.distance > b.distance;
              });

    // Return topN similar images
    if (distances.size() > topN)
        distances.resize(topN);
    return distances;
}

/***********************************************************TASK-3****************************************************************************************/




/***********************************************************TASK-4****************************************************************************************/


// Function to compute the color histogram TASK 4
// Summary: Computes the color histogram of the input image in HSV color space.
// Parameters:
//    - image: Input image for which the color histogram is to be computed.
// Returns:
//    cv::Mat: Computed color histogram.
cv::Mat computeColorHistogram(const cv::Mat &image)
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    int hBins = 50, sBins = 60;
    int histSize[] = {hBins, sBins};
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    const float *ranges[] = {hRanges, sRanges};
    int channels[] = {0, 1};

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    return hist;
}

// Function to compute texture features (magnitude and orientation histograms) TASK 4
// Summary: Computes the texture features of the input image, including magnitude and orientation histograms.
// Parameters:
//    - image: Input image for which the texture features are to be computed.
// Returns:
//    pair<cv::Mat, cv::Mat>: Pair of computed magnitude and orientation histograms.
pair<cv::Mat, cv::Mat> computeTextureFeatures(const cv::Mat &image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

    cv::Mat magnitude, angle;
    cv::cartToPolar(grad_x, grad_y, magnitude, angle, true);

    int magHistSize[] = {50};
    float magRange[] = {0, 256};
    const float *magHistRange[] = {magRange};

    cv::Mat magHist;
    cv::calcHist(&magnitude, 1, 0, cv::Mat(), magHist, 1, magHistSize, magHistRange, true, false);
    cv::normalize(magHist, magHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    int angleBins = 360; // One bin per degree
    float angleRanges[] = {0, 360};
    const float *angleHistRanges[] = {angleRanges};

    cv::Mat angleHist;
    cv::calcHist(&angle, 1, 0, cv::Mat(), angleHist, 1, &angleBins, angleHistRanges, true, false);
    cv::normalize(angleHist, angleHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    return {magHist, angleHist};
}

// Function to compare histogram TASK 4
// Summary: Compares two histograms using Bhattacharyya distance.
// Parameters:
//    - hist1: First histogram.
//    - hist2: Second histogram.
// Returns:
//    double: Bhattacharyya distance between the two histograms.
double compareHistograms(const cv::Mat &hist1, const cv::Mat &hist2)
{
    return cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
}

// Function: findSimilarImages
// Summary: Finds the N most similar images in a directory based on color and texture features.
// Parameters:
//    - targetImagePath: Path of the target image for comparison.
//    - directoryPath: Path of the directory containing images to compare with.
//    - N: Number of most similar images to find.
// Returns:
//    vector<ImageDistance>: Vector containing ImageDistance objects representing the most similar images.
vector<ImageDistance> findSimilarImages(const string &targetImagePath, const string &directoryPath, int N)
{
    cv::Mat targetImage = cv::imread(targetImagePath);
    if (targetImage.empty())
    {
        cerr << "Failed to load target image: " << targetImagePath << endl;
        return {};
    }
    cv::Mat targetColorHist = computeColorHistogram(targetImage);
    auto [targetMagHist, targetAngleHist] = computeTextureFeatures(targetImage);

    vector<ImageDistance> distances;

    for (const auto &entry : fs::directory_iterator(directoryPath))
    {
        string pathString = entry.path().string();
        if (pathString.find(".jpg") == string::npos && pathString.find(".png") == string::npos)
        {
            continue;
        }

        cv::Mat image = cv::imread(pathString);
        if (image.empty())
        {
            cerr << "Failed to load image: " << pathString << endl;
            continue;
        }
        cv::Mat colorHist = computeColorHistogram(image);
        auto [magHist, angleHist] = computeTextureFeatures(image);

        double colorDistance = compareHistograms(targetColorHist, colorHist);
        double textureMagDistance = compareHistograms(targetMagHist, magHist);
        double textureAngleDistance = compareHistograms(targetAngleHist, angleHist);

        double totalDistance = colorDistance + textureMagDistance + textureAngleDistance; // Equally weighted
        distances.push_back({pathString, totalDistance});
    }

    sort(distances.begin(), distances.end(), [](const ImageDistance &a, const ImageDistance &b)
              { return a.distance < b.distance; });

    if (distances.size() > N)
    {
        distances.resize(N);
    }

    return distances;
}

/***********************************************************TASK-4****************************************************************************************/




/***********************************************************TASK-5****************************************************************************************/
// Function: split
// Summary: Splits a string into tokens using a delimiter character.
// Parameters:
//    - s: Input string to be split.
//    - delimiter: Delimiter character used to split the string.
// Returns:
//    vector<string>: Vector containing the tokens obtained after splitting the string.
vector<string> split(const string &s, char delimiter)
{
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

// Function: loadFeatures
// Summary: Loads features from a CSV file into a map.
// Parameters:
//    - filename: Path of the CSV file containing features.
// Returns:
//    map<string, Eigen::VectorXd>: Map containing image filenames as keys and feature vectors as values.
map<string, Eigen::VectorXd> loadFeatures(const string &filename)
{
    map<string, Eigen::VectorXd> features;
    ifstream file(filename);
    string line;

    while (getline(file, line))
    {
        vector<string> tokens = split(line, ',');
        if (tokens.size() != 513)
            continue; // Filename + 512 features

        string imageFile = fs::path(tokens[0]).filename().string(); // Extract filename for comparison
        Eigen::VectorXd feature(512);
        for (int i = 0; i < 512; ++i)
        {
            feature[i] = stod(tokens[i + 1]);
        }
        features[imageFile] = feature;
    }

    return features;
}

// Function: cosineDistance
// Summary: Computes the cosine distance between two Eigen vectors.
// Parameters:
//    - v1: First Eigen vector.
//    - v2: Second Eigen vector.
// Returns:
//    double: Cosine distance between the two vectors.
double cosineDistance(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2)
{
    return 1.0 - (v1.dot(v2) / (v1.norm() * v2.norm()));
}

// Function: findSimilarImagesDNN
// Summary: Finds the N most similar images to a target image using DNN embeddings.
// Parameters:
//    - features: Map containing image filenames and their corresponding DNN feature vectors.
//    - targetImagePath: Path of the target image for comparison.
//    - N: Number of most similar images to find.
// Returns:
//    vector<pair<string, double>>: Vector of pairs containing filenames and cosine distances of the most similar images.
vector<pair<string, double>> findSimilarImagesDNN(const map<string, Eigen::VectorXd> &features, const string &targetImagePath, int N)
{
    string targetImage = fs::path(targetImagePath).filename().string();

    if (features.find(targetImage) == features.end())
    {
        cerr << "Error: Feature vector for target image '" << targetImage << "' not found." << endl;
        return {};
    }

    auto targetFeature = features.at(targetImage);
    vector<pair<string, double>> distances;

    for (const auto &[imageFile, feature] : features)
    {
        if (imageFile == targetImage)
            continue;
        double distance = cosineDistance(targetFeature, feature);
        distances.emplace_back(imageFile, distance);
    }

    sort(distances.begin(), distances.end(), [](const auto &a, const auto &b)
              { return a.second < b.second; });

    if (distances.size() > N)
    {
        distances.resize(N);
    }

    return distances;
}

/***********************************************************TASK-5****************************************************************************************/




/***********************************************************TASK-6****************************************************************************************/
// Function: computeHistogram
// Summary: Computes the color histogram of the input image in HSV color space.
// Parameters:
//    - image: Input image for which the color histogram is to be computed.
// Returns:
//    cv::Mat: Computed color histogram.
cv::Mat computeHistogram(const cv::Mat &image)
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    int h_bins = 50, s_bins = 60;
    int histSize[] = {h_bins, s_bins};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float *ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    return hist;
}

// Function: computeLBP
// Summary: Computes the Local Binary Patterns (LBP) of the input image.
// Parameters:
//    - src: Input grayscale image for which the LBP is to be computed.
// Returns:
//    cv::Mat: Computed LBP image.
cv::Mat computeLBP(const cv::Mat &src)
{
    cv::Mat dst = cv::Mat::zeros(src.rows - 2, src.cols - 2, CV_8UC1);
    for (int i = 1; i < src.rows - 1; i++)
    {
        for (int j = 1; j < src.cols - 1; j++)
        {
            unsigned char center = src.at<unsigned char>(i, j);
            unsigned char code = 0;
            code |= (src.at<unsigned char>(i - 1, j - 1) > center) << 7;
            code |= (src.at<unsigned char>(i - 1, j) > center) << 6;
            code |= (src.at<unsigned char>(i - 1, j + 1) > center) << 5;
            code |= (src.at<unsigned char>(i, j + 1) > center) << 4;
            code |= (src.at<unsigned char>(i + 1, j + 1) > center) << 3;
            code |= (src.at<unsigned char>(i + 1, j) > center) << 2;
            code |= (src.at<unsigned char>(i + 1, j - 1) > center) << 1;
            code |= (src.at<unsigned char>(i, j - 1) > center) << 0;
            dst.at<unsigned char>(i - 1, j - 1) = code;
        }
    }
    return dst;
}

// Function: featureDistance
// Summary: Calculates the Euclidean distance between two feature vectors.
// Parameters:
//    - hist1: First feature vector.
//    - hist2: Second feature vector.
// Returns:
//    double: Euclidean distance between the two feature vectors.
double featureDistance(const cv::Mat &hist1, const cv::Mat &hist2)
{
    return cv::norm(hist1, hist2, cv::NORM_L2);
}

// Function: findSimilarImagesClassic
// Summary: Finds the N most similar images to a target image using classic features (color histogram and LBP).
// Parameters:
//    - targetImagePath: Path of the target image for comparison.
//    - directoryPath: Path of the directory containing images to compare with.
//    - N: Number of most similar images to find. Default is 5.
// Returns:
//    vector<pair<string, double>>: Vector of pairs containing filenames and distances of the most similar images.
vector<pair<string, double>> findSimilarImagesClassic(const string &targetImagePath, const string &directoryPath, int N)
{
    cv::Mat targetImage = cv::imread(targetImagePath);
    cv::Mat targetHist = computeHistogram(targetImage);
    cv::Mat targetLBP = computeLBP(targetImage);
    vector<pair<string, double>> distances;

    for (const auto &entry : fs::directory_iterator(directoryPath))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
        {
            cv::Mat image = cv::imread(entry.path().string());
            cv::Mat hist = computeHistogram(image);
            cv::Mat lbp = computeLBP(image);

            double distance = featureDistance(targetHist, hist) + featureDistance(targetLBP, lbp);
            distances.push_back(make_pair(entry.path().string(), distance));
        }
    }

    sort(distances.begin(), distances.end(), [](const auto &a, const auto &b)
              { return a.second < b.second; });

    if (distances.size() > N)
        distances.resize(N);

    return distances;
}
/***********************************************************TASK-6****************************************************************************************/



/***********************************************************TASK-7****************************************************************************************/

// Function to calculate color histogram of an image
// Summary: Computes the color histogram of the input image in HSV color space.
// Parameters:
//    - image: Input image for which the color histogram is to be computed.
// Returns:
//    cv::Mat: Computed color histogram.
cv::Mat calculateHistogram(const cv::Mat& image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    int hBins = 50, sBins = 60;
    int histSize[] = { hBins, sBins };
    float hRanges[] = { 0, 180 };
    float sRanges[] = { 0, 256 };
    const float* ranges[] = { hRanges, sRanges };
    int channels[] = { 0, 1 };
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

// Function to calculate Euclidean distance between two histograms
// Summary: Calculates the Euclidean distance between two histograms.
// Parameters:
//    - hist1: First histogram.
//    - hist2: Second histogram.
// Returns:
//    double: Euclidean distance between the two histograms.
double calculateEuclideanDistance(const cv::Mat& hist1, const cv::Mat& hist2) {
    return cv::norm(hist1, hist2, cv::NORM_L2);
}

// Function to calculate Local Binary Patterns (LBP) of an image
// Summary: Computes the Local Binary Patterns (LBP) of the input grayscale image.
// Parameters:
//    - src: Input grayscale image for which the LBP is to be computed.
// Returns:
//    cv::Mat: Computed LBP image.
cv::Mat calculateLBP(const cv::Mat& src) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat lbpImage = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (int y = 1; y < gray.rows - 1; y++) {
        for (int x = 1; x < gray.cols - 1; x++) {
            uchar center = gray.at<uchar>(y, x);
            unsigned char code = 0;
            code |= (gray.at<uchar>(y-1, x-1) > center) << 7;
            code |= (gray.at<uchar>(y-1, x) > center) << 6;
            code |= (gray.at<uchar>(y-1, x+1) > center) << 5;
            code |= (gray.at<uchar>(y, x+1) > center) << 4;
            code |= (gray.at<uchar>(y+1, x+1) > center) << 3;
            code |= (gray.at<uchar>(y+1, x) > center) << 2;
            code |= (gray.at<uchar>(y+1, x-1) > center) << 1;
            code |= (gray.at<uchar>(y, x-1) > center) << 0;
            lbpImage.at<uchar>(y, x) = code;
        }
    }
    // Convert LBP image to histogram
    int histSize = 256; // LBP can have 256 possible values
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat lbpHist;
    cv::calcHist(&lbpImage, 1, 0, cv::Mat(), lbpHist, 1, &histSize, &histRange, true, false);
    cv::normalize(lbpHist, lbpHist, 0, 1, cv::NORM_MINMAX);
    return lbpHist;
}

// Function to calculate Hu Moments of an image
// Summary: Computes the Hu Moments of the input grayscale image.
// Parameters:
//    - src: Input grayscale image for which the Hu Moments are to be computed.
// Returns:
//    cv::Mat: Computed Hu Moments.
cv::Mat calculateHuMoments(const cv::Mat& src) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Moments moments = cv::moments(gray, true);
    cv::Mat huMoments;
    cv::HuMoments(moments, huMoments);
    // Log scale transform
    for (int i = 0; i < huMoments.rows; i++) {
        huMoments.at<double>(i, 0) = -1 * copysign(1.0, huMoments.at<double>(i, 0)) * log10(abs(huMoments.at<double>(i, 0)));
    }
    return huMoments;
}

// Function to compute features for all images in a directory
// Summary: Computes image features (color histogram, LBP, Hu Moments) for all images in the given directory.
// Parameters:
//    - directoryPath: Path to the directory containing images.
// Returns:
//    vector<ImageFeature_7>: Vector of ImageFeature_7 structs, each containing features of an image.
vector<ImageFeature_7> computeFeaturesForDirectory(const string& directoryPath) {
    vector<ImageFeature_7> features;
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            cv::Mat image = cv::imread(entry.path().string());
            if (!image.empty()) {
                ImageFeature_7 feature;
                feature.filePath = entry.path().string();
                feature.histogram = calculateHistogram(image);
                feature.texture = calculateLBP(image);
                feature.shape = calculateHuMoments(image);
                features.push_back(feature);
            }
        }
    }
    return features;
}

// Function to calculate weighted distance between two image features
// Summary: Calculates the weighted distance between two image features based on their color histogram, texture, and shape features.
// Parameters:
//    - f1: First image feature.
//    - f2: Second image feature.
// Returns:
//    double: Weighted distance between the two image features.
double calculateWeightedDistance(const ImageFeature_7& f1, const ImageFeature_7& f2) {
    double weightColor = 0.5, weightTexture = 0.3, weightShape = 0.2;
    double colorDistance = calculateEuclideanDistance(f1.histogram, f2.histogram);
    double textureDistance = calculateEuclideanDistance(f1.texture, f2.texture);
    double shapeDistance = calculateEuclideanDistance(f1.shape, f2.shape);

    return colorDistance * weightColor + textureDistance * weightTexture + shapeDistance * weightShape;
}

// Function to find similar images to a target image
// Summary: Finds similar images to a target image based on their color histogram, texture, and shape features.
// Parameters:
//    - targetImagePath: Path to the target image.
//    - features: Vector of image features computed for all images in the dataset.
// Returns:
//    vector<pair<string, double>>: Vector of pairs containing filenames and distances of similar images.
vector<pair<string, double>> findSimilarImages(const string& targetImagePath, const vector<ImageFeature_7>& features) {
    cv::Mat targetImage = cv::imread(targetImagePath);
    ImageFeature_7 targetFeature;
    targetFeature.histogram = calculateHistogram(targetImage);
    targetFeature.texture = calculateLBP(targetImage);
    targetFeature.shape = calculateHuMoments(targetImage);

    vector<pair<string, double>> distances;
    for (const auto& feature : features) {
        double distance = calculateWeightedDistance(targetFeature, feature);
        distances.emplace_back(feature.filePath, distance);
    }

    sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    return distances;
}

/***********************************************************TASK-7****************************************************************************************/

