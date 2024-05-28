/*
  Anuj Patel (NUID - 002874710)
  Ruchik Jani (NUID - 002825482)
  Spring 2024
  CS 5330 Computer Vision

  This program is the main source code where we are implementing different image matching methods and displaying the top matches as per user selection. 
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


/***********************************************************MAIN FUNCTION****************************************************************************************/
// Main function to select and run tasks based on user input
int main()
{
    int choice;
    cout << "Enter:\n1 Baseline Matching\n2 Histogram Matching\n3 Multi-histogram Matching\n4 Texture and Color\n5 Deep Network Embeddings\n6 Classic Features\n7 Custom Design\n";
    cin >> choice;

    if (choice == 1)
    {
    // Task 1 code
    cout << "Running Baseline Features Comparison...\n";
    string targetImagePath = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus/pic.1016.jpg"; // Update this path
    string directoryPath = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus";                // Update this path

    Mat targetImage = imread(targetImagePath);
    if (targetImage.empty())
    {
        cout << "Error: Unable to read target image " << targetImagePath << endl;
        return -1;
    }

    Mat targetBaselineFeatures = computeBaselineFeatures(targetImage);

    vector<pair<string, double>> distances; // Declare distances here

    for (const auto &entry : fs::directory_iterator(directoryPath))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
        {
            string filename = entry.path().string();
            Mat image = imread(filename);
            if (image.empty())
                continue;

            Mat imageBaselineFeatures = computeBaselineFeatures(image);
            double distance = computeDistance(targetBaselineFeatures, imageBaselineFeatures);
            distances.push_back(make_pair(filename, distance)); // Add filename and distance to distances
        }
    }

    sort(distances.begin(), distances.end(), [](const pair<string, double> &a, const pair<string, double> &b)
         { return a.second < b.second; });

    int N = min(4, static_cast<int>(distances.size())); // Ensure N does not exceed the number of images
    cout << "Top " << N << " matches for the target image " << targetImagePath << ":\n";
    for (int i = 0; i < N; ++i)
    {
        cout << distances[i].first << " (Distance: " << distances[i].second << ")" << endl;

        // Load and display each image
        cv::Mat img = cv::imread(distances[i].first);
        if (!img.empty())
        {
            cv::imshow("Similar Image", img);
            cv::waitKey(0);          // Wait for a key press to move to the next image
            cv::destroyAllWindows(); // Close the image window
        }
    }
    }

    else if (choice == 2)
    {
    // Task 2: RGB Histogram comparison code
    cout << "Running RGB Histogram comparison...\n";
    string targetImagePath = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus/pic.0164.jpg"; // Adjust as needed
    string directoryPath = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus";    // Adjust as needed
    int N = 4;                                                 // Number of top matches to find

    Mat targetImage = imread(targetImagePath);
    if (targetImage.empty())
    {
        cerr << "Error reading target image." << endl;
        return 1;
    }
    Mat targetHist = computeRGBHistogram(targetImage);

    vector<ImageDistance> distances;

    for (const auto &entry : fs::directory_iterator(directoryPath))
    {
        string imagePath = entry.path().string();
        Mat image = imread(imagePath);
        if (image.empty())
            continue; // Skip if image not loaded properly
        Mat hist = computeRGBHistogram(image);
        double distance = histogramDistance(targetHist, hist);
        distances.push_back({imagePath, distance});
    }

    // Sort images by distance
    sort(distances.begin(), distances.end(), [](const ImageDistance &a, const ImageDistance &b)
         { return a.distance < b.distance; });

    // Output top N matches
    for (int i = 0; i < N && i < distances.size(); ++i)
    {
        cout << "Match " << i + 1 << ": " << distances[i].imagePath << endl;

        // Load and display each image
        cv::Mat img = cv::imread(distances[i].imagePath);
        if (!img.empty())
        {
            cv::imshow("Similar Image", img);
            cv::waitKey(0);          // Wait for a key press to move to the next image
            cv::destroyAllWindows(); // Close the image window
        }
    }
    }

    else if (choice == 3)
    {
        // Execute Task 3
        cout << "Running Task 3: Multi-histogram Matching...\n";
        string targetImagePath = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus/pic.0274.jpg";
        string imagesDirectory = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus";
        int topN = 4;

        auto matches = findSimilarImages_3(targetImagePath, imagesDirectory, topN);
        cout << "Top three similar images based on combined whole and center histograms:\n";
        for (const auto &match : matches)
        {
            cout << "Image: " << match.imagePath << endl;

            // Load and display each image inside the loop
            cv::Mat img = cv::imread(match.imagePath);
            if (!img.empty())
            {
                cv::imshow("Similar Image", img);
                cv::waitKey(0);          // Wait for a key press to move to the next image
                cv::destroyAllWindows(); // Close the image window
            }
        }
        // No need for additional else blocks here
        // The return statement should be after the loop and conditional blocks
    }


    else if (choice == 4)
    {
        // Your new method code task 4
        cout << "Running Task 4: Texture and Color...\n";
        string targetImagePath = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus/pic.0535.jpg";
        string directoryPath = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus";
        int N = 4; // Number of similar images to find

        auto matches = findSimilarImages(targetImagePath, directoryPath, N);

        for (const auto &match : matches)
        {
            cout << match.imagePath << endl;

            // Load and display each image inside the loop
            cv::Mat img = cv::imread(match.imagePath);
            if (!img.empty())
            {
                cv::imshow("Similar Image", img);
                cv::waitKey(0);          // Wait for a key press to move to the next image
                cv::destroyAllWindows(); // Close the image window
            }
        }
    }

    else if (choice == 5)
{
    // Task 5 code
    cout << "Running DNN Embeddings comparison...\n";
    string featuresFile = "/home/newusername/Downloads/Project-2222/Project-2/src/ResNet18_olym.csv";
    map<string, Eigen::VectorXd> features = loadFeatures(featuresFile);

    string targetImage1 = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus/pic.0734.jpg";
    int N = 3;

    // Corrected function call
    auto similarImages1 = findSimilarImagesDNN(features, targetImage1, N);

    // Extract the base directory from the target image path
    string baseDirectory = targetImage1.substr(0, targetImage1.find_last_of("/\\")) + "/";

    cout << "Similar images to " << targetImage1 << ":\n";
    for (const auto &[imageFile, distance] : similarImages1)
    {
        // Construct the full path for each similar image
        string fullPath = baseDirectory + imageFile;

        cout << fullPath  << endl;

        // Load and display each image inside the loop using the full path
        cv::Mat img = cv::imread(fullPath); // Use fullPath instead of imageFile
        if (!img.empty())
        {
            cv::imshow("Similar Image", img);
            cv::waitKey(0);         
            cv::destroyAllWindows(); 
        }
        else
        {
            cout << "Could not open or find the image: " << fullPath << endl;
        }
    }
}

    else if (choice == 6)
    {
        // Task 6 code
        cout << "Running Tak 6 : classic features comparison...\n";
        string targetImage = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus/pic.0734.jpg";
        string imageDirectory = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus";

        auto similarImages = findSimilarImagesClassic(targetImage, imageDirectory);
        cout << "Top similar images to " << targetImage << ":\n";
        for (const auto &[imagePath, distance] : similarImages)
        {
            cout << imagePath << endl;

            // Load and display each image inside the loop
            cv::Mat img = cv::imread(imagePath);
            if (!img.empty())
            {
                cv::imshow("Similar Image", img);
                cv::waitKey(0);          // Wait for a key press to move to the next image
                cv::destroyAllWindows(); // Close the image window
            }

        }
    }
   
    else if (choice == 7) {
    // Task 7 code
    cout << "Running Task 7: Combined Features...\n";
    string targetImagePath = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus/pic.0368.jpg"; // Update this path
    string imagesDirectoryPath = "/home/newusername/Downloads/Project-2222/Project-2/data/olympus"; // Update this path

    auto features = computeFeaturesForDirectory(imagesDirectoryPath);
    auto similarImages = findSimilarImages(targetImagePath, features);

    cout << "Top similar images based on combined features:\n";
    for (int i = 0; i < min(5, static_cast<int>(similarImages.size())); ++i) {
        cout << similarImages[i].first << endl;

        // Load and display each image
        cv::Mat img = cv::imread(similarImages[i].first);
        if (!img.empty())
        {
            cv::imshow("Similar Image", img);
            cv::waitKey(0);          // Wait for a key press to move to the next image
            cv::destroyAllWindows(); // Close the image window
        }
    }
    cout << "\nLeast similar images:" << endl;
    for (int i = max(static_cast<int>(similarImages.size()) - 5, 0); i < similarImages.size(); ++i) {
        cout << similarImages[i].first << endl;

        // Load and display each image
        cv::Mat img = cv::imread(similarImages[i].first);
        if (!img.empty())
        {
            cv::imshow("Similar Image", img);
            cv::waitKey(0);          // Wait for a key press to move to the next image
            cv::destroyAllWindows(); // Close the image window
        }
    }
    }

    else
    {
        cout << "Invalid input. Please enter a number from 1 to 7.\n";
    }
    
    return 0;

}
/***********************************************************MAIN FUNCTION****************************************************************************************/
