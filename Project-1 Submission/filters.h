/*
  Ruchik Jani (NUID - 002825482)
  Anuj Patel (NUID - 002874710)
  Spring 2024
  CS 5330 Computer Vision

  This header file consists declarations of different functions which contain different filters that will be applied when called from vidDisplay.cpp & filters.cpp file. 
*/


#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/core.hpp>

using namespace cv;



//CUSTOM Greyscale Filter
class greyconvert {
private:
    Mat src; //source frame
    Mat dst; //destination frame

public:
    int greyscale(); //Greyscale Filter code

    void setSourceImage(const Mat &inputSrc); //Function to pass source frame to the greyscale() filter function.

    Mat getDestinationImage() const; //Function to obtain destination frame from the greyscale() filter function.
};



//Sepia Filter
class sepiaconvert {
private:
    Mat src; //source frame
    Mat dst; //destination frame

public:
    int sepiafilter(); //Sepia Filter code

    void setSourceImage(const Mat &inputSrc); //Function to pass source frame to the sepiafilter() filter function.

    Mat getDestinationImage() const; //Function to obtain destination frame from the sepiafilter() filter function.
};



//Sobel  FIlter
int sobelX3x3( Mat &src, Mat &dst ); //SobelX filter

int sobelY3x3( Mat &src, Mat &dst ); //SobelY filter

//Embossing Filter using Sobel X & Y filters
void embossEffect(cv::Mat &src, cv::Mat &dst, double directionX, double directionY);

//Function to adjust brightness and contrast
void adjustBrightnessAndContrast(Mat &image, double alpha, int beta); 

//Gradient Magntitude Image Filter
void magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

//Blurring and Quantization Filter
Mat blurQuantize(Mat &src, Mat &dst, int levels);

// Prototypes for the blur functions
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

// Function to get current time in seconds
double getTime();

#endif // FILTERS_H

