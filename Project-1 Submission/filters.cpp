/*
  Ruchik Jani (NUID - 002825482)
  Anuj Patel (NUID - 002874710)
  Spring 2024
  CS 5330 Computer Vision

  This program consists definitions of different functions which contain different filters that will be applied when called from vidDisplay.cpp file. 
*/


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filters.h>
#include <sys/time.h>
#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings


using namespace cv;

    

    // CUSTOM GREYSCALE FILTER
    // Converts a color image to greyscale using a custom transformation.
    // 
    // Arguments:
    //   - src: Source color image.
    // 
    // Returns:
    //   - 0 on success, -1 on failure (empty source image).
    int greyconvert::greyscale() 
    {
        if (src.empty()) {
            return -1; // Error: source image is empty
        }

        // Create a new Mat for the destination image
        dst = Mat(src.rows, src.cols, CV_8UC3);

        // Iterate through pixels and apply greyscale transformation
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                // Get the pixel value in the red channel
                uchar redValue = src.at<Vec3b>(i, j)[2];

                // Calculate the greyscale value using the described transformation
                uchar greyValue = 255 - redValue;

                // Set the grey value in all three channels
                dst.at<Vec3b>(i, j)[0] = greyValue;
                dst.at<Vec3b>(i, j)[1] = greyValue;
                dst.at<Vec3b>(i, j)[2] = greyValue;
            }
        }

        return 0; // Success
    }

    // Setter for the source image in the greyscale filter.
    // Arguments:
    //   - inputSrc: Source image to be set.
    void greyconvert::setSourceImage(const Mat &inputSrc) 
    {
        src = inputSrc.clone(); // Clone to avoid modifying the original image
    }

    // Getter for the destination image in the greyscale filter.
    // Returns:
    //   - Destination greyscale image.
    Mat greyconvert::getDestinationImage() const 
    {
        return dst.clone(); // Clone to avoid providing a reference to internal data
    }




    // SEPIA FILTER
    // Applies a sepia effect to a color image.
    // Returns:
    //   - 0 on success, -1 on failure (empty source image).
    int sepiaconvert::sepiafilter() 
    {
    if (src.empty()) {
        return -1; // Error: source image is empty
    }

    // Create a new Mat for the destination image
    dst = Mat(src.rows, src.cols, CV_8UC3);

    // Iterate through pixels and apply sepia transformation
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // Get the pixel values in each channel
            uchar blueValue = src.at<Vec3b>(i, j)[0];
            uchar greenValue = src.at<Vec3b>(i, j)[1];
            uchar redValue = src.at<Vec3b>(i, j)[2];

            // Calculate new values for each channel for sepia effect
            uchar newBlue = saturate_cast<uchar>(0.272 * redValue + 0.534 * greenValue + 0.131 * blueValue);
            uchar newGreen = saturate_cast<uchar>(0.349 * redValue + 0.686 * greenValue + 0.168 * blueValue);
            uchar newRed = saturate_cast<uchar>(0.393 * redValue + 0.769 * greenValue + 0.189 * blueValue);

            // Set the new values in the destination image
            dst.at<Vec3b>(i, j)[0] = newBlue;
            dst.at<Vec3b>(i, j)[1] = newGreen;
            dst.at<Vec3b>(i, j)[2] = newRed;
        }
    }

    return 0; // Success
    }

    // Setter for the source image in the sepia filter.
    // Arguments:
    //   - inputSrc: Source image to be set.
    void sepiaconvert::setSourceImage(const Mat &inputSrc) 
    {
        src = inputSrc.clone(); // Clone to avoid modifying the original image
    }

    // Getter for the destination image in the sepia filter. 
    // Returns:
    //   - Destination sepia-filtered image.
    Mat sepiaconvert::getDestinationImage() const 
    {
        return dst.clone(); // Clone to avoid providing a reference to internal data
    }




    // 3x3 SOBEL X FILTER
    // Applies a 3x3 Sobel X filter to each channel of a color image.
    // 
    // Arguments:
    //   - src: Source color image.
    //   - dst: Destination image.
    // 
    // Returns:
    //   - 0 on success.
    int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    dst = cv::Mat(src.size(), CV_16SC3);

    // Design 3x3 Positive Right Sobel X filter
    short sobelXFilterData[] = { 1, 0, -1,
                                  2, 0, -2,
                                  1, 0, -1 };
    Mat sobelXFilter(3, 3, CV_16S, sobelXFilterData);

    // Apply Sobel X filter to each channel separately
    for (int c = 0; c < src.channels(); ++c) {
        for (int i = 1; i < src.rows - 1; ++i) {
            for (int j = 1; j < src.cols - 1; ++j) {
                short pixelValue = 0;
                for (int x = -1; x <= 1; ++x) {
                    for (int y = -1; y <= 1; ++y) {
                        pixelValue += src.at<cv::Vec3b>(i + x, j + y)[c] * sobelXFilter.at<short>(x + 1, y + 1);
                    }
                }
                dst.at<cv::Vec3s>(i, j)[c] = pixelValue;
            }
        }
    }
    return 0;
    }



    // 3x3 SOBEL Y FILTER
    // Applies a 3x3 Sobel Y filter to each channel of a color image.
    // 
    // Arguments:
    //   - src: Source color image.
    //   - dst: Destination image.
    // 
    // Returns:
    //   - 0 on success.
    int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    dst = cv::Mat(src.size(), CV_16SC3);

    // Design 3x3 Positive Up Sobel Y filter
    short sobelYFilterData[] = { 1, 2, 1,
                                 0, 0, 0,
                                 -1, -2, -1 };
    cv::Mat sobelYFilter(3, 3, CV_16S, sobelYFilterData);

    // Apply Sobel Y filter to each channel separately
    for (int c = 0; c < src.channels(); ++c) {
        for (int i = 1; i < src.rows - 1; ++i) {
            for (int j = 1; j < src.cols - 1; ++j) {
                short pixelValue = 0;
                for (int x = -1; x <= 1; ++x) {
                    for (int y = -1; y <= 1; ++y) {
                        pixelValue += src.at<cv::Vec3b>(i + x, j + y)[c] * sobelYFilter.at<short>(x + 1, y + 1);
                    }
                }
                dst.at<cv::Vec3s>(i, j)[c] = pixelValue;
            }
        }
    }
    return 0;
    }



    // EMBOSSING EFFECT FILTER
    // Applies an embossing effect to a color image using Sobel X and Sobel Y filters.
    // 
    // Arguments:
    //   - src: Source color image.
    //   - dst: Destination image.
    //   - directionX: X-direction for embossing.
    //   - directionY: Y-direction for embossing.
    void embossEffect(cv::Mat &src, cv::Mat &dst, double directionX, double directionY) 
    {
    // Apply Sobel X filter
    Mat sobelXResult;
    sobelX3x3(src, sobelXResult);

    // Apply Sobel Y filter
    Mat sobelYResult;
    sobelY3x3(src, sobelYResult);

    // Combine Sobel X and Sobel Y results
    dst = src.clone();
    for (int i = 0; i < src.rows; ++i) 
        {
        for (int j = 0; j < src.cols; ++j) 
            {
            for (int c = 0; c < src.channels(); ++c) 
                {
                short pixelX = sobelXResult.at<cv::Vec3s>(i, j)[c];
                short pixelY = sobelYResult.at<cv::Vec3s>(i, j)[c];

                // Calculate embossing value using dot product with the specified direction
                short embossValue = static_cast<short>(pixelX * directionX + pixelY * directionY);

                // Adjust the pixel value to be in the valid intensity range
                embossValue = cv::saturate_cast<uchar>(128 + embossValue / 8);

                // Set the new pixel value in the destination image
                dst.at<cv::Vec3b>(i, j)[c] = embossValue;
                }
            }
        }
    }




   // ADJUST BRIGHTNESS AND CONTRAST
    // Adjusts the brightness and contrast of an image.
    // 
    // Arguments:
    //   - image: Input image.
    //   - alpha: Multiplicative factor for image intensity.
    //   - beta: Additive factor for image intensity.
    void adjustBrightnessAndContrast(Mat &image, double alpha, int beta) {
    // Convert to CV_32F for more accurate arithmetic
    image.convertTo(image, CV_32F);

    // Apply alpha and beta
    image = alpha * image + beta;

    // Saturate the values to fit in the valid range
    image = cv::max(0.0, cv::min(255.0, image));

    // Convert back to CV_8U
    image.convertTo(image, CV_8U);
    }






    // GRADIENT MAGNITUDE IMAGE FILTER
    // Computes the gradient magnitude of an image using Sobel X and Sobel Y filters.
    // 
    // Arguments:
    //   - sx: Image gradient in the X-direction.
    //   - sy: Image gradient in the Y-direction.
    //   - dst: Destination image for gradient magnitude.
    void magnitude(Mat &sx, Mat &sy, Mat &dst) 
    {
    // Ensure the input images have the same size and type
    CV_Assert(sx.size() == sy.size() && sx.type() == CV_16S && sy.type() == CV_16S);

    // Calculate the gradient magnitude using Euclidean distance
    cv::Mat_<float> magnitudeMat;
    cv::Mat_<float> magnitudeX, magnitudeY;

    // Convert to float
    sx.convertTo(magnitudeX, CV_32F);
    sy.convertTo(magnitudeY, CV_32F);

    // Calculate the squared magnitude of each component
    cv::multiply(magnitudeX, magnitudeX, magnitudeX);
    cv::multiply(magnitudeY, magnitudeY, magnitudeY);

    // Sum the squared magnitudes
    cv::add(magnitudeX, magnitudeY, magnitudeMat);

    // Take the square root to get the Euclidean distance
    cv::sqrt(magnitudeMat, magnitudeMat);

    // Convert the magnitude image to a uchar color image suitable for display
    cv::normalize(magnitudeMat, dst, 0, 255, cv::NORM_MINMAX, CV_8U);
    }





    // BLURRING AND QUANTIZATION FILTER
    // Applies Gaussian blur and quantizes the image into a fixed number of levels.
    // 
    // Arguments:
    //   - src: Source color image.
    //   - dst: Destination image.
    //   - levels: Number of quantization levels.
    // 
    // Returns:
    //   - Processed image.
    Mat blurQuantize(Mat &src, Mat &dst, int levels) {
    // Blur the image
    cv::GaussianBlur(src, dst, cv::Size(9, 9), 0);

    // Quantize the image into the specified number of levels
    std::vector<cv::Mat> channels;
    cv::split(dst, channels);
    int b = 255 / levels;
    for (int i = 0; i < channels.size(); ++i) {
    channels[i] = (channels[i] / b) * b;
    }

    cv::merge(channels, dst);
    
    //dst = (dst / b) * b;

    return dst;
    }


// Blur function 1
// Purpose: Apply a 5x5 Gaussian blur kernel to an input image using nested loops.
// Arguments:
//   - src: Input image matrix (cv::Mat&) that needs to be blurred.
//   - dst: Output image matrix (cv::Mat&) where the blurred result will be stored.
// Returns: 0 on success, -1 if the input image is empty.
int blur5x5_1(cv::Mat &src, cv::Mat &dst) 
{
    // Check if the input image is empty
    if (src.empty()) {
        return -1;
    }

    // Copy the source image to the destination image
    src.copyTo(dst);

    // Define a 5x5 Gaussian blur kernel
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    // Get the number of rows and columns in the source image
    int rows = src.rows;
    int cols = src.cols;

    // Apply the Gaussian blur filter using nested loops
    for (int i = 2; i < rows - 2; ++i) {
        for (int j = 2; j < cols - 2; ++j) {
            for (int c = 0; c < src.channels(); ++c) {
                int sum = 0;

                // Convolution operation using the Gaussian blur kernel
                for (int ki = -2; ki <= 2; ++ki) {
                    for (int kj = -2; kj <= 2; ++kj) {
                        sum += src.at<cv::Vec3b>(i + ki, j + kj)[c] * kernel[ki + 2][kj + 2];
                    }
                }

                // Update the destination pixel with the averaged result
                dst.at<cv::Vec3b>(i, j)[c] = sum / 84;
            }
        }
    }

    // Return success
    return 0;
}




// Blur function 2
 // Purpose: Apply a separable 5x5 Gaussian blur kernel to an input image using nested loops.
// Arguments:
//   - src: Input image matrix (cv::Mat&) that needs to be blurred.
//   - dst: Output image matrix (cv::Mat&) where the blurred result will be stored.
// Returns: 0 on success, -1 if the input image is empty.
int blur5x5_2(cv::Mat &src, cv::Mat &dst) 
{
    // Check if the input image is empty
    if (src.empty()) {
        return -1;
    }

    // Create the destination image with the same size and type as the source image
    dst.create(src.size(), src.type());

    // Define a 5-element separable Gaussian blur kernel
    int kernel[5] = {1, 2, 4, 2, 1};

    // Get the number of rows and columns in the source image
    int rows = src.rows;
    int cols = src.cols;

    // Apply the separable Gaussian blur filter using nested loops
    for (int i = 0; i < rows; ++i) {
        for (int j = 2; j < cols - 2; ++j) {
            for (int c = 0; c < src.channels(); ++c) {
                int sum = 0;

                // Convolution operation along the columns using the separable kernel
                for (int kj = -2; kj <= 2; ++kj) {
                    sum += src.at<cv::Vec3b>(i, j + kj)[c] * kernel[kj + 2];
                }

                // Update the destination pixel with the averaged result
                dst.at<cv::Vec3b>(i, j)[c] = sum / 10;
            }
        }
    }

    // Apply the separable Gaussian blur filter along the rows
    for (int i = 2; i < rows - 2; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int c = 0; c < src.channels(); ++c) {
                int sum = 0;

                // Convolution operation along the rows using the separable kernel
                for (int ki = -2; ki <= 2; ++ki) {
                    sum += dst.at<cv::Vec3b>(i + ki, j)[c] * kernel[ki + 2];
                }

                // Update the destination pixel with the averaged result
                dst.at<cv::Vec3b>(i, j)[c] = sum / 10;
            }
        }
    }

    // Return success
    return 0;
}



// Function to get current time in seconds
// Purpose: Get the current time in seconds using the gettimeofday function.
// Arguments: None.
// Returns: A double representing the current time in seconds, calculated by combining seconds and microseconds.
double getTime() {
    // Declare a timeval structure to store the current time
    struct timeval cur;

    // Get the current time of day and store it in the 'cur' structure
    gettimeofday(&cur, NULL);

    // Return the combined time in seconds, including both seconds and microseconds
    return (cur.tv_sec + cur.tv_usec / 1000000.0);
}
