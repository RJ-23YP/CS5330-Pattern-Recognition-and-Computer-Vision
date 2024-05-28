/*
  Ruchik Jani (NUID - 002825482)
  Anuj Patel (NUID - 002874710)
  Spring 2024
  CS 5330 Computer Vision

  This program is the main source code where we are implementing different filters and effects on an image input by pressing different keys. 
*/

#define FACE_CASCADE_FILE "/home/rj/Project-1 (Tasks 1-3)/Task-1/haarcascade_frontalface_alt2.xml"

#include <cmath>
#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp> // Include for cvtColor function, convertScaleAbs function
#include <opencv2/objdetect.hpp>  // Add this line for CascadeClassifier
#include <cctype> // Include the header for tolower
#include <cstdlib>
#include <string> // Include the header for string
#include <iostream>

using namespace cv;
using namespace std;

//Function Declarations
Mat sepiafilter(Mat src); //Sepia Filter
int detectFaces( cv::Mat grey, std::vector<cv::Rect> &faces); //Face Detection Function
int drawBoxes( cv::Mat frame, std::vector<cv::Rect> &faces, int minWidth, float scale); //Function to draw boxes around the detected face

int main()
{
    string img_address = "Sample.jpg";
    Mat img_file = imread(img_address, IMREAD_COLOR);
    Mat original_img = img_file;
    Mat grey;

    vector<Rect> faces;
    Rect last(0, 0, 0, 0);
    
    imshow("Display window", img_file);
    string caption;
    
    for(;;)
    {
        int keypress = waitKey(0); // Wait for a keystroke in the window
        if( keypress == 'q') //Exit the program if q is pressed.
        {
            break;
        }
        else if(tolower(keypress) == 'a') //Open the below website if a or A is pressed. 
        {
            // Replace "https://www.example.com" with the URL of the website you want to open
            const char* websiteURL = "http://www.codebind.com/cpp-tutorial/install-opencv-ubuntu-cpp/";

            // Use the below command to open the default web browser on Linux
            const char* command = "xdg-open";

            // Convert const char* to std::string for concatenation
            string fullCommand = string(command) + " " + string(websiteURL);

            // Execute the command
            system(fullCommand.c_str());
        }

        else if(keypress == 's') //Extension: Save any modified images and let the user add captions to the saved images
        {   
            // Prompt the user for a caption
            cout << "Enter caption for the image: ";
            
            getline(cin, caption);

            // Create a copy of the image for adding the caption
            Mat img_with_caption = img_file.clone();

            // Add the caption to the image
            putText(img_with_caption, caption, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

            // Save the image with the caption
            imwrite("Image_Saved.jpg", img_with_caption);
            
            //imwrite("Image_Saved.jpg", img_file);
            cout << "Image saved with caption: " << caption << endl;
        }
            
        else if(keypress == 'k')
        {
            img_file = original_img;
            imshow("Display window", img_file);
        }
        else if(keypress == 'g') //Extension: Apply Greyscale Filter on the image. 
        {   
            cvtColor(original_img, img_file, COLOR_BGR2GRAY); 
            imshow("Display window", img_file);
        }
        else if(keypress == 'i') //Extension: Apply Sepia Filter on the image. 
        {   
            img_file = sepiafilter(original_img); 
            imshow("Display window", img_file);
        }
        else if(keypress == 'f') //Extension: Face Detection on the image. 
        {   
            // convert the image to greyscale
            cvtColor( original_img, grey, COLOR_BGR2GRAY, 0);

            // detect faces
            detectFaces( grey, faces );

            // draw boxes around the faces
            drawBoxes( original_img, faces, 50, 1.0);

            // add a little smoothing by averaging the last two detections
            if( faces.size() > 0 ) {
            last.x = (faces[0].x + last.x)/2;
            last.y = (faces[0].y + last.y)/2;
            last.width = (faces[0].width + last.width)/2;
            last.height = (faces[0].height + last.height)/2;
            }

            img_file = original_img;

            imshow("Display window", img_file);
        }
    }

    return 0;
}


//Function definitions

//Sepia Filter
Mat sepiafilter(Mat src)
{
    {
    if (src.empty()) {
        {   cerr<<"Error: source image is empty";
        }; 
    }

    // Create a new Mat for the destination image
    Mat dst = Mat(src.rows, src.cols, CV_8UC3);

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

    return dst; // Success
    }
}


//Face Detection code
int detectFaces( cv::Mat grey, std::vector<cv::Rect> &faces ) {
  // a static variable to hold a half-size image
  static cv::Mat half;
  
  // a static variable to hold the classifier
  static CascadeClassifier face_cascade;

  // the path to the haar cascade file
  static cv::String face_cascade_file(FACE_CASCADE_FILE);

  if( face_cascade.empty() ) {
    if( !face_cascade.load( face_cascade_file ) ) {
      printf("Unable to load face cascade file\n");
      printf("Terminating\n");
      exit(-1);
    }
  }

  // clear the vector of faces
  faces.clear();
  
  // cut the image size in half to reduce processing time
  cv::resize( grey, half, cv::Size(grey.cols/2, grey.rows/2) );

  // equalize the image
  cv::equalizeHist( half, half );

  // apply the Haar cascade detector
  face_cascade.detectMultiScale( half, faces );

  // adjust the rectangle sizes back to the full size image
  for(int i=0;i<faces.size();i++) {
    faces[i].x *= 2;
    faces[i].y *= 2;
    faces[i].width *= 2;
    faces[i].height *= 2;
  }

  return(0);
}

/* Draws rectangles into frame given a vector of rectangles
   
   Arguments:
   cv::Mat &frame - image in which to draw the rectangles
   std::vector<cv::Rect> &faces - standard vector of cv::Rect rectangles
   int minSize - ignore rectangles with a width small than this argument
   float scale - scale the rectangle values by this factor (in case frame is different than the source image)
 */
int drawBoxes( cv::Mat frame, std::vector<cv::Rect> &faces, int minWidth, float scale  ) {
  // The color to draw, you can change it here (B, G, R)
  cv::Scalar wcolor(170, 120, 110);

  for(int i=0;i<faces.size();i++) {
    if( faces[i].width > minWidth ) {
      cv::Rect face( faces[i] );
      face.x *= scale;
      face.y *= scale;
      face.width *= scale;
      face.height *= scale;
      cv::rectangle( frame, face, wcolor, 3 );
    }
  }

  return(0);
}
