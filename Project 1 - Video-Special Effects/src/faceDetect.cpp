/*
  Ruchik Jani (NUID - 002825482)
  Anuj Patel (NUID - 002874710)
  Spring 2024
  CS 5330 Computer Vision

  Functions for finding faces and drawing boxes around them, and a function added by us as part of Task-11 which makes the face colorful, while the rest of the image is greyscale.

  The path to the Haar cascade file is define in faceDetect.h
*/

#include<iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <faceDetect.h>


/*
  Arguments:
  cv::Mat grey  - a greyscale source image in which to detect faces
  std::vector<cv::Rect> &faces - a standard vector of cv::Rect rectangles indicating where faces were found
     if the length of the vector is zero, no faces were found
 */
int detectFaces( cv::Mat &grey, std::vector<cv::Rect> &faces ) {
  // a static variable to hold a half-size image
  static cv::Mat half;
  
  // a static variable to hold the classifier
  static cv::CascadeClassifier face_cascade;

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
int drawBoxes( cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth, float scale  ) {
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


/*This function makes the face colorful, while the rest of the image is greyscale. The return value is a combined frame which represents this effect. 
  The arguments are the webcam video stream (frame), and a vector (faces) used to store rectangles that represent the detected faces in an 
  image or video frame.*/
cv::Mat colorface(cv::Mat &frame, std::vector<cv::Rect> faces)
{
    cv::Mat colored_face_frame, greyscale_frame, combined_frame;

    cv::Mat faceMask = cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::cvtColor(frame, greyscale_frame, cv::COLOR_BGR2GRAY);

    // Detect faces
    detectFaces(greyscale_frame, faces);

    // Create a mask for the face region
    for (const cv::Rect &face : faces)
    {
        rectangle(faceMask, face, cv::Scalar(255), cv::FILLED);
    }

    // Convert the original frame to a greyscale version
    cv::cvtColor(frame, greyscale_frame, cv::COLOR_BGR2GRAY);

    // Create a 3-channel greyscale frame
    cv::cvtColor(greyscale_frame, greyscale_frame, cv::COLOR_GRAY2BGR);

    // Set face region in the original frame to colored
    frame.copyTo(colored_face_frame, faceMask);

    combined_frame = colored_face_frame + greyscale_frame;

    return combined_frame;
}



