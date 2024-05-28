/*
  Ruchik Jani (NUID - 002825482)
  Anuj Patel (NUID - 002874710)
  Spring 2024
  CS 5330 Computer Vision

  Include file for faceDetect.cpp, face detection and drawing, and colored-face grey-background functions. 
*/

#ifndef FACEDETECT_H
#define FACEDETECT_H

// put the path to the haar cascade file here
#define FACE_CASCADE_FILE "/home/rj/Project-1/Extensions/Video/bin/haarcascade_frontalface_alt2.xml"

#include<opencv4/opencv2/opencv.hpp>
#include<iostream>

// prototypes
int detectFaces( cv::Mat &grey, std::vector<cv::Rect> &faces ); //Face Detection function
int drawBoxes( cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth = 50, float scale = 1.0  ); //Function to draw boxes around the detected face
cv::Mat colorface(cv::Mat &frame, std::vector<cv::Rect> faces); //colored-face grey-background functions

#endif
