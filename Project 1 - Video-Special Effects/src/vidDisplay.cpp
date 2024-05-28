/*
  Ruchik Jani (NUID - 002825482)
  Anuj Patel (NUID - 002874710)
  Spring 2024
  CS 5330 Computer Vision

  This program is the main source code where we are implementing different filters and effects on a Webcam video stream input by pressing different keys. 
*/


#include <iostream> //Include for standard input & output: cout, cin
#include <opencv4/opencv2/core.hpp> //Include for Size, Mat
#include <opencv4/opencv2/highgui.hpp> //Include for namedWindow, imshow, waitKey, imwrite, VideoWriter functions
#include <opencv4/opencv2/imgproc.hpp> // Include for cvtColor function, convertScaleAbs function
#include <opencv4/opencv2/videoio.hpp> // Include for VideoWriter
#include <filters.h> //Include for different filter functions
#include <faceDetect.h> //Include for face detection functions
#include <sys/time.h>  //Include for getTime function
#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) 
{   
    VideoCapture *cameravid;

    // open the video device
    cameravid = new VideoCapture(0);
    if (!cameravid->isOpened()) 
    {
        cout << "Unable to open video device" << endl;
        return -1;
    }

    // get some properties of the image
    Size Image_Prop((int)cameravid->get(CAP_PROP_FRAME_WIDTH), (int)cameravid->get(CAP_PROP_FRAME_HEIGHT));
    cout << "Expected size: " << Image_Prop.width << " " << Image_Prop.height << endl;

    const int Ntimes = 10; // Declare Ntimes variable

    namedWindow("Video Frame", 1); // identifies a window
    Mat video_frame, filter_frame, greyscale_frame, sepia_frame, grey, embossed_frame;
    Mat sobelX, sobelY, magnitudeOutput, coloredMagnitude;

    vector<Rect> faces;
    Rect last(0, 0, 0, 0);

    char last_keypress = ' '; // Initialize with a space
    
    double contrast; //Adjust contrast (1.0 means no change)
    int brightness; //Adjust brightness (positive value increases brightness)
    bool saveFrame = false;

    // Initialize VideoWriter
    VideoWriter videoWriter("/home/rj/Project-1/Extensions/Video/data/output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Image_Prop);
    
    greyconvert objec1; //Custom greyscale function object declaration
    sepiaconvert objec2; //Custom sepia function object declaration

    for (;;) 
    {
        *cameravid >> video_frame; // get a new frame from the camera, treat as a stream

        if (video_frame.empty()) {
            cout << "video_frame is empty" << endl;
            break;
        }
        
        if (last_keypress == ' ') 
        {
            imshow("Video Frame", video_frame);
        }

        // Convert the image based on the last keypress
        
        if (last_keypress == 'g') 
        {   
            cvtColor(video_frame, video_frame, COLOR_BGR2GRAY); // Convert to greyscale
            imshow("Video Frame", video_frame);
            //cvtColor(video_frame, filter_frame, COLOR_BGR2GRAY); // Convert to greyscale
            //imshow("Video Frame", filter_frame);
        }

        // Call the alternate greyscale filter
        if (last_keypress == 'h') {
            objec1.setSourceImage(video_frame);
            objec1.greyscale();
            video_frame = objec1.getDestinationImage();
            imshow("Video Frame", video_frame);
            //greyscale_frame = objec1.getDestinationImage();
            //imshow("Video Frame", greyscale_frame);
        }

        // Call the sepia filter 
        if (last_keypress == 'i') {
            objec2.setSourceImage(video_frame);
            objec2.sepiafilter();
            video_frame = objec2.getDestinationImage();
            imshow("Video Frame", video_frame);
            //sepia_frame = objec2.getDestinationImage();
            //imshow("Video Frame", sepia_frame);
        }
        
        // Call the 3x3 Sobel X filter
        if (last_keypress == 'x') {
            Mat sobelXResult;
            sobelX3x3(video_frame, sobelXResult);
            convertScaleAbs(sobelXResult, video_frame);
            imshow("Video Frame", video_frame);
        }
        
        // Call the 3x3 Sobel Y filter
        if (last_keypress == 'y') {
            Mat sobelYResult;
            sobelY3x3(video_frame, sobelYResult);
            convertScaleAbs(sobelYResult, video_frame);
            imshow("Video Frame", video_frame);
        }

        // Face Detection code
        if (last_keypress == 'f') {
            // convert the image to greyscale
            cvtColor( video_frame, grey, COLOR_BGR2GRAY, 0);

            // detect faces
            detectFaces( grey, faces );

            // draw boxes around the faces
            drawBoxes( video_frame, faces );

            // add a little smoothing by averaging the last two detections
            if( faces.size() > 0 ) {
            last.x = (faces[0].x + last.x)/2;
            last.y = (faces[0].y + last.y)/2;
            last.width = (faces[0].width + last.width)/2;
            last.height = (faces[0].height + last.height)/2;
            }

            // display the frame with the box in it
            imshow("Video Frame", video_frame);
        }

        // Grey Background, Colored Face
        if (last_keypress == 'c') {
            video_frame = colorface(video_frame, faces);
            // Display the result
            imshow("Video Frame", video_frame);
        }

        // Embossing effect filter using SobelX and SobelY filters
        if (last_keypress == 'e') {
            // Apply embossing effect
            embossEffect(video_frame, embossed_frame, 0.7071, 0.7071);
            // Display the result
            imshow("Video Frame", embossed_frame);
        }

        // Adjust brightness and contrast
        if (last_keypress == 'a') {
            cout<<"Enter contrass & brightness: ";
            cin>>contrast>>brightness;
            adjustBrightnessAndContrast(video_frame, contrast, brightness); 
            imshow("Video Frame", video_frame); 
        }
        
        // Save the current frame to VideoWriter
        if (saveFrame)
        {
            videoWriter.write(video_frame);
        }

        //Magnitude Gradient Image Filter
        if (last_keypress == 'm') {
            
            cvtColor(video_frame, grey, COLOR_BGR2GRAY); // Convert to greyscale
            
            cv::Sobel(grey, sobelX, CV_16S, 1, 0);
            cv::Sobel(grey, sobelY, CV_16S, 0, 1);

            magnitude(sobelX, sobelY, magnitudeOutput);

            // Convert the magnitude image to a colorful representation
            cv::applyColorMap(magnitudeOutput, coloredMagnitude, cv::COLORMAP_JET);

            // Combine the original frame with the colored magnitude image
            cv::addWeighted(video_frame, 0.7, coloredMagnitude, 0.3, 0, video_frame);

            // Display the result
            cv::imshow("Video Frame", video_frame);
        } 

        //Blur & Quantization Filter    
        if (last_keypress == 'l') {
            
            // Apply blur and quantization filter
            Mat blurQuantized;
            blurQuantize(video_frame, blurQuantized, 10);

            // Display the result
            imshow("Video Frame", blurQuantized);
        } 
        
         // Gaussian Blur filter
        if (last_keypress == 'b') 
        {
            cout << "Choose Gaussian Blur function (1 or 2): ";
            int choice;
            cin >> choice;

            double startTime, endTime, difference;

            if (choice == 1) {
                startTime = getTime();
                for (int i = 0; i < Ntimes; i++) {
                    blur5x5_1(video_frame, filter_frame);
                }
                endTime = getTime();
                difference = (endTime - startTime) / Ntimes;

                // Display timing information
                printf("Time per frame (blur5x5_1): %.4lf seconds\n", difference);
            } else if (choice == 2) {
                startTime = getTime();
                for (int i = 0; i < Ntimes; i++) {
                    blur5x5_2(video_frame, filter_frame);
                }
                endTime = getTime();
                difference = (endTime - startTime) / Ntimes;

                // Display timing information
                printf("Time per frame (blur5x5_2): %.4lf seconds\n", difference);
            } else {
                cout << "Invalid choice. Please choose 1 or 2." << endl;
                }

            // Display the result
            imshow("Video Frame", filter_frame);
        }

        // see if there is a waiting keystroke
        char keypress = waitKey(1); // Reduce the delay for more responsive interaction

        switch (keypress) 
        {
            case 'q': goto exit; // exit the loop if 'q' is pressed

            case 's': imwrite("/home/rj/Project-1/Extensions/Video/data/Image_Saved.jpeg", video_frame); saveFrame = false; break;

            case 'g': last_keypress = 'g'; saveFrame = false; break; // OpenCV CVTColor greyscale function

            case 'h': last_keypress = 'h'; saveFrame = false; break; // Custom greyscale filter
            
            case 'i': last_keypress = 'i'; saveFrame = false; break; // Sepia filter

            case 'x': last_keypress = 'x'; saveFrame = false; break; // SobelX filter

            case 'y': last_keypress = 'y'; saveFrame = false; break; // SobelY filter

            case 'f': last_keypress = 'f'; saveFrame = false; break; // Face Detection Activated
            
            case 'c': last_keypress = 'c'; saveFrame = false; break; // Face is colorful, the rest of the image is grey. 

            case 'b': last_keypress = 'b'; saveFrame = false; break; // Gaussian Blur Filters 

            case 'e': last_keypress = 'e'; saveFrame = false; break; // Embossing Effect Filter

            case 'a': last_keypress = 'a'; saveFrame = false; break; // Adjust brightness & contrast by pixel-wise modification

            case 'm': last_keypress = 'm'; saveFrame = false; break; // Gradient Magnitude Image Filter

            case 'l': last_keypress = 'l'; saveFrame = false; break; // Blurred Quantization Image Filter

            case 'v': saveFrame = true; break; // Extension: Set the flag to save the frame, and then save the video frame of a particular filter effect. 

            case 'k': last_keypress = ' '; saveFrame = false; break; // Reset to normal color video

        }
    }
    
    exit:
    delete cameravid;
    return 0;
}