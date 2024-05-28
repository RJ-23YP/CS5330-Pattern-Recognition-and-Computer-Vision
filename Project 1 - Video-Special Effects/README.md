
# Project 1 - Video: Special Effects

## **Project Description**

The aim of this poject is to develop a video-special effects application using C++ and OpenCV. The application will support reading and displaying images and live video, applying various real-time image processing effects, and responding to user inputs. The key tasks include implementing filters for grayscale, custom grayscale, sepia tone, blur, and edge detection (Sobel filters). Additionally, the project will detect faces in video streams and integrate three custom video effects. The project also includes timing analysis and performance optimization for some filters, providing hands-on experience with OpenCV’s image manipulation capabilities and enhancing understanding of real-time video processing.

## **OS & IDE**

OS used - Linux OS

IDE used for Code Development - VSCode

Program Execution - Terminal 

## **Instructions to run the code**

Our code is set up so that there are bin, include, obj, data, and src subdirectories in the project directory. All .cpp files are in src.  All .h or .hpp files are in include. All executables are written by the makefile to the bin folder. All .o files (objects) for the source codes are written by the makefile to the obj folder. The makefile is in the project directory. The data folder consists of the images used as inputs in the code, or images/videos saved from the code. 

### **Task-1**

Go to the *src* folder where the *imgDisplay.cpp* file is located, and run the following commands in the terminal to execute the code:

```bash
g++ -o img imgDisplay.cpp `pkg-config --cflags --libs opencv4`
./img
```

The first command compiles the *image_retrieval.cpp* which is the C++ source code and generates an executable named *img* in the same folder. The second command will run the executable file and the terminal will prompt the user to press a key after displaying the image. 

**Note:** The user needs to update the path address of the image to be displayed in the code. The *haarcascade_frontalface_alt2.xml* is to be included in the same folder as the executable for face detection. Change the path of the file in the code accordingly. 

### **Task-2 to 11**

Go to the project directory then type in the following commands in the Linux Terminal to create an executable in the bin folder:

```bash
make
cd bin
./vid 
```
The first command compiles the *vidDisplay.cpp*, *faceDetect.cpp*, and *filters.cpp* which are C++ source codes in conjunction with the *faceDetect.h* and *filters.h* header files, and generates an executable named *vid* in the *bin* folder. The second command changes the directory to the bin folder which contains the exeecutable. The third command will run the executable file and the terminal will prompt the user to press a key in order to apply the desired effect on the video stream (filter or face detection). There are also keys which the user can press to save a frame, or adjust the birghtness and contrast of the video. 

**Note:** The frames captured from the video, as well as video clips will be saved to the *data* folder by default in the code. The user needs to change the path address of this folder in the code as per their system. The *haarcascade_frontalface_alt2.xml* is to be included in the *bin* folder along with the executables for face detection. Change the path of the file in the *faceDetect.h* code accordingly. 

### **Extensions**

We have implemented the following extensions in this project:

- Implement the effects for still images and enable the user to save the modified images - This extension is carried out by following the Task-1 instructions. 
- Let the user save short video sequences with the special effects - This extension is carried out by following the Task-2 to 11 instructions. 
- Let the user add captions to images or video sequences when they are saved (i.e. create a meme generator) - This extension is carried out by following the Task-1 instructions. 


## **Demonstration video**

This is a demonstration video of some of the filters applied on a live webcam stream.

https://drive.google.com/file/d/1ihQoTMNheRd2vUc3zXfu2uD5yHZ1jsVj/view?usp=sharing


