
# Project 4: Calibration and Augmented Reality

## **OS & IDE**

OS used - Linux OS

IDE used for Code Development - VSCode

Program Execution - Terminal 

## **Running the executables**

### **Task-1 to 3**

To execute the program to calculate and save the camera calibration parameters to an XML file, the user has to type in the following commands in the linux Terminal to generate and run an executable:

```bash
$ g++ -std=c++17 -o cameracalibration  cameracalibration.cpp `pkg-config --cflags --libs opencv4`

$ ./cameracalibration
```

Thesea are the instructions for keypress:
- Press q - exit the program
- Press s - To save the calibration images to a specified folder

**Note:** The user also needs to update the path address of the XML file and the calibration images folder in the code as per their system. 
 
### **Task-4 to 7**

To execute the training system code the user has to type in the following commands in the linux Terminal to generate and run an executable:

```bash
$ g++ -std=c++17 -o virtualobject  virtualobject.cpp `pkg-config --cflags --libs opencv4`

$ ./virtualobject
```

Thesea are the instructions for keypress:
- Press q - exit the program
- Press x - To toggle the display of 3-D axes on the target in video feed
- Press p - To toggle the display of a virtual object overlayed on the target in video feed
- Press h - To toggle the display of harris corner detection on the target in video feed

**Note:** The user also needs to update the path address of the XML file in the code as per their system. 

## **Running the extensions**

### **Extension-1: Test out different cameras**

This extension has been implemented in the Task-1 to 3 program. The code will ask the user to select between webcam and mobile phone camera stream inputs for camera calibration. When the user presses 1, webcam is selected. When 2 is pressed, video feed from the phone is selected. Both of these options can be implemented for comparison of the quality of calibration results. The user needs to update the URL address of the mobile phone webcam streaming application as per the latest IP address. The mobile phone as well as the laptop running the code should be connected to WiFi. 

### **Extension-2: Insert virtual objects in a pre-recorded video**

This extension has been implemented in the Task-4 to 7 program. The code will ask the user to select between mobile phone and pre-recorded video stream. When the user presses 1, mobile phone is selected. When 2 is pressed, the pre-recorded video is selected as video stream input. The user can then press x to project 3D axes, p to insert a pyramid or press h for Harris corner detection on the pre-recorded video. The user needs to update the file path of this video file. 

## **Demonstration Video**

The demonstration video of this system is taken for 2 types of video streams:
- **Mobile Phone** 
- **Pre-recorded Video**

First, I have taken the video using mobile phone, and then, implemented the code on the pre-recorded video. 

https://drive.google.com/file/d/1gHle0tYBz8ALc4jjy4D0i3S_HfKWoQNS/view?usp=sharing

