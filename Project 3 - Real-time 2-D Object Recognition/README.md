
# Project-3: Real-time 2-D Object Recognition 

## **OS & IDE**

OS used - Linux OS

IDE used for Code Development - VSCode

Program Execution - Terminal 

## **Running the executables**

### **Task-1 to 9 (except Task-5)**

Our code is set up so that there are bin, include, obj, and src subdirectories in the project directory. All .cpp files are in src.  All .h or .hpp files are in include. All executables are written by the makefile to the bin folder. All .o files (objects) for the source codes are written by the makefile to the obj folder. The makefile is in the project directory. 

Go to the project directory then type in the following commands in the Linux Terminal to create an executable in the bin folder:

```bash
make

cd bin

./recognize 
```

The first command compiles the *main_recognition.cpp*, *segmentation.cpp*, and *classification.cpp* which are C++ source codes in conjunction with the *segmentation.h* and *classification.h* header files, and generates an executable named *recognize* in the *bin* folder. The second command changes the directory to the bin folder which contains the executable. The third command will run the executable file and the terminal will prompt the user to select the classifier. If K-NN classifier is selected, it will ask the user to input the value of K. 

**Note:** The csv databse file path has been hardcoded in the program. In order to run the code, the path need to be changed accordingly.   

### **Task-5**

To execute the training system code the user has to type in the following commands in the linux Terminal to generate and run an executable:

```bash
g++ -std=c++17 -o training training_system.cpp `pkg-config --cflags --libs opencv4`

./training
```

**Note:** The user also needs to update the path of the csv database file and the training set folder in the code as per their system. 

## **Demonstration Video**

This is the link to the demonstration video of our system with a live video stream of the object captured from our phone. We were using the DroidCam mobile app to carry out this task. 

The first run of the program involves the Nearest Neighbour classifier, and in the second run, the program is executed with the K-Nearest Neighbour classifier:

https://drive.google.com/file/d/1mol3Z8MxhSMIhYUqEfuX8jcwVSrL_huF/view
