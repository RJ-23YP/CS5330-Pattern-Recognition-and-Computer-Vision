
# Project-2: Content-Based Image Retrieval 

## **OS & IDE**

OS used - Linux OS

IDE used for Code Development - VSCode

Program Execution - Terminal 

## **Running the executables**

### **Task-1 to 7**

Our code is set up so that there are bin, include, obj, data, and src subdirectories in the project directory. All .cpp files are in src.  All .h or .hpp files are in include. All executables are written by the makefile to the bin folder. All .o files (objects) for the source codes are written by the makefile to the obj folder. The makefile is in the project directory. The data folder consists of the image dataset olympus used in our program. 

Go to the project directory then type in the following commands in the Linux Terminal to create an executable in the bin folder:

```bash
make

cd bin

./match 
```

The first command compiles the *image_retrieval.cpp* & *computes.cpp* which are C++ source codes in conjunction with the *computes.h* header file and generates an executable named *match* in the *bin* folder. The second command changes the directory to the bin folder which contains the exeecutable. The third command will run the executable file and the terminal will prompt the user to select which task is to be executed. The top N matches will be displayed as per the task.  

**Note:** The target image and directory paths have been hardcoded in the program. In order to run the code, the paths need to be changed accordingly.  
