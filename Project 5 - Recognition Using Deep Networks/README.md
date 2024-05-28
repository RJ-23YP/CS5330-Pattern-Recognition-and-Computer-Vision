
# Project 5: Recognition using Deep Networks

## **OS & IDE**

OS used - Linux OS

IDE used for Code Development - VSCode

Program Execution - Terminal/VSCode 

## **Running the executables**

### **Task-1A to 1D**

To execute the program to define the neural network, train, test and save the model, the user can either press the run button in VsCode IDE, or type in the following command in the linux terminal after changing the directory to the current folder:

```bash
python3 Task1AtoD.py
```

### **Task-1E**

To execute the program to evaluate the neural network on the test dataset loaded from MNIST digit dataset, the user can either press the run button in VsCode IDE, or type in the following command in the linux terminal after changing the directory to the current folder:

```bash
python3 Task1E.py
```

### **Task-1F**

To execute the program to evaluate the neural network on the a custom dataset created by us, the user can either press the run button in VsCode IDE, or type in the following command in the linux terminal after changing the directory to the current folder:

```bash
python3 Task1F.py
```

**Note:** The user also needs to update the path location of the input image directory where the original images are saved, and the path location of the output iamge directory where the resized images are saved in the code as per their system. 

### **Task-2**

To execute the program to analyse the first layer of the neural network, the user can either press the run button in VsCode IDE, or type in the following command in the linux terminal after changing the directory to the current folder:

```bash
python3 Task2.py
```

### **Task-3**

To execute the program to apply transfer learning on the greek letters dataset, the user can either press the run button in VsCode IDE, or type in the following command in the linux terminal after changing the directory to the current folder:

```bash
python3 Task3.py
```

**Note:** The user also needs to update the path location of the following directories in the code as per their system:

    1) Training and Testing Set folder
    2) Save location for the model weights and optimizer of the modified network
    3) Input and Output image directories for the training set images to be resized

Link for Additional Data: 

https://drive.google.com/file/d/1mQl0nExJlXiW-daIEH7bJe3nWXqrFKuP/view?usp=sharing

### **Task-4**

To execute the program to run experiment-1 & 2 to observe the effect of change in network architecture, the user can either press the run button in VsCode IDE, or type in the following command in the linux terminal after changing the directory to the current folder:

```bash
python3 Task4.py
```

## **Running the extensions**

### **Extension-1: Additional dimension explored in Task-4**

This extension has already been implemented in the Task-4 code as we have evaluated 4 dimensions along which to change the network architecture, instead of the 3 dimensions. 

### **Extension-2: Replace the first network layer with a Gabor Filter bank**

To execute the program to retrain the network using Gabor filters and analyze it's performance, the user can either press the run button in VsCode IDE, or type in the following command in the linux terminal after changing the directory to the current folder:

```bash
python3 Extension2.py
```

**Note:** The user also needs to update the path location of the model and optimizer save folders. 

### **Extension-3: Examine a different pre-trained network**

To execute the program to analyse the first layer of the ResNet-18 network, the user can either press the run button in VsCode IDE, or type in the following command in the linux terminal after changing the directory to the current folder:

```bash
python3 Extension3.py
```