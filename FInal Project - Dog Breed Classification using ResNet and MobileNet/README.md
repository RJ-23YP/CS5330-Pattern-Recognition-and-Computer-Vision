
# Final Project: Dog Breed Classification using ResNet and MobileNet

## **Project Description**

The aim of this project is to implement deep neural networks for the task of dog breed classification using the Stanford Dogs Dataset. The performance of three neural network architectures: ResNet18, ResNet50, and MobileNetV2 has been compared. The models have been trained on the dataset, which consists of 10,222 images across 120 dog breed categories. Additionally, the YOLOv3 object detection model has been incorporated for tracking dogs in video frames while predicting their breeds using the trained classification models. The models were evaluated on test images and videos, and their performance was analyzed in terms of accuracy, loss, and training time. 

## **Instructions to run the code**

### **Train the Model**
To train ResNet18 model, execute the following command:
```bash
Python3 ResNet18.py
```
To  train ResNet50 model, execute the following command:
```bash
Python3 ResNet50.py
```
To train MobileNetV2 model, execute the following command:
```bash
Python3 MobileNetV2.py
```
**Note:** The user needs to update the path location of the model and optimizer save folders. The user also needs to update the directory path where the dataset folders and labels.csv file are stored. 

### **Testing the Model**

To evaluate the performance of the ResNet18 model, execute the following command:
```bash
Python3 ResNet18_dogbreed.py
```
To evaluate the performance of the ResNet50 model, execute the following command:
```bash
Python3 ResNet50_dogbreed.py
```
To evaluate the performance of the MobileNetV2 model, execute the following command:
```bash
Python3 MobileNetV2_dogbreed.py
```
**Note:** The user needs to update the path location of the folders where the YOLOv3 weights and configuration files have been saved. The user also needs to update the model and labels.csv file folders. To provide the image or video dataset, the user needs to update the respective file locations. 

## **Presentation video**

Link to the presentation video:

https://drive.google.com/file/d/1QIcaOKeDKIco3ysKPfK51n6utqRqvDPE/view

Link to the PPT:

https://drive.google.com/file/d/18_vBMRj8C9vGqbtVDWYbcujHbe4JWpOc/view

## **Demonstration videos**

Link to a zip folder containing demonstration videos of our code on test dataset:

https://drive.google.com/file/d/1g-1EnMj9uu15pKE93gt1vGFwOTdvYYDI/view?usp=sharing


## **Dataset**

    1) Kaggle Dataset: 

    https://www.kaggle.com/c/dog-breed-identification/data?select=test

    2) Test Dataset (Images)
    
    https://drive.google.com/drive/folders/1xUGIQsOdMO_G9UeeqgrxRuw7GdLQHL1t

    3) Test Dataset (Videos)

    https://www.youtube.com/watch?v=fcyshDExRuQ

