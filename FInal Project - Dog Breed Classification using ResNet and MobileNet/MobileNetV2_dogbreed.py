"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This script uses deep learning models to detect dogs in images or videos and predict their breeds.
"""
# Import statements
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torchvision.models
import torch.nn.functional as F
import sys

# Defines a neural network class Net which uses a MobileNetV2 model as a base for dog breed classification.
class Net(torch.nn.Module):
    """A network that adapts MobileNetV2 for the task of classifying dog breeds."""
    def __init__(self, base_model, num_classes):
        super(Net, self).__init__()
        self.base_model = base_model # Base model, typically pretrained.
        self.base_out_features = 1280 * 7 * 7 # Calculate output features based on MobileNetV2's last layer.
        self.linear1 = torch.nn.Linear(self.base_out_features, 1280) # # First fully connected layer.
        self.linear2 = torch.nn.Linear(1280, 512) ## Second fully connected layer
        self.output = torch.nn.Linear(512, num_classes)  # Output layer to classify breeds.


    def forward(self, x):
        # Forward pass through the network.
        x = self.base_model.features(x)  # Pass input through the base model.
        x = torch.flatten(x, 1)  # Flatten the output for the dense layer.
        x = F.relu(self.linear1(x))  # Apply ReLU activation function.
        x = F.relu(self.linear2(x))  # Apply ReLU activation function.
        x = self.output(x)  # Final output layer.
        return x

def load_model(model_path):
    """Loads the MobileNetV2 model with frozen layers and custom output layers for breed classification."""
    mobile_net = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in mobile_net.parameters():
        param.requires_grad = False
    model = Net(base_model=mobile_net, num_classes=120)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode.
    return model

def load_yolo_model():
    """Loads the YOLOv3 model configured for detecting objects, set specifically to detect dogs here."""
    config_path = '/home/newusername/PRCV/final_project/Yolo/yolov3.cfg'
    weights_path = '/home/newusername/PRCV/final_project/Yolo/yolov3.weights'
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # Set OpenCV as the backend.
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Set CPU as the target device.
    return net

# Prepare transformations for image input to the neural network.
def prepare_transform():
    # Resize, crop, convert to tensor, and normalize images.
    return transforms.Compose([
        transforms.Resize(256),  # Resize images to 256 pixels on the smallest side.
        transforms.CenterCrop(224),  # Crop a central square of 224 x 224 pixels.
        transforms.ToTensor(),  # Convert the image to a tensor.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet's mean and std.
    ])

# Detect dogs in a frame using YOLOv3 and draw bounding boxes around detected dogs.
def detect_dogs(net, frame, transform):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)  # Convert frame to blob format.
    net.setInput(blob)  # Set the blob as input to the network.
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outputs = net.forward(output_layers)  # Forward pass through the network.

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Scores for each class.
            class_id = np.argmax(scores)  # Class with the highest score.
            confidence = scores[class_id]  # Confidence for the class.
            if confidence > 0.5 and class_id == 16:  # If confidence > 50% and class is dog.
                # Calculate bounding box coordinates.
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                centerX, centerY, width, height = box.astype("int")
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Apply non-max suppression.
    if len(indices) > 0 and isinstance(indices, tuple):
        indices = indices[0]

    for i in indices:
        i = int(i)
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around each dog.

    return frame

# Additional functions handle video processing and image processing respectively.
def predict_breed_video(video_path, model, transform, label_encoder, net):
    # Processes video to detect and classify dog breeds.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow('Video with Predictions', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video with Predictions', 1920, 1080)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_dogs(net, frame, transform)  # Detect dogs in the frame.

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB.
        image = Image.fromarray(frame_rgb)  # Convert numpy array to PIL Image.
        image = transform(image)  # Apply transformations.
        image = image.unsqueeze(0)  # Add batch dimension.

        with torch.no_grad():
            output = model(image)  # Predict using the model.
            probabilities = F.softmax(output, dim=1)  # Calculate class probabilities.
            max_prob, preds = torch.max(probabilities, 1)  # Find the class with the highest probability.
            confidence = max_prob.item() * 100  # Convert probability to percentage.
            predicted_breed = label_encoder.inverse_transform(preds.numpy())[0]  # Get breed name.

        cv2.putText(frame, f'Match {confidence:.2f}% with {predicted_breed}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Video with Predictions', frame)  # Display frame with predictions.

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key.
            break

    cap.release()
    cv2.destroyAllWindows()  # Release resources.



def predict_breed_images(images_directory, model, transform, label_encoder):
    # Processes images to detect and classify dog breeds.
    for image_file in os.listdir(images_directory):
        image_path = os.path.join(images_directory, image_file)  # Get image path.
        image = Image.open(image_path)  # Open image.

        image = transform(image)  # Apply transformations.
        image = image.unsqueeze(0)  # Add batch dimension.

        with torch.no_grad():
            output = model(image)  # Predict using the model.
            probabilities = F.softmax(output, dim=1)  # Calculate class probabilities.
            max_prob, preds = torch.max(probabilities, 1)  # Find the class with the highest probability.
            confidence = max_prob.item() * 100  # Convert probability to percentage.
            predicted_breed = label_encoder.inverse_transform(preds.numpy())[0]  # Get breed name.

        print(f'{image_file}: {predicted_breed} with {confidence:.2f}% confidence')  # Print prediction result.


# Main function that orchestrates the processing based on user input.
def main(argv):
    MODEL_PATH = '/home/newusername/PRCV/final_project/MODEL/v2model.pth'
    LABEL_CSV_PATH = '/home/newusername/PRCV/final_project/Data/kaggle/labels.csv'

    model = load_model(MODEL_PATH)  # Load the breed classification model.
    label_df = pd.read_csv(LABEL_CSV_PATH)  # Load breed labels.
    label_encoder = LabelEncoder().fit(label_df['breed'])  # Encode labels.

    transform = prepare_transform()  # Prepare image transformations.
    yolo_net = load_yolo_model()  # Load the YOLO model.

    choice = input("Would you like to process a video or images? Enter 'video' or 'images': ").lower().strip()
    if choice == 'video':
        VIDEO_PATH = '/home/newusername/PRCV/final_project/video/4x.mp4'
        predict_breed_video(VIDEO_PATH, model, transform, label_encoder, yolo_net)  # Process video.
    elif choice == 'images':
        IMAGES_DIRECTORY = '/home/newusername/PRCV/final_project/dog_1o'
        predict_breed_images(IMAGES_DIRECTORY, model, transform, label_encoder)  # Process images.
    else:
        print("Invalid input. Please enter 'video' or 'images'.")

if __name__ == "__main__":
    main(sys.argv)  # Execute the main function with system arguments.

