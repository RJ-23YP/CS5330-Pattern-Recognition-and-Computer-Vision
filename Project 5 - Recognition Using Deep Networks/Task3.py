"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This file contains the code to implement transfer learning on greek letters dataset by modifying the network trained for digit recognition. 
"""


#import statements
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt 
import math
from Task1AtoD import MyNetwork, training_phase  # Import your MNIST network from Task 1
from Task1F import resize_images


# Defines a transformation class to preprocess images.
class GreekTransform: 
    def __init__(self):
        pass

    def __call__(self, x): # Preprocesses the input image by converting to grayscale, rotating, cropping to 28x28, and inverting colors.
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# Loads a pre-trained MNIST network, freezes its weights, and replaces the last layer for fine-tuning with new data.
def load_pretrained_model(model_path):
    # Instantiate the MNIST network
    model = MyNetwork()

    #Print the original network architecture
    print(model)
    
    # Load the pre-trained weights
    model.load_state_dict(torch.load(model_path))

    # Freeze the network weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer with a new Linear layer with three nodes
    model.fc2 = nn.Linear(50, 3)
    model.fc2.requires_grad = True #Unfreeze the weights of the new layer for model training

    return model


# Tests a trained model using a test dataset, calculates accuracy, and visualizes predictions on sample images.
def test_model(model, modelsavepath, test_loader):

    # Load the saved model weights
    model.load_state_dict(torch.load(modelsavepath))

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for accuracy calculation
    correct = 0
    total = 0

    # Iterate through the test loader
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            num_images = len(images)
            num_rows = min(3, math.ceil(num_images / 3))  # Calculate the number of rows for the grid
            fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))  # Create subplots

            for i, ax_row in enumerate(axes):
                for j, ax in enumerate(ax_row):
                    idx = i * 3 + j  # Calculate the index of the current image
                    if idx < num_images:  # Check if the index is within the number of images
                        image = images[idx]
                        ax.imshow(image.squeeze().numpy(), cmap='gray')
                        ax.set_title(f"Predicted: {predicted[idx].item()}, True: {labels[idx].item()}")
                        ax.axis('off')
                    else:
                        ax.axis('off')  # Turn off axis for empty subplot

            plt.show()
    
     # Calculate and print accuracy
    accuracy = correct / total * 100
    print(f"Test Set Accuracy: {accuracy:.2f}%")


# Main function to execute the training and testing of a model on the Greek letters dataset.
def main():
    # Path to the directory containing the folders alpha, beta, and gamma
    training_set_path = "Task3/greek_train/greek_train"
    testing_set_path = "Task3/greek_train/greek_test"

    # Create DataLoader for the Greek training dataset
    greek_train = DataLoader(
        datasets.ImageFolder(training_set_path,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              GreekTransform(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                          ])),
        batch_size=5,
        shuffle=True
    )
    
    # Load the pre-trained model
    model_path = "./models/model.pth"  # Adjust the path accordingly
    model = load_pretrained_model(model_path)

    # Print the modified model architecture
    print(model)

    #Define the loss function, optimizer and no. of epochs
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 15
    modelsavepath = "./greekdatamodel/model.pth"
    optimizersavepath = "./greekdatamodel/optimizer.pth"

    # Training loop for the Greek letters dataset
    # Lists to store training and testing losses, and training and testing accuracies
    train_losses = []
    train_counter = []
    train_accuracy = []
    # Loop over epochs
    for epoch in range(1, epochs + 1):
        #Training phase
        training_phase(model, greek_train, optimizer, criterion, train_losses, train_accuracy, train_counter, epoch, modelsavepath, optimizersavepath)

    # Plot the training loss
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Training Loss/Error'])
    plt.xlabel('Number of Training samples')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.show()


    #Create your custom test dataset and use the below three lines of code to resize the images
    # Provide the directory of images which need to be resized
    input_dir = "Task3/Resizer/Input" 
    # Store the resized images in this folder
    output_dir = "Task3/Resizer/Output"

    # Resize images to 128x128 for further processing
    resize_images(input_dir, output_dir, 128, 128)

    # Create DataLoader for the Greek test dataset
    greek_test = DataLoader(
        datasets.ImageFolder(testing_set_path,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              GreekTransform(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                          ])),
        batch_size=10,
        shuffle=False
    )
    
    # Call the test_model in your main function after training
    test_model(model, modelsavepath, greek_test) 

if __name__ == "__main__":
    main()
