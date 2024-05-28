"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This file contains the code to analyze the network and analyze how it processes data by visualizing weights of the first layer by extracting the filters and their effect on an image.
"""


# import statements
import torch
import matplotlib.pyplot as plt
import cv2
from Task1AtoD import MyNetwork
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


#Visualize the filters of the first convolutional layer.
def visualize_filters(model):
    
    # Get the weights of the first layer (conv1)
    weights = model.conv1.weight

    # Print the shape of the filter weights
    print("Shape of weights:", weights.shape)

    # Determine the number of filters
    num_filters = weights.size(0)

    # Calculate the number of rows and columns for subplot grid
    num_rows = (num_filters - 1) // 4 + 1
    num_cols = min(num_filters, 4)

    # Visualize the filters using pyplot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 2))
    for i in range(num_filters):
        row = i // num_cols
        col = i % num_cols
        filter_weights = weights[i, 0].detach().cpu().numpy()  # Convert to numpy array
        axes[row, col].imshow(filter_weights)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        axes[row, col].set_title(f"Filter {i+1}")

    # Remove extra subplots if present
    for i in range(num_filters, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()
    return


#Load the first training example image.
def load_first_image(train_loader):
    with torch.no_grad():
        image, _ = next(iter(train_loader))
    return image.squeeze().numpy()


#Apply filters to the first training example image.
def apply_filters(model, image):
    image_np = image
    weights = model.conv1.weight
    filtered_images = []
    with torch.no_grad():
        # Apply each filter to the image using OpenCV's filter2D function
        for i in range(weights.size(0)):
            filter_weights = weights[i, 0].detach().cpu().numpy()
            filtered_image = cv2.filter2D(image_np, -1, filter_weights)
            filtered_images.append(filtered_image)
    return filtered_images


#Display the original image and filtered images.
def display_images(filtered_images, model):
    
    # Create a 5x4 grid of subplots
    fig, axes = plt.subplots(5, 4, figsize=(12, 10))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    weights = model.conv1.weight 

    num_filters = weights.size(0) 

    # Loop through num_filters
    for i in range(num_filters):
        # Plot the filter
        filter_img = weights[i, 0].detach().numpy()
        axes[2*i].imshow(filter_img, cmap='gray')
        axes[2*i].set_xticks([])
        axes[2*i].set_yticks([])
        axes[2*i].set_title(f"Filter {i+1}")

        # Plot the filtered image
        axes[2*i+1].imshow(filtered_images[i], cmap='gray')
        axes[2*i+1].set_xticks([])
        axes[2*i+1].set_yticks([])
        axes[2*i+1].set_title(f"Result {i+1}")

    # Remove any unused subplots
    for i in range(2*num_filters, len(axes)):
        fig.delaxes(axes[i])

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    # Display the plot
    plt.show()

    return


#This function performs tasks related to visualizing filters and applying them to an image using a trained network.
def main():
    # Load the trained network from file
    model = MyNetwork()
    model.load_state_dict(torch.load('./models/model.pth'))

    # Print the model architecture
    print(model)

    # Task-2A: Visualize filters
    visualize_filters(model)

    # Task-2B: Apply filters to an image
    # Create a DataLoader for the training dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Load the first training example image
    image = load_first_image(train_loader)

    # Apply filters to the first training example image
    filtered_images = apply_filters(model, image)

    # Display the original image and filtered images
    display_images(filtered_images, model)
    return

if __name__ == "__main__":
    main()


