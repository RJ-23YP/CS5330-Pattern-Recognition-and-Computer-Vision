"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This program contains the codes to analyze the first layer of the ResNet-18 network. 
"""

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models


# Load the pre-trained ResNet-18 model from PyTorch's model zoo
pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


# Visualizes the filters of the first convolutional layer of the provided model.
def visualize_filters(model):
    # Access the weights of the first convolutional layer and prepare for visualization
    weights = pretrained_model.conv1.weight.data

    # Print the shape of the filter weights
    print("Shape of weights:", weights.shape)

    num_filters = weights.size(0)

    # Determine the layout of the subplots
    num_rows = (num_filters - 1) // 4 + 1
    num_cols = min(num_filters, 4)

    # Create a figure with subplots to display each filter
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 2))
    for i in range(num_filters):
        row = i // num_cols
        col = i % num_cols
        # Separate filter weights for each RGB channel
        filter_weights = weights[i].detach().cpu().numpy()
        # Transpose the filter weights to match the shape expected by imshow
        filter_weights = filter_weights.transpose(1, 2, 0)
        # Min-max normalization
        filter_weights = (filter_weights - filter_weights.min()) / (filter_weights.max() - filter_weights.min())
        axes[row, col].imshow(filter_weights)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        axes[row, col].set_title(f"Filter {i+1}")
    for i in range(num_filters, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])
    plt.tight_layout()
    plt.show()


# Loads and transforms the first image from the MNIST dataset to match the input requirements of ResNet-18.
def load_and_transform_image(train_loader):
    with torch.no_grad():
        # Extract the first batch of images
        for images, _ in train_loader:
            # Convert the tensor back to a PIL image to apply transformations that expect a PIL image
            image = transforms.ToPILImage()(images[0])

            # Define transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to match ResNet-18's input dimensions
                transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet norms
            ])
            
            # Apply the transformations to the PIL image
            image = transform(image)
            return image.unsqueeze(0)  # Add batch dimension and return


# Applies the first convolutional layer of the model to the image tensor.
def apply_filters(model, image_tensor):
    # Ensure model is in evaluation mode.
    model.eval()
    
    # Extract the first convolutional layer from the model.
    first_conv_layer = model.conv1
    
    # Apply the first convolutional layer to the image tensor.
    with torch.no_grad():
        # Expected shape of output: (1, num_filters, H', W')
        filtered_images_tensor = first_conv_layer(image_tensor)
        
    # Convert the tensor of filtered images to a list of NumPy arrays for visualization.
    filtered_images = []
    num_filters = filtered_images_tensor.size(1)
    for i in range(num_filters):
        # Convert each filtered image to a NumPy array.
        # Squeeze to remove batch dimension and use detach() to remove from computation graph.
        # Move the tensor to CPU memory with cpu(), if not already, before converting to NumPy.
        filtered_image = filtered_images_tensor[0, i].cpu().detach().numpy()
        filtered_images.append(filtered_image)
    
    return filtered_images 


# Displays the original filters and the images resulting from applying these filters.
def display_images(filtered_images, pretrained_model):
    fig, axes = plt.subplots(5, 4, figsize=(12, 10))
    axes = axes.flatten()

    # Display each filter and its corresponding filtered image
    weights = pretrained_model.conv1.weight 
    num_filters = weights.size(0) 
    for i in range(min(2*num_filters, len(axes))):  # Corrected loop range
        if i % 2 == 0:  # For filters
            filter_img = weights[i//2, 0].detach().numpy()
            axes[i].imshow(filter_img, cmap='gray')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f"Filter {i//2 + 1}")
        else:  # For filtered images
            filtered_img_idx = i // 2
            axes[i].imshow(filtered_images[filtered_img_idx], cmap='gray')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f"Result {filtered_img_idx + 1}")
    
    # Remove any unused subplots
    for i in range(2*num_filters, len(axes)):
        fig.delaxes(axes[i])
    
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.show()


# Main functi# Main function to orchestrate the visualization processon
def main():

    #Print the network architecture
    print(pretrained_model)

    visualize_filters(pretrained_model)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load and transform the first image from the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    image = load_and_transform_image(train_loader)

    # Apply the first convolutional layer's filters to the image
    filtered_images = apply_filters(pretrained_model, image)

    # Display the filters and their effects on the image
    display_images(filtered_images, pretrained_model)

if __name__ == "__main__":
    main()