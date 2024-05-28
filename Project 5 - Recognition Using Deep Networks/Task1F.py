"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This file contains the code for evaluating the network performance by running the trained model on a custom dataset which consists of handwritten digits.
"""


# import statements
import os
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image as PILImage  # Rename the PIL Image module to avoid conflict
from Task1AtoD import MyNetwork
from wand.image import Image as WandImage  # Rename the wand Image module to avoid conflict


# Resize images in the input directory and save them in the output directory.
def resize_images(input_dir, output_dir, width, height):
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all files in the input directory
    files = os.listdir(input_dir)

    # Iterate over each file in the input directory
    for file in files:
        # Construct the input and output file paths
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        # Open the image file
        with WandImage(filename=input_path) as img:  # Use WandImage for opening the image
            # Resize the image to 28x28
            img.resize(width, height)
            # Save the resized image using PIL's save method
            img.save(filename=output_path)
    
    return


# Preprocess resized images in the input directory.
def preprocess_images(input_dir, model):
   
    images = []
    predictions = []

    # List all files in the input directory
    files = os.listdir(input_dir)

    
    # Iterate over each file in the input directory
    for file in files:
        # Construct the input file path
        input_path = os.path.join(input_dir, file)

        # Load the image using PIL's Image module
        image = PILImage.open(input_path)

        # Convert the image to grayscale
        grayscale_image = image.convert("L")

        # Invert the intensities of the grayscale image
        inverted_image = PILImage.eval(grayscale_image, lambda x: 255 - x)

        # Convert the inverted image to a tensor
        image_tensor = torchvision.transforms.functional.to_tensor(inverted_image)

        # Normalize the image intensities to match MNIST data
        normalized_image = torchvision.transforms.functional.normalize(
            image_tensor, (0.1307,), (0.3081,)
        )

        # Add a batch dimension to the image tensor
        image_tensor = normalized_image.unsqueeze(0)

        # Run the image tensor through the network
        output = model(image_tensor)

        # Get the predicted label
        prediction = torch.argmax(output, dim=1).item()

        # Append image and prediction to lists
        images.append(inverted_image)
        predictions.append(prediction)

    return images, predictions


# Display images and their predicted labels in a 4x3 grid
def display_images(images, predictions):

    fig, axes = plt.subplots(4, 3, figsize=(10, 12)) # Create a 4x3 grid of subplots

    # Iterate over images and predictions
    for i in range(12):
        ax = axes[i // 3, i % 3]
        if i < len(images):
            ax.imshow(images[i], cmap="gray")
            ax.set_title(f"Prediction: {predictions[i]}")
            ax.axis("off")
        else:
            ax.axis("off")

    #Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    return


#Main function to execute the program by loading the model and then calling the resizing, prediciton and display functions
def main():
    
    # Load the trained network from file
    model = MyNetwork() 
    model.load_state_dict(torch.load('./models/model.pth'))
    model.eval()

    # Provide the directory of images which need to be resized
    input_dir = "Task1EtoF/Original Dataset" 
    # Store the resized images in this folder
    output_dir = "Task1EtoF/Task 1F"

    # Resize images in the input directory
    resize_images(input_dir, output_dir, 28, 28)

    # Preprocess resized images and get predictions 
    images, predictions = preprocess_images(output_dir, model)

    # Display images and predictions
    display_images(images, predictions)
    return

if __name__ == "__main__":
    main()


    
