"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This file contains the code for evaluating the network performance by running the trained model on the test set. 
"""


# import statements
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Task1AtoD import MyNetwork


#Evaluates the trained model on the test dataset.
def evaluate_model(model, test_loader):

    # Create a 3x3 grid of subplots to visualize images and predictions
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    # Iterate over the first 10 examples in the test set
    for i, (image, label) in enumerate(test_loader):
        if i >= 9:  # Only plot the first 9 examples
            break

        # Forward pass through the model to obtain predictions 
        output = model(image)
        prediction = torch.argmax(output, dim=1)

        # Plot the image and display prediction
        axes[i // 3, i % 3].imshow(image.squeeze().numpy(), cmap='gray')
        axes[i // 3, i % 3].set_title(f'Prediction: {prediction.item()}')   
        axes[i // 3, i % 3].axis('off')

        # Print output values and prediction details
        output_values = [f'{value:.2f}' for value in output.squeeze().detach().numpy()]
        print(f"Image {i+1}")
        print(f"Output values: {output_values}")
        max_index = torch.argmax(output)
        print(f"Index of max output value: {max_index.item()}. This value is located at postion {max_index.item() + 1} in the list of all output values for this image.")
        print(f"Predicted Label: {prediction.item()}, True Label: {label.item()}\n") 
        
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    return


# Main function to evaluate the trained model on the test set.
def main(argv):

    # Load the trained network from file and set the model to evaluation mode
    model = MyNetwork()
    model.load_state_dict(torch.load('./models/model.pth'))
    model.eval() 

    # Load the test dataset
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Call the function to evaluate the model
    evaluate_model(model, test_loader)
    return

if __name__ == "__main__":
    main(sys.argv)