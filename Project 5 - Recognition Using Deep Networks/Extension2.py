"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This program contains the codes to retrain the network using Gabor filters and analyze it's performance. 
"""


# import statements
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchviz import make_dot
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from Task1AtoD import train_model


# Function to generate Gabor filters using OpenCV
def gabor_generate(ksize, sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype)
    return kernel


# Neural network architecture defined with Gabor filters as the first layer
class MyNetworkWithGabor(nn.Module):
    def __init__(self, gabor_filters, num_fc_nodes): #Initialize the layers and parameters for the neural network model.
        super(MyNetworkWithGabor, self).__init__()
        self.gabor_filters = nn.Parameter(gabor_filters, requires_grad=False) # Set Gabor filters as a parameter
        self.conv2 = nn.Conv2d(gabor_filters.shape[0], 20, kernel_size=5) # Second convolutional layer
        # Adjust in_features to match the actual flattened size of the tensor
        self.fc1 = nn.Linear(20 * 12 * 12, num_fc_nodes)  # Corrected from the original calculation
        self.fc2 = nn.Linear(num_fc_nodes, 10) # Second fully connected layer (output layer)

    #Forward pass of the neural network
    def forward(self, x):
        x = F.conv2d(x, self.gabor_filters, padding=2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Ensure the view matches the actual tensor size before the fully connected layer
        x = x.view(-1, 20*12*12)  # Correctly match the flattened dimensions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Main function to execute the entire program workflow including data loading, model training, and visualization.
def main(argv):

    # Set random seed for reproducibility and disable cuDNN for deterministic results
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Create an instance of your network and visualize the architecture
    # model = MyNetwork()
    num_filters = 10
    gabor_filters = torch.zeros((num_filters, 1, 5, 5))
    for i, theta in enumerate(np.linspace(0, np.pi, num_filters, endpoint=False)):
        gabor_kernel = gabor_generate(ksize=5, sigma=2.0, theta=theta, lamda=10.0, gamma=0.5, psi=0)
        gabor_filters[i, 0] = torch.Tensor(gabor_kernel)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNetworkWithGabor(gabor_filters, num_fc_nodes=50)#.to(device)
    x = torch.randn(1, 1, 28, 28)#.to(device)  # Ensure the input tensor is on the same device as the model
    output = model(x)
    make_dot(output, params=dict(model.named_parameters())).render("Task-2 Output/gabor/gabor_network", format="png")

    # Task-3: Define data loaders, loss criterion, and optimizer
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform),
        batch_size=1000, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    modelsavepath = "./gabor_models/gabor.pth"
    optimizersavepath = "./gabor_models/optimizer.pth"
    epochs = 5
    
    # Train the model
    train_model(model, train_loader, test_loader, optimizer, criterion, epochs, modelsavepath, optimizersavepath)
    return

if __name__ == "__main__":
    main(sys.argv)
