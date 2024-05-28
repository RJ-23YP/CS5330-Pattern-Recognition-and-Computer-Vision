"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This file contains the code to optimize the network performance by experimentation with different dimensions along which we can change the network architecture.
"""


# Import statements
import random
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import matplotlib.pyplot as plt



#########################################################Experiment-1#################################################################
# Defines a convolutional neural network for image classification, specifically designed for the Fashion-MNIST dataset.
class FashionConvNet(nn.Module):
   def __init__(self, num_filters_conv1, kernel_size_conv1, num_filters_conv2, kernel_size_conv2):
       super(FashionConvNet, self).__init__()

       # Calculate output dimensions for linear layers
       self.calculate_linear_size(kernel_size_conv1, kernel_size_conv2, num_filters_conv2)  # Function call for readability
     
       # Define convolutional layers with ReLU activation and pooling
       self.conv_block = nn.Sequential(
           nn.Conv2d(1, num_filters_conv1, kernel_size=kernel_size_conv1),
           nn.MaxPool2d(2),
           nn.ReLU(),
           nn.Conv2d(num_filters_conv1, num_filters_conv2, kernel_size=kernel_size_conv2),
           nn.Dropout(0.5),  # Regularization to prevent overfitting
           nn.MaxPool2d(2),
           nn.ReLU()
       )
       # Define fully connected layers with ReLU activation
       self.fc_layer1 = nn.Linear(self.linear_size, 50)
       self.fc_layer2 = nn.Linear(50, 10) # Output for 10 classes in Fashion-MNIST

   # Forward pass of the model:
   def forward(self, x):
       x = self.conv_block(x)
       x = x.view(-1, self.linear_size)  # Flatten output
       x = F.relu(self.fc_layer1(x))
       x = F.log_softmax(self.fc_layer2(x), dim=1)
       return x
   
   # Calculates the output size for linear layers based on input image size, kernel sizes, and pooling.
   def calculate_linear_size(self, kernel_size_conv1, kernel_size_conv2, num_filters_conv2):
       self.after_pool1 = (28 - kernel_size_conv1 + 1) // 2
       self.after_pool2 = (self.after_pool1 - kernel_size_conv2 + 1) // 2
       self.linear_size = num_filters_conv2 * self.after_pool2 * self.after_pool2


# Trains the model for a given number of epochs and evaluates its performance on the test set.
def train_and_evaluate(train_loader, test_loader, model, loss_func, optimizer, epochs):

   test_losses = [] # Stores test loss after each epoch
   test_accuracies = [] # Stores test accuracy after each epoch

   # Call train loop and test loop for each epoch
   for epoch in range(epochs):
       print(f"Epoch {epoch+1}\n")
       train_loop(train_loader, model, loss_func, optimizer)
       test_loop(test_loader, model, loss_func, test_losses, test_accuracies)
   print("Training completed.")
   # Return the test results of the last epoch
   return test_losses[-1], test_accuracies[-1]


# Performs one epoch of training on the model.
def train_loop(data_loader, model, loss_func, optimizer):
   # Set model in training mode
    model.train()  # Set model to training mode
    for batch_idx, (data, target) in enumerate(data_loader):
        # Forward pass: compute predictions and loss
        predictions = model(data)
        loss = loss_func(predictions, target)

        # Backward pass: compute gradients and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


# Evaluates the model performance on the test set for one epoch.
def test_loop(data_loader, model, loss_func, test_losses, test_accuracies):
    model.eval()  # Set model to evaluation mode
    total_size = len(data_loader.dataset)  # Total number of samples in test set
    num_batches = len(data_loader)  # Number of batches in the test loader
    test_loss, correct = 0, 0

    # No gradient computation needed during evaluation
    with torch.no_grad():
        for data, target in data_loader:
            predictions = model(data)
            # Update test loss
            test_loss += loss_func(predictions, target).item()
            # Update correct predictions count
            correct += (predictions.argmax(1) == target).type(torch.float).sum().item()

    # Calculate and log test accuracy
    test_loss /= num_batches
    test_losses.append(test_loss)
    correct /= total_size
    test_accuracies.append(correct)
    print(f"Test Results: Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")
    return


# Helper function to draw graphs for the experiment
def visualize_results(results, filter_values, combinations):
   # There will be 9 combinations and bar width is 0.20
   indices = np.arange(9)
   bar_width = 0.20
   colors = ['red', 'yellow', 'green', 'blue',]

   # Generate legends for bars
   legends = []
   for value in filter_values:
       legends.append(f"L1: {value}, L2: {value*2}")

   # Plot different bars
   for i in range(4):
       plt.bar(indices + i * bar_width, results[i], bar_width, color=colors[i])

   # Add legends and labels
   plt.legend(legends, loc='upper right')
   plt.xticks(indices + bar_width, combinations)
   plt.ylabel('Loss Function')
   plt.show()
   return
#########################################################Experiment-1#################################################################



#########################################################Experiment-2#################################################################
class CustomNetwork(nn.Module):
   def __init__(self, dropout_prob, add_layer=False):
       super(CustomNetwork, self).__init__()
       # Define the main structure of the neural network model
       self.layer_stack = nn.Sequential(
           nn.Conv2d(1, 20, kernel_size=5), # First convolutional layer
           nn.MaxPool2d(2), # Max pooling layer
           nn.ReLU(), # ReLU activation function
           nn.Conv2d(20, 40, kernel_size=7), # Second convolutional layer
           nn.Dropout(dropout_prob), # Dropout layer with adjustable probability
           nn.MaxPool2d(2), # Max pooling layer
           nn.ReLU() # ReLU activation function
       )
       # Option to add an additional dropout layer
       self.add_layer = add_layer
       # Fully connected layers
       self.fc1 = nn.Linear(360, 50)
       self.fc2 = nn.Linear(50, 10)

   def forward(self, x):
       x = self.layer_stack(x) # Apply the convolutional layers
       x = x.view(-1, 360) # Flatten the tensor to a linear form
       # Decide whether to add an additional dropout layer after the first fully connected layer
       if self.add_layer:
           x = F.relu(F.dropout(self.fc1(x))) # Apply dropout after the first fully connected layer
       else:
           x = F.relu(self.fc1(x)) # No additional dropout
       x = F.log_softmax(self.fc2(x), dim=1) # Apply log_softmax after all layers
       return x


def display_graph(results, dropout_probs, ylabel):
   # Set up the indices for the bar plot
   ind = np.arange(10)
   width = 0.2
   colors = ['red', 'green']
   # Generate the legends for the bars
   legends = ["No Additional Layer", "Additional Layer"]
   # Plot the bars for each case
   for i in range(2):
       plt.bar(ind + i * width, results[i], width, color=colors[i])
   # Add legends and labels to the plot
   plt.legend(legends, loc='upper right')
   plt.xticks(ind + width, dropout_probs)
   plt.ylabel(ylabel)
   plt.show()
   return
#########################################################Experiment-2#################################################################



# This function defines the main training loop for hyperparameter tuning of a convolutional neural network on the Fashion-MNIST dataset. It will be used to first run Experiment-1, and then Experiment-2
def main():

    #Define the hyperparameters and load the training and test datasets for Experiment-1 & 2. 
    # Set random seed for reproducibility (helps ensure same results when running multiple times)
    random_seed = 47
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False  # Deterministic cudnn behavior (if applicable)

    # Training settings
    learning_rate = 0.01
    epochs = 5
    loss_func = nn.NLLLoss()  # Negative Log Likelihood loss for multi-class classification

    # Load and prepare data
    train_loader = DataLoader(
        datasets.FashionMNIST(
            root="fashion_data",
            train=True,
            download=True,
            transform=ToTensor()
        ),
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.FashionMNIST(
            root="fashion_data",
            train=False,
            download=True,
            transform=ToTensor()
        ),
        batch_size=64,
        shuffle=True
    )


    #Experiment-1: Size of filters and Number of filters
    # Hyperparameter options
    filter_sizes = [3, 5, 7]  # Options for kernel size of convolutional layers
    filter_numbers = [5, 10, 15, 20]  # Options for number of filters in the first layer

    # Containers for results (loss, accuracy, best configuration)
    size_losses = []  # Stores losses for each filter number (4 elements, each with 9 losses)
    size_accuracies = []  # Stores accuracies for each filter number ( مشابه size_losses) (similar to size_losses)
    best_loss = None  # Stores the best validation loss achieved so far
    best_combo = []  # Stores the hyperparameter combination (filters & kernel sizes) that achieved the best loss
    combos = []  # List of all possible combinations of kernel sizes (used for plotting)

    # Generate list of all kernel size combinations for later plotting
    for kernel_size_conv1 in filter_sizes:
        for kernel_size_conv2 in filter_sizes:
            combos.append(f"K{kernel_size_conv1}, K{kernel_size_conv2}")  # e.g., "K3, K5"

    # Main loop for hyperparameter tuning: trains the model with different filter and kernel sizes
    for filters_conv1 in filter_numbers:
        # Stores loss and accuracy for this filter number (array of length 9)
        losses = []
        accuracies = []

        # Iterates through all combinations of kernel sizes for the first and second convolutional layers
        for kernel_size_conv1 in filter_sizes:
            for kernel_size_conv2 in filter_sizes:
                # Creates a new model instance with the current hyperparameters
                model = FashionConvNet(filters_conv1, kernel_size_conv1, 2*filters_conv1, kernel_size_conv2)
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                # Trains and evaluates the model, recording loss and accuracy
                test_loss, test_accuracy = train_and_evaluate(train_loader, test_loader, model, loss_func, optimizer, epochs)
                losses.append(test_loss)
                accuracies.append(test_accuracy)

                # Tracks the best model configuration (filters & kernel sizes) based on minimum loss
                if best_loss is None or test_loss < best_loss:
                    best_loss = test_loss
                    best_combo = [filters_conv1, kernel_size_conv1, 2*filters_conv1, kernel_size_conv2]

        # Stores results (losses & accuracies) for this filter number
        size_losses.append(losses)
        size_accuracies.append(accuracies)

    # Visualizes the average loss and accuracy graphs for different filter numbers
    visualize_results(size_losses, filter_numbers, combos)
    visualize_results(size_accuracies, filter_numbers, combos)

    # Prints detailed information about the experiment
    print("Losses for each filter number configuration:")
    print(size_losses)
    print("Accuracies for each filter number configuration:")
    print(size_accuracies)
    print("Best model configuration (filters & kernel sizes) based on minimum loss:")
    print(best_combo)



    # Experiment 2: Investigating Dropout Layer Impact
    # Dropout Rate Options
    dropout_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Include/Exclude Dropout Layer Options
    include_dropout_layer = [False, True]

    # Containers for Results (Loss & Accuracy)
    experiment_losses = []  # Stores losses for each dropout layer inclusion (2 elements, each 10 losses)
    experiment_accuracies = []  # Stores accuracies for each dropout layer inclusion (similar to experiment_losses)
    best_loss_experiment = None  # Stores the overall best validation loss achieved
    best_config_experiment = []  # Stores hyperparameter combination (dropout rate & inclusion) for best loss

    # Main Loop for Experiment: Train with Different Parameter Combinations
    for include_layer in include_dropout_layer:
        # List to Store Experiment Results for Current Dropout Layer Inclusion (10 elements)
        experiment_loss_subset = []
        experiment_accuracy_subset = []

        # Iterate Through All Dropout Rates
        for current_dropout_rate in dropout_rates:
            # Create Model Instance with Current Configuration
            model_experiment = CustomNetwork(current_dropout_rate, include_layer)
            optimizer_experiment = torch.optim.SGD(model_experiment.parameters(), lr=learning_rate)

            # Train, Evaluate, and Record Results
            test_loss_experiment, test_accuracy_experiment = train_and_evaluate(train_loader, test_loader, model_experiment, loss_func, optimizer_experiment, epochs)
            experiment_loss_subset.append(test_loss_experiment)
            experiment_accuracy_subset.append(test_accuracy_experiment)

            # Track Best Performing Model Configuration (Lowest Loss)
            if best_loss_experiment is None or test_loss_experiment < best_loss_experiment:
                best_loss_experiment = test_loss_experiment
                best_config_experiment = [include_layer, current_dropout_rate]

        # Store Results for Current Dropout Layer Inclusion Option
        experiment_losses.append(experiment_loss_subset)
        experiment_accuracies.append(experiment_accuracy_subset)

    # Visualize Loss and Accuracy Graphs (functions not shown, likely for plotting)
    display_graph(experiment_losses, dropout_rates, "Loss Curve")
    display_graph(experiment_accuracies, dropout_rates, "Accuracy Curve")

    # Print Experiment Results
    print("Experiment Losses:")
    print(experiment_losses)
    print("Experiment Accuracies:")
    print(experiment_accuracies)
    print("Best Performing Configuration (Dropout Layer Inclusion & Rate):")
    print(best_config_experiment)
    return

if __name__ == "__main__":
   main()
