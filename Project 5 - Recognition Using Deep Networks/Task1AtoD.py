"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This file contains the codes for loading the MNIST dataset, building a neural network, training the model and saving it to file.
"""


# import statements
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchviz import make_dot


# class definitions 
class MyNetwork(nn.Module): #Define your network architecture
    
    def __init__(self): #Initialize the layers and parameters for the neural network model.
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)
        
    #computes a forward pass for the network
    def forward(self, x): 
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x


#Plot the first six digits from the test dataset along with their corresponding labels.
def plot_first_six_digits(test_dataset):
    
    # Set up matplotlib figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    for i, ax in enumerate(axes.flat):
        # Get the i-th digit image and its corresponding label
        image, label = test_dataset[i]
        # Convert image to tensor and then squeeze to remove singleton dimensions
        image = torchvision.transforms.ToTensor()(image)
        # Plot the digit image
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    return


#Train the given model for the specified number of epochs. Plot the training and testing losses and accuracies.
def train_model(model, train_loader, test_loader, optimizer, criterion, epochs, modelsavepath, optimizersavepath):
    
     # Lists to store training and testing losses, and training and testing accuracies
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]
    train_accuracy = []
    test_accuracy = [] 

    # Run a test phase without training to initialize the test losses and accuracies
    testing_phase(model, test_loader, criterion, test_losses, test_accuracy) #Without this test run, you will get a dimensional error between x and y plots for test loss
    
    # Loop over epochs
    for epoch in range(1, epochs + 1):

        #Training phase
        training_phase(model, train_loader, optimizer, criterion, train_losses, train_accuracy, train_counter, epoch, modelsavepath, optimizersavepath)

        #Testing phase 
        testing_phase(model, test_loader, criterion, test_losses, test_accuracy)

    # Plot the training and testing losses
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Training Loss', 'Testing Loss'])
    plt.xlabel('Number of Training samples')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.show()

    # Plot the training and testing accuracies
    fig1 = plt.figure()
    plt.plot(train_counter, train_accuracy, color='#90EE90')
    plt.plot(test_counter, test_accuracy, color='#9B30FF')
    plt.legend(['Training Accuracy', 'Testing Accuracy'])
    plt.xlabel('Number of Training samples')
    plt.ylabel('Accuracy (%)')
    plt.show()
    return


# Performs one training epoch for a given model, storing losses, accuracy, and checkpoints.
def training_phase(model, train_loader, optimizer, criterion, train_losses, train_accuracy, train_counter, epoch, modelsavepath, optimizersavepath):
   
    model.train() # Set model to training mode
    log_interval = 10 # Define interval for logging progress
    total_correct = 0
    total_samples = 0

    # Iterate through each batch in the training data 
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # Clear gradients for each batch
        optimizer.zero_grad()

        # Forward pass and calculate loss 
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and parameter update for optimization
        loss.backward()
        optimizer.step()

        # Log training progres
        if batch_idx % log_interval == 0:
            '''print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))'''
        
            # Store training information for visualization
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                
            # Calculate and store training accuracy
            _, predicted = torch.max(output, 1)
            total_train = target.size(0)
            correct_train = (predicted == target).sum().item()
            train_accuracy.append(100 * correct_train / total_train)
            
            # Update total number of correct predictions and total number of samples
            total_correct += correct_train
            total_samples += total_train 

            # Save model checkpoint and optimizer state
            torch.save(model.state_dict(), modelsavepath)
            torch.save(optimizer.state_dict(), optimizersavepath)

    # Calculate average accuracy for the epoch
    epoch_accuracy = 100 * total_correct / total_samples
    
    # Print average accuracy for the epoch
    print('Train Epoch: {}\tAverage Accuracy: {:.2f}%'.format(epoch, epoch_accuracy)) 
    
    return
    

# Perform the testing phase for the given model, and then track & update testing losses and accuracies.
def testing_phase(model, test_loader, criterion, test_losses, test_accuracy):
    
    model.eval() # Set the model to evaluation mode

    # Initialize variables for test loss and correct predictions
    test_loss = 0
    correct = 0

    # Iterate through the test data loader
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += torch.sum(criterion(output, target)).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    
    # Calculate average test loss
    test_loss /= len(test_loader.dataset) 
    test_losses.append(test_loss)

    # Calculate and store testing accuracy
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
    test_accuracy.append(100 * correct / len(test_loader.dataset)) 

    # Print test set statistics
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return


# Main function to execute the entire program workflow including data loading, model training, and visualization.
def main(argv):

    # Set random seed for reproducibility and disable cuDNN for deterministic results
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Task-1: Load the MNIST test dataset and plot the first six example digits
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    plot_first_six_digits(test_dataset)

    # Task-2: Create an instance of your network and visualize the architecture
    model = MyNetwork()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    make_dot(output, params=dict(model.named_parameters())).render("/home/rj/Project-5/Task-2 Output/network_graph", format="png")

    # Task-3: Define data loaders, loss criterion, and optimizer
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                  transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                            ])),
        batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                            ])),
        batch_size=1000, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    modelsavepath = "./models/model.pth"
    optimizersavepath = "./models/optimizer.pth"
    epochs = 5
    
    # Train the model
    train_model(model, train_loader, test_loader, optimizer, criterion, epochs, modelsavepath, optimizersavepath)
    return

if __name__ == "__main__":
    main(sys.argv)
