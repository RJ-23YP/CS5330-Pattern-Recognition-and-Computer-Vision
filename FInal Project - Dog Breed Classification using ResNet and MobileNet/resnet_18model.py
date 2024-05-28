"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This file contains the code for training the ResNet 18 model and saving its weights. 
"""


# import statements
import sys
import torch 
import torchvision
import torch.nn.functional as F
from torchviz import make_dot
from mobilenetmodel import data_preprocess, create_data_loaders, train_model, visualize_training 


# Class Definitions 
# Define the neural network architecture
class CNN(torch.nn.Module):    
    # Initialize the neural network model.
    def __init__(self, base_model, num_classes):
        super(CNN, self).__init__()
        self.base_model = base_model # Store the base model for feature extraction
        self.base_out_features = base_model.fc.in_features # Get the number of output features from the base model's fully connected layer
        self.linear1 = torch.nn.Linear(self.base_out_features, 512) # First linear layer for feature transformation
        self.output = torch.nn.Linear(512, num_classes) # Final linear layer for classification
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)) # Adaptive average pooling layer to convert feature map to fixed size

    # Forward pass of the neural network
    def forward(self, x):
        # Feature extraction through convolutional layers
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.avgpool(x) # Global average pooling
        x = torch.flatten(x, 1) # Flatten the feature map
        # Fully connected layers for classification
        x = F.relu(self.linear1(x)) 
        x = self.output(x)
        return x


# Function Definitions
# Function for loading the model for further training
def create_model(num_classes):
    # Load the pre-trained ResNet18 model
    resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze the base model parameters
    for param in resnet18.parameters():
        param.requires_grad = False

    # Create the final model
    model = CNN(resnet18, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    return model, device


# Trains a dog breed identification model and saves it for future use.
def main(argv):
    # Parameters
    data_dir = '/home/rj/Final Project/Kaggle/dog-breed-identification'
    sample_size = None  # Set to a small number to test, or None for full dataset
    batch_size = 64
    num_epochs = 20

    # Load data
    frame_lbl = data_preprocess(data_dir, sample_size)

    # Create data loaders
    dataloader_train, dataloader_validation = create_data_loaders(frame_lbl, batch_size, val_split=0.05)

    # Create model
    model, device = create_model(num_classes=120)

    # Visualize the architecture
    x = torch.randn(1, 3, 224, 224).cuda()
    output = model(x)
    make_dot(output, params=dict(model.named_parameters())).render("/home/rj/Final Project/Kaggle/res18arc", format="png")

    # Loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Model Training Phase
    loss_train, loss_acc, loss_val, acc_val = train_model(
        dataloader_train, dataloader_validation, device, model,
        loss_fn, optimizer, num_epochs
    )

    # Save the trained model
    model_path = f'{data_dir}/res18models/newmodel.pth'
    optimizer_path = f'{data_dir}/res18models/newoptimizer.pth'
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    print(f"Model saved successfully at: {model_path}")
    print(f"Optimizer saved successfully at: {optimizer_path}")

    # Plot the accuracy and loss curves
    visualize_training(loss_train, loss_val, loss_acc, acc_val, num_epochs)

if __name__ == "__main__":
    main(sys.argv)
 