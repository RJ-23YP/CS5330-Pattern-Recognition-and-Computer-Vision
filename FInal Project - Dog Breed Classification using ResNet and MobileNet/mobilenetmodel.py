"""
Ruchik Jani (NUID - 002825482)
Anuj Patel (NUID - 002874710)
Spring 2024
CS 5330 Computer Vision

This file contains the code for training the MobileNet V2 model and saving its weights. 
"""


# import statements
import sys
import os
import pandas
import matplotlib.pyplot as plott
from PIL import Image
import torch 
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import torchvision.models 
from torchvision import transforms
import torchmetrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import time 
import seaborn
from torchviz import make_dot


# Class Definitions 
# Dataset class for loading images and their labels
class loader_image(Dataset):
    # Initialize the dataset with dataframe, transformation function, and test mode flag.
    def __init__(self, dataframe, transform=None, test=False):
        self.data = dataframe.values  # Convert dataframe to a NumPy array
        self.transform = transform
        self.test = test

    # Get an item (image and label) from the dataset at the specified index.
    def __getitem__(self, index):
        image_path, label = self.data[index]  # Unpack the image path and label
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image  # Return only the image if in test mode
        else:
            return image, label  # Return both image and label if not in test mode

    # Return the length of the dataset.
    def __len__(self):
        return len(self.data)
        
    
# Define the neural network architecture
class CNN(torch.nn.Module):

    # Initialize the neural network model.
    def __init__(self, base_model, num_classes):
        super(CNN, self).__init__()
        self.base_model = base_model
        self.base_out_features = 1280 * 7 * 7  #Input size is 224x224
        self.linear1 = torch.nn.Linear(self.base_out_features, 1280)
        self.linear2 = torch.nn.Linear(1280, 512)
        self.output = torch.nn.Linear(512, num_classes)

    # Forward pass of the neural network
    def forward(self, x):
        x = self.base_model.features(x)  # Extract features using base model
        x = torch.flatten(x, 1)  # Flatten the feature map
        x = torch.nn.functional.relu(self.linear1(x))  # Apply ReLU activation to first linear layer
        x = torch.nn.functional.relu(self.linear2(x))  # Apply ReLU activation to second linear layer
        x = self.output(x)  # Final output logits
        return x


# Function Definitions
# Train the model using the given hyper-parameters
def train_model(train_loader, val_loader, device, model, criterion, optimizer, epochs):
    
    training_losses = []  # List to store training losses for each epoch
    validation_losses = []  # List to store validation losses for each epoch
    training_accuracies = []  # List to store training accuracies for each epoch
    validation_accuracies = []  # List to store validation accuracies for each epoch
    training_times = []  # List to store training times for each epoch

    # Metric for computing accuracy
    train_acc_metric = torchmetrics.Accuracy(num_classes=120, average='macro', task='multiclass')
    val_acc_metric = torchmetrics.Accuracy(num_classes=120, average='macro', task='multiclass')

    for epoch in range(epochs):
        start_time = time.time()
        print(f'Training epoch {epoch + 1}')
      
        # Training
        model.train()
        train_batch_losses = [] # List to store training losses for each batch
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_batch_losses.append(loss.item())
            train_acc_metric(outputs.cpu(), targets.cpu())

        # Validation
        model.eval()
        val_batch_losses = [] # List to store validation losses for each batch
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_batch_losses.append(loss.item())
            val_acc_metric(outputs.cpu(), targets.cpu())

        # Update loss and accuracy lists
        training_losses.append(sum(train_batch_losses) / len(train_batch_losses))
        validation_losses.append(sum(val_batch_losses) / len(val_batch_losses))
        training_accuracies.append(train_acc_metric.compute())
        validation_accuracies.append(val_acc_metric.compute())
        train_acc_metric.reset()
        val_acc_metric.reset()

        end_time = time.time()
        epoch_time = end_time - start_time
        training_times.append(epoch_time)

        print(f'Training Loss: {training_losses[-1]:.3f}, Training Accuracy: {training_accuracies[-1]:.3f}, '
              f'Validation Loss: {validation_losses[-1]:.3f}, Validation Accuracy: {validation_accuracies[-1]:.3f}, '
              f'Time: {epoch_time:.2f}s')

    print('Training is complete.')

    # Calculate averages
    avg_training_loss = sum(training_losses) / len(training_losses)
    avg_validation_loss = sum(validation_losses) / len(validation_losses)
    avg_training_accuracy = sum(training_accuracies) / len(training_accuracies)
    avg_validation_accuracy = sum(validation_accuracies) / len(validation_accuracies)
    avg_training_time = sum(training_times) / len(training_times)

    print(f'Average Training Loss: {avg_training_loss:.3f}')
    print(f'Average Validation Loss: {avg_validation_loss:.3f}')
    print(f'Average Training Accuracy: {avg_training_accuracy:.3f}')
    print(f'Average Validation Accuracy: {avg_validation_accuracy:.3f}')
    print(f'Average Training Time per Epoch: {avg_training_time:.2f}s')

    return training_losses, training_accuracies, validation_losses, validation_accuracies


# Function for data preprocessing involving the encoding of labels and creating a dictionary of class indices
def data_preprocess(data_dir, sample_size=None):
    # Read the csv files
    labels_df = pandas.read_csv(os.path.join(data_dir, 'labels.csv'))
    test_df = pandas.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    # Number of images in Training set and test set
    print(f'Training set: {labels_df.shape[0]}, Test set: {test_df.shape[0]}')
    print(labels_df['breed'].value_counts())

    # Encode the breed into digits
    label_encoder = LabelEncoder()
    labels_df['label'] = label_encoder.fit_transform(labels_df['breed'])

    # Create a breed-2-index dictionary
    breed_ind = dict(zip(labels_df['label'].unique(), labels_df['breed'].unique()))

    # Change the id to full file path
    train_dir = os.path.join(data_dir, 'train')
    labels_df['id'] = labels_df['id'].apply(lambda x: os.path.join(train_dir, f'{x}.jpg'))

    # Drop the breed column
    labels_df = labels_df.drop('breed', axis=1)

    # Reduce the number of samples
    if sample_size:
        labels_df = labels_df.sample(sample_size, random_state=42)

    return labels_df


# Function for splitting the dataset and applying transformations to create dataloaders.
def create_data_loaders(frame_lbl, batch_size, val_split):
    # Create transformers
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tf_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Split the dataset using stratified sampling
    split = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    for train_idx, val_idx in split.split(frame_lbl, frame_lbl['label']):
        train_df = frame_lbl.loc[train_idx]
        val_df = frame_lbl.loc[val_idx]

    # Create datasets
    training_set = loader_image(train_df, transform=tf_train)
    validation_set = loader_image(val_df, transform=tf_val)

    # Create dataloaders
    dataloader_train = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    dataloader_validation = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    print(f'Training set: {len(train_df)}, Validation set: {len(val_df)}')
    return dataloader_train, dataloader_validation


# Function for loading the model for further training
def create_model(num_classes):
    # Load the pre-trained MobileNet model
    mobile_net = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze the base model parameters
    for param in mobile_net.parameters():
        param.requires_grad = False

    # Create the final model
    model = CNN(base_model=mobile_net, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device


# Function to plot the training and validation loss and accuracy curves.
def visualize_training(loss_train, loss_val, loss_acc, acc_val, epoch):
    seaborn.set_style("darkgrid") # Set plot style to seaborn. 
   
    # Plot training and validation loss
    plott.figure(figsize=(10, 5))
    plott.plot(loss_train, label='Training Loss')
    plott.plot(loss_val, label='Validation Loss')
    plott.xlabel('Epoch')
    plott.ylabel('Loss')
    plott.title('Training and Validation Loss')
    plott.legend()
    plott.xticks(range(0, epoch + 1))  # Set x-axis ticks for all epochs
    plott.show()

    # Plot training and validation accuracy
    plott.figure(figsize=(10, 5))
    plott.plot(loss_acc, label='Training Accuracy')
    plott.plot(acc_val, label='Validation Accuracy')
    plott.xlabel('Epoch')
    plott.ylabel('Accuracy')
    plott.title('Training and Validation Accuracy')
    plott.legend()
    plott.xticks(range(0, epoch + 1))  # Set x-axis ticks for all epochs
    plott.show()
    return


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

    #Visualize the architecture
    x = torch.randn(1, 3, 224, 224).cuda()
    output = model(x)
    make_dot(output, params=dict(model.named_parameters())).render("/home/rj/Final Project/Kaggle/mobilearc", format="png")

    # Loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # Model Training Phase
    loss_train, loss_acc, loss_val, acc_val = train_model(
        dataloader_train, dataloader_validation, device, model,
        loss_fn, optimizer, num_epochs
    )

    # Save the trained model
    model_path = f'{data_dir}/mobilenetmodels/v2model.pth'
    optimizer_path = f'{data_dir}/mobilenetmodels/v2optimizer.pth'
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    print(f"Model saved successfully at: {model_path}")
    print(f"Optimizer saved successfully at: {optimizer_path}")

    # Plot the accuracy and loss curves
    visualize_training(loss_train, loss_val, loss_acc, acc_val, num_epochs)

if __name__ == "__main__":
    main(sys.argv)
 