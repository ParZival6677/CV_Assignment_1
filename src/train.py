import torch
import torch.optim as optim
from cnn_model import CNNModel
from dataset_loader import load_data
import torch.nn as nn
import json
import os

def weights_init_small(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

def train_model(activation='relu', init_method='xavier'):
    # Loading data
    train_loader, test_loader, val_loader, num_classes = load_data()

    # Creating a model with a dynamic number of classes
    model = CNNModel(num_classes)

    # Initializing weights
    if init_method == 'small_random':
        model.apply(weights_init_small)
    elif init_method == 'xavier':
        model.apply(weights_init_xavier)

    # Setting the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Training the model
    num_epochs = 10
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

        # Evaluation on the validation set
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        print(f'Validation Accuracy: {accuracy}%')

    # Creating a folder to save models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Saving the model
    model_save_path = f'models/model_cnn_{activation}_{init_method}.pth'
    torch.save(model.state_dict(), model_save_path)

    # Creating a folder to save metrics if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Saving loss and accuracy data to a file
    metrics_save_path = f'results/train_val_metrics_{activation}_{init_method}.json'
    with open(metrics_save_path, 'w') as f:
        json.dump({'train_losses': train_losses, 'val_accuracies': val_accuracies}, f)

# Training with different combinations of activation and initialization
train_model(activation='relu', init_method='small_random')
train_model(activation='relu', init_method='xavier')
train_model(activation='sigmoid', init_method='small_random')
train_model(activation='sigmoid', init_method='xavier')
