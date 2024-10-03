import torch.nn as nn
import torch

class CNNModel(nn.Module):
    def __init__(self, num_classes, activation_fn='relu'):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5) 
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)  

        # Set activation function
        if activation_fn == 'relu':
            self.activation = nn.functional.relu
        elif activation_fn == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 32 * 32)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
