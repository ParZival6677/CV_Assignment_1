import torch
from cnn_model import CNNModel
from dataset_loader import load_data

def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Loading data
_, test_loader, _, num_classes = load_data()

# List of all activation and initialization combinations
combinations = [
    ('relu', 'small_random'),
    ('relu', 'xavier'),
    ('sigmoid', 'small_random'),
    ('sigmoid', 'xavier')
]

# Evaluate each combination
for activation_fn, init_method in combinations:
    model_save_path = f'models/model_cnn_{activation_fn}_{init_method}.pth'
    
    try:
        # Creating model with selected activation
        model = CNNModel(num_classes, activation_fn=activation_fn)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        # Evaluate the model
        accuracy = evaluate(model, test_loader)
        print(f'Test Accuracy for {activation_fn} + {init_method}: {accuracy}%')
    
    except FileNotFoundError:
        print(f"Model file not found for {activation_fn} + {init_method}: {model_save_path}")
