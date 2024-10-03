import matplotlib.pyplot as plt
import json

def plot_loss_accuracy(train_losses, val_accuracies, title_suffix):
    plt.figure(figsize=(10, 5))

    # Plotting training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title(f'Training Loss - {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Validation Accuracy - {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# All combinations of activations and initialization methods
combinations = [
    ('relu', 'small_random'),
    ('relu', 'xavier'),
    ('sigmoid', 'small_random'),
    ('sigmoid', 'xavier')
]

# Visualization for each combination
for activation_fn, init_method in combinations:
    try:
        metrics_file = f'results/train_val_metrics_{activation_fn}_{init_method}.json'
        
        # Loading metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        train_losses = metrics['train_losses']
        val_accuracies = metrics['val_accuracies']

        # Visualizing losses and accuracies for each combination
        plot_loss_accuracy(train_losses, val_accuracies, f'{activation_fn} + {init_method}')
    
    except FileNotFoundError:
        print(f"Metrics file not found for {activation_fn} + {init_method}: {metrics_file}")
