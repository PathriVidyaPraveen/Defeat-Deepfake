import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

def main():
    # Load metrics
    with open('local_logs/training_metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)

    # Access individual metrics
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    train_accuracies = metrics['train_accuracies']
    val_accuracies = metrics['val_accuracies']
    train_precisions = metrics['train_precisions']
    val_precisions = metrics['val_precisions']
    train_recalls = metrics['train_recalls']
    val_recalls = metrics['val_recalls']
    train_f1_scores = metrics['train_f1_scores']
    val_f1_scores = metrics['val_f1_scores']
    train_cm = metrics['train_cms']
    val_cm = metrics['val_cms']

    epochs = np.arange(1, len(train_losses) + 1)
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

    # Plot 1: Loss
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Progression', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax2.plot(epochs, train_accuracies, 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, val_accuracies, 'r-s', label='Val Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Accuracy Progression', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Precision & Recall
    ax3.plot(epochs, train_precisions, 'b-o', label='Train Precision', linewidth=2, markersize=4)
    ax3.plot(epochs, val_precisions, 'r-s', label='Val Precision', linewidth=2, markersize=4)
    ax3.plot(epochs, train_recalls, 'g-^', label='Train Recall', linewidth=2, markersize=4)
    ax3.plot(epochs, val_recalls, 'm-d', label='Val Recall', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Precision & Recall', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: F1 Score
    ax4.plot(epochs, train_f1_scores, 'b-o', label='Train F1', linewidth=2, markersize=4)
    ax4.plot(epochs, val_f1_scores, 'r-s', label='Val F1', linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('F1 Score', fontsize=11)
    ax4.set_title('F1 Score Progression', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots_new/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    fig.suptitle('Confusion Matrices', fontsize = 16, fontweight = 'bold')

    # Train confusion matrix
    sns.heatmap(train_cm[-1], annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=True)
    ax1.set_title('Train Confusion Matrix', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    # Val confusion matrix
    sns.heatmap(val_cm[-1], annot=True, fmt='d', cmap='Greens', ax=ax2, cbar=True)
    ax2.set_title('Validation Confusion Matrix', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png', dpi=300,bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()


