import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    # Define confusion matrices
    cm_hh = np.array([[0, 25], [0, 120]])
    cm_ll = np.array([[6, 19], [25, 95]])
    
    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Test Set Confusion Matrices', fontsize=16, fontweight='bold')
    
    # Plot HH Confusion Matrix
    disp_hh = ConfusionMatrixDisplay(confusion_matrix=cm_hh, display_labels=['Real', 'Fake'])
    disp_hh.plot(ax=ax1, cmap='Blues', values_format='d')
    ax1.set_title('HH Test Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Plot LL Confusion Matrix
    disp_ll = ConfusionMatrixDisplay(confusion_matrix=cm_ll, display_labels=['Real', 'Fake'])
    disp_ll.plot(ax=ax2, cmap='Greens', values_format='d')
    ax2.set_title('LL Test Confusion Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/confusion_matrices_test.png', dpi=300, bbox_inches='tight')
    print("Confusion matrices plot saved to plots/confusion_matrices.png")
    plt.show()

if __name__ == "__main__":
    main()