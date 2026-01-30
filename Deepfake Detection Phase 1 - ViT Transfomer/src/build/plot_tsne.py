import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from torch.cuda.amp import autocast, GradScaler

from dataset import PreprocessedVideoDataset
from model import TemporalViT

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def extract_features(model, dataloader, device, num_samples_per_class=50):
    """
    Extract 896-dimensional fused features from the model.
    Returns features and labels for specified number of samples per class.
    """
    model.eval()
    
    features_dict = {0: [], 1: []}  # Store features per class
    labels_dict = {0: [], 1: []}    # Store labels per class
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Extracting features"):
            frames = frames.to(device)
            b, t, c, h, w = frames.shape
            
            # 1. Fold time for frame-wise feature extraction
            x = frames.view(b * t, c, h, w)

            # 2. Split Streams
            x_rgb = x[:, :3, :, :]
            x_wav = x[:, 3:, :, :] 

            # 3. Extract Features
            rgb_features = model.vit(x_rgb)       # (B*T, 768)
            wav_features = model.wavelet_cnn(x_wav) # (B*T, 128)
            
            # 4. Concatenate to get 896-dim features
            fused_features = torch.cat((rgb_features, wav_features), dim=1) # (B*T, 896)
            
            # Average over time to get per-video features
            fused_features = fused_features.view(b, t, -1).mean(dim=1)  # (B, 896)
            
            # Store features by class
            for i, label in enumerate(labels):
                label_val = label.item()
                if len(features_dict[label_val]) < num_samples_per_class:
                    features_dict[label_val].append(fused_features[i].cpu().numpy())
                    labels_dict[label_val].append(label_val)
            
            # Check if we have enough samples
            if all(len(features_dict[k]) >= num_samples_per_class for k in [0, 1]):
                break
    
    # Combine features and labels
    all_features = []
    all_labels = []
    
    for label in [0, 1]:
        all_features.extend(features_dict[label][:num_samples_per_class])
        all_labels.extend(labels_dict[label][:num_samples_per_class])
    
    all_features = np.array(all_features)  # (100, 896)
    all_labels = np.array(all_labels)      # (100,)
    
    print(f"Extracted features shape: {all_features.shape}")
    print(f"Class 0 samples: {np.sum(all_labels == 0)}")
    print(f"Class 1 samples: {np.sum(all_labels == 1)}")
    
    return all_features, all_labels


def plot_tsne(features, labels, save_path='logs/tsne_plot.png', use_pca=True, pca_components=50):
    """
    Apply PCA (optional) and t-SNE, then plot the results.
    """
    print(f"\nOriginal feature dimension: {features.shape}")
    
    # Apply PCA for dimensionality reduction (optional but recommended for t-SNE)
    if use_pca and features.shape[1] > pca_components:
        print(f"Applying PCA to reduce to {pca_components} dimensions...")
        pca = PCA(n_components=pca_components, random_state=42)
        features_reduced = pca.fit_transform(features)
        print(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        print(f"Reduced feature dimension: {features_reduced.shape}")
    else:
        features_reduced = features
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
    features_2d = tsne.fit_transform(features_reduced)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot each class with different colors
    colors = ['#FF6B6B', '#4ECDC4']  # Red for fake, Teal for real
    labels_text = ['Real', 'Fake']
    
    for label_val in [0, 1]:
        mask = labels == label_val
        plt.scatter(
            features_2d[mask, 0], 
            features_2d[mask, 1],
            c=colors[label_val],
            label=labels_text[label_val],
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title('t-SNE Visualization of Fused Features (RGB + Wavelet)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nt-SNE plot saved to: {save_path}")
    
    # Also save as PDF for publication quality
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot (PDF) saved to: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--samples_per_class', type=int, default=50, help='Number of samples per class')
    parser.add_argument('--use_pca', action='store_true', default=False, help='Use PCA before t-SNE')
    parser.add_argument('--pca_components', type=int, default=50, help='Number of PCA components')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load validation dataset (or you can use train)
    val_path = os.path.join(args.data_path, 'val')
    val_dataset = PreprocessedVideoDataset(val_path)
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,  # Shuffle to get diverse samples
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # Load model
    print("Loading model...")
    model = TemporalViT(num_frames=8).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: Model file {args.model_path} not found. Using untrained model.")
    
    # Extract features
    print(f"\nExtracting {args.samples_per_class} samples per class...")
    features, labels = extract_features(model, val_loader, device, num_samples_per_class=args.samples_per_class)
    
    # Save features for later use
    features_save_path = 'logs/extracted_features.pkl'
    os.makedirs('logs', exist_ok=True)
    with open(features_save_path, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)
    print(f"Features saved to {features_save_path}")
    
    # Plot t-SNE
    plot_tsne(
        features, 
        labels, 
        save_path='logs/tsne_visualization.png',
        use_pca=args.use_pca,
        pca_components=args.pca_components
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()