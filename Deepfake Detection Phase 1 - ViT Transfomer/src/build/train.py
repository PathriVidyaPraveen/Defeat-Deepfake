import argparse
from configparser import NoSectionError
from networkx import bridges
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os

from dataset import PreprocessedVideoDataset
from model import TemporalViT

# --- Consistent Video Transform ---
class VideoTransformSubset(Dataset):
    """
    Applies transforms to the entire 4D video tensor (T, C, H, W) 
    to ensure temporal consistency (e.g., if we flip, we flip ALL frames).
    """
    def __init__(self, subset, augment=False):
        self.subset = subset
        self.augment = augment

    def __getitem__(self, index):
        x, y = self.subset[index]
        # x shape: (T, 3, 224, 224)
        
        if self.augment:
            # Random Horizontal Flip
            if torch.rand(1).item() < 0.5:
                # Flip the width dimension (dim 3)
                x = torch.flip(x, dims=[3])

            
            # Random brightness adjustment
            if torch.rand(1).item() < 0.5:
                
                brightness_factor = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
                x = x * brightness_factor

            # Random gaussian noise
            if torch.rand(1).item()  < 0.3:
                noise = torch.randn_like(x) * 0.02
                x = x + noise

            # Clamp values
            x = torch.clamp(x, 0.0, 1.0)
                
        return x, y

    def __len__(self):
        return len(self.subset)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    loop = tqdm(loader, desc="Training", leave=False)
    for frames, labels in loop:
        frames, labels = frames.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        loop.set_postfix(loss=loss.item())

    # Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = 100. * np.mean(all_preds == all_labels)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return running_loss / len(loader), acc, prec, rec, f1, cm

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = 100. * np.mean(all_preds == all_labels)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
            
    return running_loss / len(loader), acc, prec, rec, f1, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    with open('./logs/training_log.txt', 'w') as f:
        f.write(f"Using device: {device}")
        f.write(f"Training started at {torch.cuda.Event().record}\n")
        f.write("=" * 60 + "\n\n")

    
    
    # Load train and val datasets
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')

    train_dataset = PreprocessedVideoDataset(train_path)
    val_dataset = PreprocessedVideoDataset(val_path)

    # Compute balanced sampler BEFORE wrapping with transforms
    
    
    # Get all training labels
    train_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        train_labels.append(label)
    
    train_labels = np.array(train_labels)
    class_counts = np.bincount(train_labels.astype(int))
    print(f"Train counts: Real={class_counts[0]}, Fake={class_counts[1]}")
    
    # Compute sample weights (inversely proportional to class frequency)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[int(label)] for label in train_labels]
    
    # Sample approximately 2x the minority class to balance
    min_count = np.min(class_counts)
    num_samples = int(min_count * 2)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    
    # Apply transforms AFTER computing sampler
    train_dataset = VideoTransformSubset(train_dataset, augment=True)
    val_dataset = VideoTransformSubset(val_dataset, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              sampler=sampler, 
                              num_workers=args.num_workers, pin_memory=True)
                              
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 5. Model & Differential Learning Rate
    model = TemporalViT(num_frames=8).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Low LR for Backbone, High LR for Head
    optimizer = optim.AdamW([
        {'params': model.vit.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},  # Backbone: Very slow learning
        {'params': model.temporal_encoder.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},      # Backbone: Very slow learning
        {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay' : 0.01} # Head: Standard learning
    ])
    

    # Cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-7
    )
    best_acc = 0.0
    print(f"Starting training for {args.epochs} epochs...")
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1_scores = []
    train_cms = []

    # Validation metrics
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    val_cms = []


    for epoch in range(args.epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1, train_cm = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = validate(model, val_loader, criterion, device)
        
        # print(f"Epoch {epoch+1}/{args.epochs}")
        # print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        # print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")

        with open('logs/training_log.txt', 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.epochs}\n")
            f.write(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}\n")
            f.write(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}\n")


        # Store training metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_precisions.append(train_prec)
        train_recalls.append(train_rec)
        train_f1_scores.append(train_f1)
        train_cms.append(train_cm)
        
        # Store validation metrics
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_precisions.append(val_prec)
        val_recalls.append(val_rec)
        val_f1_scores.append(val_f1)
        val_cms.append(val_cm)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("Saved Best Model!")
        
        scheduler.step(val_acc)

    print(f"Completed training for {args.epochs} epochs\n")
    # Save all metrics for plotting
    import pickle 
    metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_precisions': train_precisions,
            'val_precisions': val_precisions,
            'train_recalls': train_recalls,
            'val_recalls': val_recalls,
            'train_f1_scores': train_f1_scores,
            'val_f1_scores': val_f1_scores,
            'train_cms': train_cms,
            'val_cms': val_cms,
            'best_acc': best_acc
        }

    with open('logs/training_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    print(f"\n{'='*60}")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()