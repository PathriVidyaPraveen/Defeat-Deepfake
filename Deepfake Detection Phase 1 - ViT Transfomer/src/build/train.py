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

from torch.cuda.amp import autocast, GradScaler

from dataset import PreprocessedVideoDataset
from model import TemporalViT

# --- Consistent Video Transform ---
class VideoTransformSubset(Dataset):
    def __init__(self, subset, augment=False):
        self.subset = subset
        self.augment = augment
        
        # ImageNet normalization for ViT
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __getitem__(self, index):
        x, y = self.subset[index]
        # x shape: (T, 6, 224, 224) 
        # Channels 0-2: RGB (already scaled to [0,1] in dataset.py)
        # Channels 3-5: Wavelet coefficients
        
        if self.augment:
            # Geometric Augmentation (Apply to BOTH RGB and Wavelet)
            # Random Horizontal Flip
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[3])  # Flip width

            # Random rotation (90 degree increments)
            if torch.rand(1).item() < 0.3:
                k = int(torch.randint(1, 4, (1,)).item())  
                x = torch.rot90(x, k, dims=[2, 3])

            # Temporal Augmentation
            # Random temporal shift/drop
            if torch.rand(1).item() < 0.2:
                drop_idx = int(torch.randint(0, x.shape[0], (1,)).item())
                replace_idx = max(0, drop_idx - 1)
                x[drop_idx] = x[replace_idx]

            # Random temporal reverse
            if torch.rand(1).item() < 0.2:
                x = torch.flip(x, dims=[0])

            # Photometric Augmentation (RGB only)
            rgb = x[:, :3, :, :]
            wav = x[:, 3:, :, :]

            # Random brightness
            if torch.rand(1).item() < 0.5:
                brightness_factor = 0.7 + torch.rand(1).item() * 0.6  # [0.7, 1.3]
                rgb = rgb * brightness_factor

            # Random contrast
            if torch.rand(1).item() < 0.3:
                contrast_factor = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
                mean_val = rgb.mean(dim=[2, 3], keepdim=True)
                rgb = (rgb - mean_val) * contrast_factor + mean_val

            # Random color jitter
            if torch.rand(1).item() < 0.3:
                jitter = torch.randn(1, 3, 1, 1) * 0.1
                rgb = rgb + jitter

            # Random Gaussian noise
            if torch.rand(1).item() < 0.3:
                noise = torch.randn_like(rgb) * 0.03
                rgb = rgb + noise

            # Random cutout/erasing
            if torch.rand(1).item() < 0.2:
                h, w = rgb.shape[2], rgb.shape[3]
                cut_h, cut_w = h // 4, w // 4
                top = int(torch.randint(0, h - cut_h, (1,)).item())  
                left = int(torch.randint(0, w - cut_w, (1,)).item()) 
                rgb[:, :, top:top+cut_h, left:left+cut_w] = 0

            # Clamp RGB to [0, 1]
            rgb = torch.clamp(rgb, 0.0, 1.0)
            
            # Recombine before normalization
            x = torch.cat((rgb, wav), dim=1)

        
        rgb = x[:, :3, :, :]
        wav = x[:, 3:, :, :]

        # Normalize RGB with ImageNet stats
        rgb = (rgb - self.mean) / self.std

        # Normalize wavelets to zero mean, unit variance 
        wav_mean = wav.mean()
        wav_std = wav.std() + 1e-6
        wav = (wav - wav_mean) / wav_std

        # Recombine
        x = torch.cat((rgb, wav), dim=1)

        return x, y

    def __len__(self):
        return len(self.subset)
    

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    loop = tqdm(loader, desc="Training", leave=False)
    for frames, labels in loop:
        frames, labels = frames.to(device), labels.to(device)
        
        optimizer.zero_grad()

       
        with autocast():
            outputs = model(frames)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

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

    
    scaler = GradScaler()

    # Create directories if they don't exist
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    with open('./logs/training_log.txt', 'w') as f:
        f.write(f"Using device: {device}\n")
        f.write(f"Training started\n")
        f.write("=" * 60 + "\n\n")

    # Load train and val datasets
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')

    train_dataset = PreprocessedVideoDataset(train_path)
    val_dataset = PreprocessedVideoDataset(val_path)

    # Compute balanced sampler before wrapping with transforms
    # Get all training labels
    train_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        train_labels.append(label)
    
    train_labels = np.array(train_labels)
    class_counts = np.bincount(train_labels.astype(int))
    print(f"Train counts: Real={class_counts[0]}, Fake={class_counts[1]}")
    
    # Compute sample weights 
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
    
    # Apply transforms after computing sampler
    train_dataset = VideoTransformSubset(train_dataset, augment=True)
    val_dataset = VideoTransformSubset(val_dataset, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              sampler=sampler, 
                              num_workers=args.num_workers, pin_memory=True)
                              
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model & Differential Learning Rate
    model = TemporalViT(num_frames=8).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- OPTIMIZER  ---
    optimizer = optim.AdamW([
        # ViT Fine-tuning (Lower LR)
        {'params': model.vit.encoder.layers[10].parameters(), 'lr': 5e-6, 'weight_decay': 0.05},
        {'params': model.vit.encoder.layers[11].parameters(), 'lr': 5e-6, 'weight_decay': 0.05},
        {'params': model.vit.encoder.ln.parameters(), 'lr': 5e-6, 'weight_decay': 0.05},
        
        # Wavelet Stream (Moderate LR)
        {'params': model.wavelet_cnn.parameters(), 'lr': 2e-4, 'weight_decay': 0.08}, 
        
        # Temporal CNN (New Component - Higher LR usually safe for fresh layers)
        {'params': model.temporal_cnn.parameters(), 'lr': 1e-3, 'weight_decay': 0.01}, 
        
        # Classifier
        {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 0.01}  
    ])
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose='True'
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
        train_loss, train_acc, train_prec, train_rec, train_f1, train_cm = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = validate(
            model, val_loader, criterion, device
        )

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
        
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"Saved Best Model! Val Acc: {val_acc:.2f}%")
        
       

    print(f"Completed training for {args.epochs} epochs\n")
    
    # Save all metrics for plotting
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
    print(f"{'='*60}")

if __name__ == "__main__":
    main()