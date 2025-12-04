import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np


from dataset import PreprocessedVideoDataset
from model import TemporalViT

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    loop = tqdm(loader, desc="Training", leave=False)
    for frames, labels in loop:
        frames, labels = frames.to(device), labels.to(device)


        #  # Debug: Check input data
        # print(f"Input shape: {frames.shape}")
        # print(f"Input dtype: {frames.dtype}")
        # print(f"Input min: {frames.min().item():.4f}, max: {frames.max().item():.4f}")
        # print(f"Has NaN: {torch.isnan(frames).any().item()}")
        # print(f"Has Inf: {torch.isinf(frames).any().item()}")
        
        
        optimizer.zero_grad()
        
        # device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        # with autocast(device_type = device_type):
        outputs = model(frames)
        loss = criterion(outputs, labels)
            
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()


        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)

        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        loop.set_postfix(loss=loss.item())

    # Calculate Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100. * np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return running_loss / len(loader), accuracy, precision, recall, f1

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

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100. * np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

            
    return running_loss / len(loader), accuracy, precision, recall, f1, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to PROCESSED .pt folder')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading preprocessed dataset...")
    full_dataset = PreprocessedVideoDataset(args.data_path, transform=None)
    
    # --- Print Class Counts ---
    targets = []
    # Attempt to read targets quickly 
    if hasattr(full_dataset, 'samples'):
         targets = [s[1] for s in full_dataset.samples]
         print(f"Total Dataset Balance -> Class 0: {targets.count(0)}, Class 1: {targets.count(1)}")
    else:
         print("Note: Could not quick-scan dataset for class counts (custom structure).")
  

    # Split 70/15/15
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Apply transforms
    # train_dataset.dataset.transform = train_transform
    train_full = PreprocessedVideoDataset(args.data_path, transform=train_transform)
    val_full = PreprocessedVideoDataset(args.data_path, transform=val_transform)
    test_full = PreprocessedVideoDataset(args.data_path, transform=val_transform)

    # Use the same split indices
    generator = torch.Generator().manual_seed(42)  # For reproducibility
    train_dataset, _ = random_split(train_full, [train_size, val_size], generator=generator)
    generator = torch.Generator().manual_seed(42)  # Reset to get same split
    _, val_dataset = random_split(val_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = TemporalViT(num_frames=8).to(device)
    
    # --- CLASS WEIGHTS ---
    class_weights = torch.tensor([5.0, 1.0]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = GradScaler(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0.0

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, cm = validate(model, val_loader, criterion, device)
        
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        # Save based on acc
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved Best Model!")
        
        # Step scheduler
        scheduler.step(val_acc)

if __name__ == "__main__":
    main()