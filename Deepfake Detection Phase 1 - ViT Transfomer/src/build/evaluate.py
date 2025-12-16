import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import os

from dataset import PreprocessedVideoDataset
from model import TemporalViT


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Evaluating"):
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = 100. * accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, prec, rec, f1, cm
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to processed data root')
    parser.add_argument('--model_path', type=str, default='./models/best_model.pth')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset from test folder
    test_path = os.path.join(args.data_path, 'test')
    print(f"Loading test data from: {test_path}")
    
    test_dataset = PreprocessedVideoDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    
    
    # Load model
    model = TemporalViT(num_frames=8).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at {args.model_path}")
        return
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Evaluate
    acc, prec, rec, f1, cm = evaluate(model, test_loader, device)
    
    
    # Save results to file
    with open('./logs/test_results.txt', 'w') as f:
        f.write("TEST SET RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy:  {acc:.2f}%\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    
    print("\nResults saved to test_results.txt")

if __name__ == "__main__":
    main()