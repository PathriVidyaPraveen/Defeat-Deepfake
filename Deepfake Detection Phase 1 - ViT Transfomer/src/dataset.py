from math import isnan
import os
import torch
from torch.utils.data import Dataset

class PreprocessedVideoDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        
        # Load Real (Label 0)
        self._load_dir(os.path.join(data_path, '0_Celeb-real'), 0)
        # Load Fake (Label 1)
        self._load_dir(os.path.join(data_path, '1_Celeb-synthesis'), 1)
        
        print(f"Loaded {len(self.samples)} preprocessed videos.")

    def _load_dir(self, dir_path, label):
        if os.path.exists(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pt')]
            self.samples.extend([(f, label) for f in files])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Load tensor (Shape: T, 3, 224, 224)
        video_tensor = torch.load(path).float()

        # Check for invalid values
        if torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any():
            print(f"Warning: Invalid values found in {path}. Replacing with zeros.")
            video_tensor = torch.zeros_like(video_tensor)
        

        min_val = video_tensor.min()
        max_val = video_tensor.max()

        if max_val > min_val:  # Avoid division by zero
            video_tensor = (video_tensor - min_val) / (max_val - min_val)
        else:
            video_tensor = torch.zeros_like(video_tensor)

        # Clamp to ensure valid range
        video_tensor = torch.clamp(video_tensor, 0.0, 1.0)

        # Ensure values are in [0, 1] range
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0
        
        if self.transform:
            # Apply transform to every frame
            transformed_frames = []
            for t in range(video_tensor.shape[0]):
                frame = self.transform(video_tensor[t])
                transformed_frames.append(frame)
            video_tensor = torch.stack(transformed_frames)
            
        return video_tensor, label