import os
import torch
from torch.utils.data import Dataset

class PreprocessedVideoDataset(Dataset):
    def __init__(self, data_path, num_frames=8):
        self.data_path = data_path
        self.num_frames = num_frames
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
        
        try:
            # Load tensor (Shape: T, 3, 224, 224)
            video_tensor = torch.load(path).float()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(self.num_frames, 3, 224, 224), label

        # 1. Sanity Check
        if torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any():
            video_tensor = torch.zeros_like(video_tensor)

        # 2. Handle Padding (If video is too short)
        # We ensure T=8 here strictly.
        if video_tensor.shape[0] < self.num_frames:
            diff = self.num_frames - video_tensor.shape[0]
            last_frame = video_tensor[-1].unsqueeze(0)
            # Add slight noise to padding to avoid "identical feature" collapse
            padding = last_frame.repeat(diff, 1, 1, 1)
            padding = padding + (torch.randn_like(padding) * 0.01)
            video_tensor = torch.cat([video_tensor, padding], dim=0)
        
        # Truncate if too long
        video_tensor = video_tensor[:self.num_frames]

        # 3. Per-Channel Standardization (Robust)
        # Calculate stats across (Time, Height, Width) for each Channel
        mean = video_tensor.mean(dim=[0, 2, 3], keepdim=True)
        std = video_tensor.std(dim=[0, 2, 3], keepdim=True)
        
        #  Clamp std to prevent division by near-zero (common in Wavelet LH/HL/HH)
        std = torch.clamp(std, min=1e-3)
        
        video_tensor = (video_tensor - mean) / std

        return video_tensor, label