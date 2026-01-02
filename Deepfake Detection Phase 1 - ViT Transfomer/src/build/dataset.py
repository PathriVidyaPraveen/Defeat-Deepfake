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
            # Load tensor (Shape: T, 6, 224, 224)
            # Channels 0-2: RGB (0-255 range usually, coming from cv2)
            # Channels 3-5: Wavelets (Float, centered around 0)
            video_tensor = torch.load(path).float()
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # FIX: Must return 6 channels to match valid data!
            return torch.zeros(self.num_frames, 6, 224, 224), label

        # 1. Sanity Check
        if torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any():
            video_tensor = torch.zeros_like(video_tensor)

        # 2. Handle Padding (Ensure T=8)
        T = video_tensor.shape[0]
        if T < self.num_frames:
            diff = self.num_frames - T
            last_frame = video_tensor[-1].unsqueeze(0)
            # Add slight noise to padding to avoid "identical feature" collapse in LSTM
            padding = last_frame.repeat(diff, 1, 1, 1)
            padding = padding + (torch.randn_like(padding) * 0.01)
            video_tensor = torch.cat([video_tensor, padding], dim=0)
        
        # Truncate if too long
        video_tensor = video_tensor[:self.num_frames]

        # 3. SCALING (The correct way)
        # RGB (Channels 0-2) are likely 0-255. We need them 0-1 for the model.
        # Wavelets (Channels 3-5) are already small floats. We leave them alone.
        
        rgb = video_tensor[:, :3, :, :] / 255.0  # Scale RGB to [0, 1]
        wavelets = video_tensor[:, 3:, :, :]     # Keep wavelets raw
        
        # Recombine
        video_tensor = torch.cat([rgb, wavelets], dim=1)

        return video_tensor, label