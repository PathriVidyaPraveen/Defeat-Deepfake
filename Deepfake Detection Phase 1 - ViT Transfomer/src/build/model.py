import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class WaveletModel(nn.Module):
    """
    Lightweight CNN desgined to extract features from high-frequency wavelet bands
    Input: (B, 3, 224, 224) -> Output: (B, 128)
    """
    def __init__(self, output_dim=128):
        super(WaveletModel, self).__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Global Average Pooling to flatten spatial dims
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            
            # Project to embedding
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class TemporalViT(nn.Module):
    def __init__(self, num_frames=8, num_classes=2):
        super(TemporalViT, self).__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)

        # --- FREEZING STRATEGY ---
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 blocks
        for i in [10, 11]:
            for param in self.vit.encoder.layers[i].parameters():
                param.requires_grad = True
        for param in self.vit.encoder.ln.parameters():
            param.requires_grad = True
        
        self.vit.heads = nn.Sequential(nn.Identity())
        
        # Dimensions
        self.rgb_embed_dim = 768
        self.wavelet_embed_dim = 128
        self.wavelet_cnn = WaveletModel(output_dim=self.wavelet_embed_dim)
        
        total_input_dim = self.rgb_embed_dim + self.wavelet_embed_dim # 896

        # --- NEW: TEMPORAL CNN INSTEAD OF LSTM ---
        # Input: (Batch, 896, 8) -> Output: (Batch, 256, 8)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(in_channels=total_input_dim, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Second layer for deeper temporal abstraction
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Global Average Pooling over time (Squash 8 frames into 1 vector)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        b, t, c, h, w = x.shape
        
        # 1. Fold time for frame-wise feature extraction
        x = x.view(b * t, c, h, w)

        # 2. Split Streams (Ensure dataloader provides 6 channels!)
        x_rgb = x[:, :3, :, :]
        x_wav = x[:, 3:, :, :] 

        # 3. Extract Features
        rgb_features = self.vit(x_rgb)       # (B*T, 768)
        wav_features = self.wavelet_cnn(x_wav) # (B*T, 128)
        
        # 4. Concatenate
        fused_features = torch.cat((rgb_features, wav_features), dim=1) # (B*T, 896)
        
        # 5. Unfold time
        # Shape becomes (Batch, Time, Features)
        temporal_input = fused_features.view(b, t, -1)
        
        # 6. Permute for CNN
        # Conv1d expects (Batch, Channels, Time) -> We have (Batch, Time, Features)
        # So we swap dimensions 1 and 2
        temporal_input = temporal_input.permute(0, 2, 1) # (B, 896, T)
        
        # 7. Temporal Processing
        t_out = self.temporal_cnn(temporal_input) # (B, 128, T)
        
        # 8. Pooling & Classify
        t_out = self.global_pool(t_out) # (B, 128, 1)
        out = self.classifier(t_out)
        
        return out