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

        # FIX 1: Unfreeze last 2 blocks of ViT for fine-tuning
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Unfreeze encoder blocks 10 and 11 (last 2 blocks)
        for i in [10, 11]:
            for param in self.vit.encoder.layers[i].parameters():
                param.requires_grad = True
        
        # Unfreeze layer norm
        for param in self.vit.encoder.ln.parameters():
            param.requires_grad = True
        
        self.vit.heads = nn.Sequential(nn.Identity())
        self.rgb_embed_dim = 768

        # Trainable Wavelet CNN 
        self.wavelet_embed_dim = 128
        self.wavelet_cnn = WaveletModel(output_dim=self.wavelet_embed_dim)

        # Fusion
        self.num_frames = num_frames

        # LSTM input
        total_input_dim = self.rgb_embed_dim + self.wavelet_embed_dim
        
        # LSTM Parameters
        self.temporal_encoder = nn.LSTM(
            input_size=total_input_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,  # Reduced from 0.5
            bidirectional=True
        )
        
        # Adding dropout to entire model
        self.dropout = nn.Dropout(p=0.3)  # Reduced from 0.5
        
        # Bidirectional LSTM
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),  
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        b, t, c, h, w = x.shape
        
        # Fold time
        x = x.view(b * t, c, h, w)

        # Split channels - VERIFY THIS IS CORRECT
        x_rgb = x[:, :3, :, :]
        x_wav = x[:, 3:, :, :]  # Must be channels 3-5, NOT 0-2!

        # RGB stream - now partially trainable
        rgb_features = self.vit(x_rgb)
        
        # Wavelet stream for CNN 
        wav_features = self.wavelet_cnn(x_wav)
        
        # Feature fusion
        fused_features = torch.cat((rgb_features, wav_features), dim=1)
        
        # Unfold time
        lstm_input = fused_features.view(b, t, -1)
        
        # Temporal encoding with LSTM
        lstm_out, (hidden, cell) = self.temporal_encoder(lstm_input)
        
        # Use last hidden state
        hidden = hidden.view(2, 2, b, 256)
        last_hidden = hidden[-1]
        last_hidden = last_hidden.permute(1, 0, 2).contiguous()
        last_hidden = last_hidden.view(b, -1)
        
        # Classification
        features = self.dropout(last_hidden)
        out = self.classifier(features)

        return out