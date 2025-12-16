import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class TemporalViT(nn.Module):
    def __init__(self, num_frames=8, num_classes=2):
        super(TemporalViT, self).__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)
        
        self.embed_dim = 768 
        self.num_frames = num_frames
        self.vit.heads = nn.Sequential(nn.Identity())
        
        # Add LSTM for temporal modeling
        self.temporal_encoder = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)
        
        # Updated classifier for LSTM output (bidirectional = 2x hidden size)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        b, t, c, h, w = x.shape
        
        # Fold time
        x = x.view(b * t, c, h, w)
        
        # Backbone
        features = self.vit(x) # (b*t, 768)
        
        # Unfold time
        features = features.view(b, t, -1) # (b, t, 768)
        
        # Temporal encoding with LSTM
        lstm_out, (hidden, cell) = self.temporal_encoder(features)
        
        # Use last hidden state (concatenated from both directions)
        # hidden shape: (num_layers * 2, batch, hidden_size)
        hidden = hidden.view(2, 2, b, 256)  # (num_layers, directions, batch, hidden)
        last_hidden = hidden[-1]  # Last layer: (2, batch, 256)
        last_hidden = last_hidden.permute(1, 0, 2).contiguous()  # (batch, 2, 256)
        last_hidden = last_hidden.view(b, -1)  # (batch, 512)
        
        # Regularize
        features = self.dropout(last_hidden)
        
        out = self.classifier(features)
        return out