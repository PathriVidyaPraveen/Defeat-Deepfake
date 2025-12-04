# model.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class TemporalViT(nn.Module):
    def __init__(self, num_frames=8, num_classes=2):
        super(TemporalViT, self).__init__()
        # Load pretrained weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)
        
        # We need the feature dimension (usually 768 for Base ViT)
        self.embed_dim = 768 
        
        # Replace the head with Identity to get raw embeddings
        self.vit.heads = nn.Sequential(nn.Identity())
        
        self.num_frames = num_frames
        
        # Classification head aggregates features from all frames
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * num_frames, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_frames, 3, 224, 224)
        b, t, c, h, w = x.shape
        
        # Fold time into batch dimension: (b*t, c, h, w)
        x = x.view(b * t, c, h, w)
        
        # Pass through ViT backbone
        features = self.vit(x) # Output: (b*t, 768)
        
        # Unfold time: (b, t, 768)
        features = features.view(b, t, -1)
        
        # Flatten time and features: (b, t*768)
        features = features.reshape(b, -1)
        
        # Classify
        out = self.classifier(features)
        return out