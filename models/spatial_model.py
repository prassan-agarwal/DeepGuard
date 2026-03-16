import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class SpatialBranch(nn.Module):
    def __init__(self, feature_dim=256, pretrained=True):
        super(SpatialBranch, self).__init__()
        
        # Load MobileNetV3-Small (very lightweight, suitable for 4GB VRAM)
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v3_small(weights=weights)
        
        # Remove the classification head (classifier), keep the feature extractor
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MobileNetV3-Small outputs 576 channels before the final classifier
        # We project this down to our desired feature_dim
        self.fc = nn.Sequential(
            nn.Linear(576, feature_dim),
            nn.SiLU(), # Swish activation used in MobileNetV3
            nn.Dropout(p=0.2)
        )
        
    def forward(self, sequences):
        # sequences shape: [Batch_size, Sequence_length, Channels, Height, Width]
        b, s, c, h, w = sequences.size()
        
        # We need to process each frame independently through the spatial branch.
        # Reshape to treat (batch * sequence) as a massive batch of single frames
        x = sequences.view(b * s, c, h, w)
        
        # Extract spatial features
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # Shape: [(b * s), 576]
        
        # Project to feature_dim
        x = self.fc(x) # Shape: [(b * s), feature_dim]
        
        # Reshape back to sequence format
        # Shape: [Batch_size, Sequence_length, feature_dim]
        # This allows the temporal branch or fusion to process the sequence over time
        # Or, we can aggregate here (e.g., mean over sequence). We'll aggregate.
        x = x.view(b, s, -1)
        
        # Aggregate spatial features over time (mean pooling)
        # We output a single spatial feature vector per video
        spatial_out = torch.mean(x, dim=1) # Shape: [Batch_size, feature_dim]
        
        # Return both the per-frame features (for the Temporal branch) 
        # and the aggregated features (for final Fusion)
        return spatial_out, x
