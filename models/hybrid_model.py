import torch
import torch.nn as nn
from models.spatial_model import SpatialBranch
from models.temporal_model import TemporalBranch
from models.frequency_model import FrequencyBranch

class DeepfakeHybridModel(nn.Module):
    def __init__(self, spatial_dim=256, temporal_hidden=128, freq_dim=128):
        super(DeepfakeHybridModel, self).__init__()
        
        # Initialize the three branches
        self.spatial = SpatialBranch(feature_dim=spatial_dim, pretrained=True)
        # BiLSTM outputs hidden_size * 2 (because it's bidirectional)
        self.temporal = TemporalBranch(input_dim=spatial_dim, hidden_dim=temporal_hidden, num_layers=1)
        self.frequency = FrequencyBranch(feature_dim=freq_dim)
        
        # Calculate combined dimension
        # spatial_dim + (temporal_hidden * 2) + freq_dim
        # e.g., 256 + 256 + 128 = 640
        combined_dim = spatial_dim + (temporal_hidden * 2) + freq_dim
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # Single output logit for BCEWithLogitsLoss
        )
        
    def forward(self, sequences):
        # 1. Spatial Branch
        # spatial_out represents the aggregated (mean) spatial features across the sequence
        # sequence_features represents the features per frame in the sequence
        spatial_out, sequence_features = self.spatial(sequences)
        
        # 2. Temporal Branch
        # Feed the per-frame features directly into BiLSTM
        temporal_out = self.temporal(sequence_features)
        
        # 3. Frequency Branch
        # Feed raw sequences to calculate FFT
        freq_out = self.frequency(sequences)
        
        # 4. Feature Fusion
        # Concatenate outputs from all three branches along the feature dimension
        fused_features = torch.cat((spatial_out, temporal_out, freq_out), dim=1)
        
        # 5. Final Classification
        # Outputs a raw logit
        logits = self.classifier(fused_features)
        
        return logits
