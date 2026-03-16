import torch
import torch.nn as nn

class TemporalBranch(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=2):
        super(TemporalBranch, self).__init__()
        
        # We use a Bidirectional LSTM (BiLSTM) over the spatial features
        # This is massively more memory efficient than a full 3D CNN
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, sequence_features):
        # sequence_features shape: [Batch_size, Sequence_length, Feature_dimension]
        # This is the 'x' returned from the spatial branch alongside the mean
        
        # Process sequence
        output, (hn, cn) = self.lstm(sequence_features)
        
        # Extract features from the final timestep
        # Because it's bidirectional, we concatenate the final forward and backward states
        # hn shape: [num_layers * num_directions, batch, hidden_size]
        final_forward = hn[-2, :, :]
        final_backward = hn[-1, :, :]
        
        # Shape: [Batch_size, hidden_dim * 2]
        temporal_out = torch.cat((final_forward, final_backward), dim=1)
        
        return temporal_out
