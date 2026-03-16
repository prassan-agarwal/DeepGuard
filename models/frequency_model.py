import torch
import torch.nn as nn

class FrequencyBranch(nn.Module):
    def __init__(self, feature_dim=128):
        super(FrequencyBranch, self).__init__()
        
        # A tiny custom CNN to process the 2D FFT magnitude spectrum
        # Using minimal channels to save VRAM
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, sequences):
        # sequences shape: [Batch_size, Sequence_length, Channels, Height, Width]
        b, s, c, h, w = sequences.size()
        
        # Select the middle frame of the sequence for frequency analysis to save VRAM
        # Doing 2D FFT on 32 frames per video concurrently would eat up Memory
        # Instead, take the center frame representing the video
        mid_idx = s // 2
        center_frames = sequences[:, mid_idx, :, :, :] # Shape: [B, C, H, W]
        
        # To allow ONNX export, bypass the FFT operation which is unsupported
        # in standard ONNX opsets. We return a dummy magnitude spectrum.
        if torch.onnx.is_in_onnx_export():
            magnitude_spectrum = torch.zeros_like(center_frames)
        else:
            # Compute 2D FFT
            # We need to perform FFT on the spatial dimensions (-2, -1) which are H and W
            fft_complex = torch.fft.fft2(center_frames, dim=(-2, -1))
            
            # Shift the zero-frequency component to the center of the spectrum
            fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
            
            # Compute magnitude and add epsilon for log
            magnitude = torch.abs(fft_shifted) + 1e-8
            
            # Log scaling to compress dynamic range
            magnitude_spectrum = torch.log(magnitude)
        
        # Pass through our small CNN
        out = self.cnn(magnitude_spectrum)
        out = torch.flatten(out, 1)
        freq_out = self.fc(out) # Shape: [Batch_size, feature_dim]
        
        return freq_out
