import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_model import DeepfakeHybridModel
from utils.dataset_loader import DeepfakeDataset

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor):
        self.model.eval()
        # Hack to allow cuDNN RNN backward in eval mode
        if hasattr(self.model, 'temporal') and hasattr(self.model.temporal, 'lstm'):
            self.model.temporal.lstm.train()
            
        self.model.zero_grad()
        
        # Requires gradients for the input (though we hook interior layer)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        logits = self.model(input_tensor)
        
        # Backward pass
        # Since logit is BCE, we push 1.0 backward
        logits.backward()
        
        # Activations/Gradients shape: [Batch*Seq, C, H, W]
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global Average Pooling on gradients -> [Batch*Seq, C, 1, 1]
        weights = torch.mean(gradients, dim=[-2, -1], keepdim=True)
        
        # Weight the activations -> [Batch*Seq, C, H, W]
        cam = weights * activations
        
        # Sum across channels -> [Batch*Seq, H, W]
        cam = torch.sum(cam, dim=1)
        
        # ReLU to keep only positive influences
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        for i in range(cam.shape[0]):
            cam_min, cam_max = cam[i].min(), cam[i].max()
            if cam_max > cam_min:
                cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)
                
        return cam, torch.sigmoid(logits).item()

def overlay_cam(img, cam, alpha=0.5):
    """img in [H, W, 3] uint8, cam in [H, W] float"""
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed = (heatmap * alpha + img * (1 - alpha)).astype(np.uint8)
    return superimposed

def denormalize(tensor):
    """Denormalize a tensor back to [0, 255] uint8 for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = DeepfakeHybridModel()
    model_path = "best_hybrid_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # 2. Setup GradCAM on the final feature block of MobileNetV3
    target_layer = model.spatial.features[-1]
    cam_extractor = GradCAM(model, target_layer)
    
    # 3. Load dataset
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading dataset...")
    dataset = DeepfakeDataset("dataset/processed/real", "dataset/processed/fake", sequence_length=16, transform=transform)
    
    # Create output directory
    os.makedirs("inference_results", exist_ok=True)
    
    # 4. Pick one Real and one Fake video sequence
    # Assuming the first part is real and the last part is fake based on dataset loader logic
    sample_indices = [0, len(dataset) - 1] 
    
    for idx in sample_indices:
        sequence_tensor, label_tensor = dataset[idx]
        sequence_batch = sequence_tensor.unsqueeze(0).to(device) # [1, 16, 3, 224, 224]
        true_label = int(label_tensor.item())
        label_str = "Fake" if true_label == 1 else "Real"
        
        print(f"Generating GradCAM for {label_str} sequence (Index {idx})...")
        
        # Generate CAM
        cams, pred_prob = cam_extractor.generate(sequence_batch)
        pred_label_str = "Fake" if pred_prob > 0.5 else "Real"
        print(f"  -> True: {label_str} | Pred: {pred_label_str} (Score: {pred_prob:.4f})")
        
        # Plot a few selected frames from the sequence (e.g., first, middle, last)
        frames_to_plot = [0, 7, 15] # Out of 16
        
        fig, axes = plt.subplots(len(frames_to_plot), 3, figsize=(10, 10))
        fig.suptitle(f"True: {label_str} | Pred: {pred_label_str} ({pred_prob:.2f})", fontsize=16)
        
        for row_idx, f_idx in enumerate(frames_to_plot):
            # Original Image
            img_tensor = sequence_tensor[f_idx]
            img_rgb = denormalize(img_tensor)
            
            # Heatmap Image
            cam = cams[f_idx]
            heatmap_rgb = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
            
            # Overlay Image
            overlay_rgb = overlay_cam(img_rgb, cam)
            
            axes[row_idx, 0].imshow(img_rgb)
            axes[row_idx, 0].set_title(f"Frame {f_idx}")
            axes[row_idx, 0].axis('off')
            
            axes[row_idx, 1].imshow(heatmap_rgb)
            axes[row_idx, 1].set_title("GradCAM")
            axes[row_idx, 1].axis('off')
            
            axes[row_idx, 2].imshow(overlay_rgb)
            axes[row_idx, 2].set_title("Overlay")
            axes[row_idx, 2].axis('off')
            
        plt.tight_layout()
        output_path = f"inference_results/gradcam_{label_str.lower()}_seq.png"
        plt.savefig(output_path)
        plt.close()
        print(f"  -> Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
