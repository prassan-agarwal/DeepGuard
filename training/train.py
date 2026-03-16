import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.dataset_loader import DeepfakeDataset
from models.hybrid_model import DeepfakeHybridModel

def get_transforms():
    # Basic transforms for our 224x224 models
    # We apply these to each frame independently
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Standard ImageNet normalization since MobileNet is pretrained on it
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_model(real_dir, fake_dir, num_epochs=20, batch_size=2, accumulation_steps=4, sequence_length=16):
    
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Optional but highly recommended for RTX 3050 Ti:
    # torch.backends.cudnn.benchmark = True
    
    # 2. Dataset and DataLoader setup
    transform = get_transforms()
    dataset = DeepfakeDataset(real_dir=real_dir, fake_dir=fake_dir, sequence_length=sequence_length, transform=transform)
    
    # Important: Small batch size to fit in 4GB VRAM
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    print(f"Dataset loaded. Total sequences (videos): {len(dataset)}")
    
    # 3. Model Initialization
    # Make sure to import this from the appropriate file
    model = DeepfakeHybridModel()
    model = model.to(device)
    
    # 4. Loss and Optimizer
    # BCEWithLogitsLoss combines Sigmoid and Binary Cross Entropy for numeric stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 5. Mixed Precision Setup
    # This scaler handles FP16 gradients to prevent underflow
    scaler = torch.cuda.amp.GradScaler()
    
    # 6. Training Loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # We start with cleared gradients
        optimizer.zero_grad()
        
        # Use tqdm for progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (sequences, labels) in progress_bar:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1) # [B, 1]
            
            # Forward pass with Automatic Mixed Precision
            with torch.cuda.amp.autocast():
                logits = model(sequences)
                loss = criterion(logits, labels)
                
                # Normalize loss to account for gradient accumulation
                loss = loss / accumulation_steps
                
            # Backward pass (scaled)
            scaler.scale(loss).backward()
            
            # Update weights only after accumulation_steps have passed
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Step optimizer and update scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # Slightly more memory efficient
            
            # Tracking metrics
            running_loss += loss.item() * accumulation_steps # Un-normalize for display
            
            # Calculate accuracy
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Formatting progress bar
            progress_bar.set_postfix({
                'Loss': f"{running_loss / (batch_idx + 1):.4f}", 
                'Acc': f"{correct_predictions / total_samples:.4f}"
            })
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_samples
        print(f"Epoch {epoch+1} Summary | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
        
        # Simple checkpointing
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            print("Saved new best model checkpoint.")

if __name__ == "__main__":
    
    # Update these paths to point to your processed local dataset
    REAL_DIR = "dataset/processed/real"
    FAKE_DIR = "dataset/processed/fake"
    
    # Run the training loop!
    train_model(
        real_dir=REAL_DIR, 
        fake_dir=FAKE_DIR, 
        num_epochs=5,       # Increase for real training
        batch_size=2,       # Keep very small (2) for 4GB VRAM
        accumulation_steps=4, # Simulates batch_size of 8
        sequence_length=16  # Load 16 evenly spaced frames per video
    )
