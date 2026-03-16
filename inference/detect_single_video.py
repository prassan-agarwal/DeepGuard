import os
import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from torchvision import transforms
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_model import DeepfakeHybridModel

def process_single_video(video_path, model_path=None, sequence_length=16):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_hybrid_model.pth")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = DeepfakeHybridModel()
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 2. Extract and Process Frames
    print(f"Processing video: {video_path}")
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print("Error: Could not open video or video is empty.")
        return
        
    # Sample indices
    indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
    
    processed_frames = []
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Extracting {sequence_length} frames and detecting faces...")
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Detect Face
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            face = rgb_frame[y:y+h, x:x+w]
        else:
            # Fallback: simple center crop if no face detected
            h, w, c = rgb_frame.shape
            min_dim = min(h, w)
            start_y = (h - min_dim) // 2
            start_x = (w - min_dim) // 2
            face = rgb_frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
            
        # Apply transforms
        face_tensor = transform(face)
        processed_frames.append(face_tensor)
        
    cap.release()
    
    if len(processed_frames) < sequence_length:
        print(f"Warning: Only extracted {len(processed_frames)} frames. Padding...")
        while len(processed_frames) < sequence_length:
            processed_frames.append(processed_frames[-1])
            
    # Stack into [1, Sequence, C, H, W]
    input_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)
    
    # 3. Inference
    print("Running inference...")
    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()
        
    is_fake = probability >= 0.5
    confidence = probability if is_fake else (1 - probability)
    
    print("\n" + "="*30)
    print(f"RESULT: {'FAKE' if is_fake else 'REAL'}")
    print(f"Probability of being Fake: {probability:.4f}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference/detect_single_video.py <path_to_video>")
    else:
        video_file = sys.argv[1]
        process_single_video(video_file)
