import torch
import sys
sys.path.append('.')
from models.hybrid_model import DeepfakeHybridModel

m = DeepfakeHybridModel()
m.load_state_dict(torch.load('best_hybrid_model.pth', map_location='cpu'))
m.eval()
d = torch.randn(1, 16, 3, 224, 224)
try:
    torch.onnx.export(m, d, 'test.onnx', opset_version=14)
except Exception as e:
    print("Caught ONNX error:")
    # print the first 10 lines of the error message to see the exact op
    print('\n'.join(str(e).split('\n')[:10]))
