import torch
import UGCVQA_NR_model
import argparse

def convert_to_onnx(model_path, onnx_path):
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU and CUDA installation.")
        return

    # 1. Load model
    model = UGCVQA_NR_model.resnet50(pretrained=False)
    model.load_state_dict(torch.load(model_path))  # Load on default device (GPU)
    model = model.to(device)  # Move model to GPU
    model.eval()
    
    # 2. Create dummy input on GPU
    dummy_input = torch.randn(1, 30, 3, 448, 448, device=device)
    
    # 3. Export to ONNX
    torch.onnx.export(
        model,                    # Model being exported
        dummy_input,             # Dummy input
        onnx_path,               # Output file
        export_params=True,      # Store trained weights
        opset_version=11,        # ONNX version
        do_constant_folding=True,# Optimize constant folding
        input_names=['input'],   # Input names
        output_names=['score'],  # Output names
        dynamic_axes={           # Dynamic axes
            'input': {0: 'batch_size', 1: 'num_frames'},
            'score': {0: 'batch_size'}
        }
    )

    print(f"Model has been converted to ONNX and saved at {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='ckpts/UGCVQA_NR_model.pth')
    parser.add_argument('--onnx_path', type=str, default='ckpts/UGCVQA_NR_model.onnx')
    args = parser.parse_args()
    
    convert_to_onnx(args.model_path, args.onnx_path)