import torch
import UGCVQA_NR_model
from torchvision import transforms

def convert_model_to_onnx():
    # 1. Khởi tạo model và load weights
    model = UGCVQA_NR_model.resnet50(pretrained=True)
    model.load_state_dict(torch.load('ckpts/UGCVQA_NR_model.pth'))
    model.eval()

    # 2. Tạo input dummy với kích thước phù hợp
    # Dựa vào data_loader.py và test_NR_demo.py:
    # - Batch size: 1
    # - Số frame: video_length_read (phụ thuộc vào fps của video)
    # - Channel: 3 (RGB)
    # - Height & Width: 448 (sau khi transform)
    batch_size = 1
    num_frames = 32  # Có thể điều chỉnh tùy theo use case
    channels = 3
    height = 448
    width = 448
    
    dummy_input = torch.randn(batch_size, num_frames, channels, height, width)

    # 3. Export sang ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "UGCVQA_NR_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_video'],
        output_names=['quality_score'],
        dynamic_axes={
            'input_video': {
                0: 'batch_size',
                1: 'num_frames'
            },
            'quality_score': {
                0: 'batch_size'
            }
        }
    )

if __name__ == "__main__":
    convert_model_to_onnx()

    import onnx
    onnx_model = onnx.load("UGCVQA_NR_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("Model đã được chuyển đổi và kiểm tra thành công!")