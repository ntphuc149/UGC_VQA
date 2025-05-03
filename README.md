# UGC_VQA: Video Quality Assessment for User Generated Content

A deep learning framework for assessing the quality of user-generated videos using No-Reference (NR) and Full-Reference (FR) methods.

## Overview

This project provides state-of-the-art video quality assessment models specifically designed for user-generated content (UGC). The framework supports both No-Reference (NR) and Full-Reference (FR) evaluation methods.

<img src="model-architecture.png" alt="Model Architecture" width="800"/>

### Key Features

- 🚀 **No-Reference (NR) Assessment**: Evaluate video quality without reference videos
- 📊 **Full-Reference (FR) Assessment**: Compare distorted videos with reference videos
- 🔄 **Multi-scale Processing**: Support for multiple video resolutions
- ⚡ **GPU Acceleration**: Optimized for CUDA-enabled devices
- 🔧 **ONNX Export**: Convert trained models to ONNX format for deployment
- 🧪 **Pre-trained Models**: Ready-to-use checkpoints included

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.1+ (for GPU support)

### Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/UGC_VQA.git
cd UGC_VQA

# Install required packages
pip install torch torchvision opencv-python pillow pandas numpy scipy
```

### Project Structure

```
UGC_VQA/
├── UGCVQA_NR_model.py      # No-Reference model implementation
├── UGCVQA_FR_model.py      # Full-Reference model implementation
├── data_loader.py          # Dataset loading utilities
├── train_NR.py             # Training script for NR model
├── train_FR.py             # Training script for FR model
├── test_NR.py              # Testing script for NR model
├── test_FR.py              # Testing script for FR model
├── test_NR_demo.py         # Demo script for NR inference
├── test_FR_demo.py         # Demo script for FR inference
├── convert_pth_2_onnx.py   # Model conversion to ONNX
├── config.pbtxt           # Triton Inference Server config
└── ckpts/                 # Model checkpoints directory
```

## Models

### No-Reference (NR) Model

The NR model assesses video quality without requiring reference videos. It utilizes a modified ResNet-50 architecture with hyper-structure connections for enhanced feature extraction.

**Key Components:**
- Multi-scale feature extraction
- Temporal pooling for video sequence processing
- Global average and standard deviation pooling

### Full-Reference (FR) Model

The FR model compares distorted videos with reference videos to predict quality scores. It employs structural similarity computations across multiple feature levels.

**Key Components:**
- Dual-stream architecture for reference and distorted videos
- Multi-level similarity computation
- Temporal pooling for sequence aggregation

## Usage

### Training

#### No-Reference Model

```bash
python train_NR.py \
    --database UGCCompressed \
    --model_name UGCVQA \
    --conv_base_lr 0.00001 \
    --datainfo json_files/ugcset_mos.json \
    --videos_dir /path/to/videos \
    --epochs 100 \
    --train_batch_size 4 \
    --ckpt_path ckpts/
```

#### Full-Reference Model

```bash
python train_FR.py \
    --database UGCCompressed \
    --model_name UGCVQA \
    --conv_base_lr 0.00001 \
    --datainfo json_files/ugcset_dmos.json \
    --videos_dir /path/to/videos \
    --epochs 100 \
    --train_batch_size 4 \
    --ckpt_path ckpts/
```

### Testing

#### No-Reference Testing

```bash
python test_NR.py \
    --database UGCCompressed \
    --videos_dir_test /path/to/test/videos \
    --datainfo_test json_files/ugcset_mos.json \
    --trained_model ckpts/UGCVQA_NR_model.pth \
    --output_name NR_output.txt
```

#### Full-Reference Testing

```bash
python test_FR.py \
    --database UGCCompressed \
    --videos_dir_test /path/to/test/videos \
    --datainfo_test json_files/ugcset_dmos.json \
    --trained_model ckpts/UGCVQA_FR_model.pth \
    --output_name FR_output.txt
```

### Demo Usage

#### No-Reference Demo

```bash
# Single-scale assessment
python test_NR_demo.py \
    --method_name single-scale \
    --dist /path/to/video.mp4 \
    --output result.txt \
    --is_gpu

# Multi-scale assessment
python test_NR_demo.py \
    --method_name multi-scale \
    --dist /path/to/video.mp4 \
    --output result.txt \
    --is_gpu
```

#### Full-Reference Demo

```bash
# Single-scale assessment
python test_FR_demo.py \
    --method_name single-scale \
    --ref /path/to/reference.mp4 \
    --dist /path/to/distorted.mp4 \
    --output result.txt \
    --is_gpu

# Multi-scale assessment
python test_FR_demo.py \
    --method_name multi-scale \
    --ref /path/to/reference.mp4 \
    --dist /path/to/distorted.mp4 \
    --output result.txt \
    --is_gpu
```

### ONNX Conversion

```bash
python convert_pth_2_onnx.py \
    --model_path ckpts/UGCVQA_NR_model.pth \
    --onnx_path ckpts/UGCVQA_NR_model.onnx
```

## Dataset Format

The framework expects dataset information in JSON format:

```json
{
    "train": {
        "dis": ["video1.mp4", "video2.mp4", ...],
        "mos": [3.5, 4.2, ...]
    },
    "test": {
        "dis": ["video3.mp4", "video4.mp4", ...],
        "mos": [2.8, 4.5, ...]
    }
}
```

For FR models, additional reference video lists are required:

```json
{
    "train": {
        "ref": ["ref1.mp4", "ref2.mp4", ...],
        "dis": ["dist1.mp4", "dist2.mp4", ...],
        "mos": [3.5, 4.2, ...]
    }
}
```

## Model Architecture Details

### No-Reference Architecture

The NR model uses a modified ResNet-50 backbone with:
- Hyper-structure connections between layers
- Multi-level feature aggregation
- Temporal pooling for video processing

### Full-Reference Architecture

The FR model employs:
- Parallel processing of reference and distorted videos
- Structural similarity computation at multiple scales
- Feature-level comparison and aggregation

## Performance Metrics

The models are evaluated using:
- **SROCC**: Spearman Rank-Order Correlation Coefficient
- **PLCC**: Pearson Linear Correlation Coefficient
- **KRCC**: Kendall Rank-Order Correlation Coefficient
- **RMSE**: Root Mean Square Error

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{ugcvqa2024,
  title={UGC_VQA: Video Quality Assessment for User Generated Content},
  author={Your Name},
  booktitle={Proceedings of the Conference},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ResNet architecture from [torchvision](https://pytorch.org/vision/stable/models.html)
- Temporal pooling inspired by subjective assessment principles

## Contact

For any questions or issues, please open an issue in this repository or contact [your.email@domain.com](mailto:your.email@domain.com).

## Updates

- **v1.0.0**: Initial release with NR and FR models
- **v1.1.0**: Added ONNX export functionality
- **v1.2.0**: Improved multi-scale processing

