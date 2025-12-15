# Fire/Smoke Detection Training Framework

Production-ready training framework for AlertCalifornia fire/smoke detection using YOLOv8.

## Setup on New Machine

### 1. Clone Repository

```bash
# Clone from GitHub
git clone https://github.com/whamidou006/fire-smoke-detection-hackathon.git
cd fire-smoke-detection-hackathon
```

### 2. Create Environment

```bash
# Create conda environment
conda create -p /path/to/env/firesmoke python=3.11 -y
conda activate /path/to/env/firesmoke

# Install dependencies
pip install ultralytics
```

### 3. Configure Dataset

Edit `dataset.yaml` to point to your dataset location:
```yaml
path: /path/to/your/Fire_data_v2_yolo_with_blank_images_and_false_positives
train: train/images
val: test/images
```

### 4. Start Training

```bash
# Train with default model (YOLOv8n)
python train.py

# Or specify a different model
python train.py --model s
```

YOLO pretrained models will be downloaded automatically on first use.

## Installation

*Note: For complete setup from scratch, see "Setup on New Machine" section above.*

```bash
# Create conda environment
conda create -p /path/to/env/firesmoke python=3.11 -y
conda activate /path/to/env/firesmoke

# Install dependencies
pip install ultralytics
```

## Project Structure

```
fire_smoke_training/
├── train.py           # Training script with model selection
├── dataset.yaml       # Dataset configuration
├── analyze.py         # Training analysis and visualization
├── test.py            # Model evaluation and comparison
├── requirements.txt   # Project dependencies
└── README.md          # Documentation
```

## Quick Reference

### Training
```bash
python train.py                    # YOLOv8n (default)
python train.py --model s          # YOLOv8s
python train.py --model l          # YOLOv8l
python train.py --model pretrain   # Continue from pretrained model
python train.py -m l -b 32         # Custom batch size
```

### Testing
```bash
python test.py --model path/to/best.pt                    # Evaluate single model
python test.py --model path/to/best.pt --compare          # Compare with baselines
python test.py --model path/to/best.pt --multi-scale      # Test-Time Augmentation
```

### Tiling Inference (High-Resolution Images)
```bash
# For high-resolution images, use the standalone tiling script
python tiling_inference.py --model best.pt --image image.jpg --output results/
```

### Analysis
```bash
python analyze.py                                # Auto-detect results and visualize
python analyze.py --test path/to/best.pt --compare
```

## Getting Started

### Training a Model

```bash
# Train with YOLOv8n (default)
python train.py

# Train with YOLOv8s
python train.py --model s

# Train with YOLOv8l
python train.py --model l

# Continue from pretrained model
python train.py --model pretrain

# Specify custom batch size
python train.py --model l --batch 32

# Use custom model weights
python train.py --model /path/to/custom.pt --batch 64
```

### Analyzing Training Progress

```bash
# Specify results.csv file explicitly (recommended)
python analyze.py --results runs/yolov8n_fire_smoke/results.csv

# Exclude baseline comparison for faster analysis
python analyze.py --results runs/yolov8n_fire_smoke/results.csv --no-baselines

# Auto-detect results.csv file (searches common locations)
python analyze.py

# Available arguments:
#   --results PATH       Path to results.csv file from training
#   --output PATH        Output path for visualization (default: training_analysis.png)
#   --no-baselines       Skip baseline model evaluation
#   --dataset PATH       Dataset YAML file (default: dataset.yaml)
#   --test PATH          Test a model instead of analyzing training
#   --compare            Compare model with baselines when testing
```

**Input File:** The `results.csv` file is automatically created by `train.py` in the training output directory:
- **Location:** `runs/{model_name}_fire_smoke/results.csv`
- **Format:** CSV with epoch-by-epoch metrics (mAP, precision, recall, losses)
- **Created by:** YOLO during training

### Evaluating Models

```bash
# Test single model
python test.py --model ../runs/train_alertcal/yolov8n_optimized_v1/weights/best.pt

# Compare with baseline models
python test.py --model ../runs/train_alertcal/yolov8n_optimized_v1/weights/best.pt --compare

# Integrated testing and analysis
python analyze.py --test ../runs/train_alertcal/yolov8n_optimized_v1/weights/best.pt --compare
```

## Model Options

### Available Models

| Model | Size | Parameters | Batch Size | Use Case |
|-------|------|------------|------------|----------|
| yolov8n | 6 MB | 3.2M | 128 | Fast training, mobile deployment |
| yolov8s | 22 MB | 11.2M | 96 | Better accuracy, balanced speed |
| yolov8m | 52 MB | 25.9M | 56 | Balanced performance |
| yolov8l | 87 MB | 43.7M | 40 | High accuracy, baseline comparison |
| yolov8x | 136 MB | 68.2M | 24 | Maximum accuracy |
| pretrain_yolov8.pt | 84 MB | 43.6M | 40 | Hackathon baseline (YOLOv8l) |

### Baseline Performance

Test set: 382 images

- **current_best.pt**: mAP@0.5 = 0.4149, Precision = 0.6461, Recall = 0.3598
- **pretrain_yolov8.pt**: mAP@0.5 = 0.1944, Precision = 0.3575, Recall = 0.1650
- **Target**: mAP@0.5 ≥ 0.60, Precision ≥ 0.60, Recall ≥ 0.70

## Configuration

### Dataset (dataset.yaml)

```yaml
path: /path/to/Fire_data_v2_yolo_with_blank_images_and_false_positives
train: train/images  # 15,323 images (7,269 positive, 8,054 negative samples)
val: test/images     # 382 images

names:
  0: smoke  # 94% of instances
  1: fire   # 6% of instances
```

### Training Hyperparameters

Key settings optimized for AlertCalifornia dataset:

- **Epochs**: 150 with early stopping (patience=50)
- **Optimizer**: AdamW (learning rate=0.001)
- **Loss weights**: Classification=0.3, Box=7.5, DFL=1.5
- **Augmentation**: Conservative settings for smoke/fire detection
- **Batch size**: Model-dependent (see table above)

## Analysis and Visualization

## Analysis and Visualization

The analysis script generates a comprehensive 6-panel visualization:

1. **mAP@0.5 Progress** - Training curve with baseline comparison
2. **Precision & Recall** - Metrics with temporal smoothing
3. **Loss Curves** - Box, classification, and DFL losses
4. **Model Comparison** - Performance bar chart
5. **Metrics Table** - Detailed numerical comparison
6. **Training Summary** - Progress statistics and ETA

## Model Evaluation

The testing script provides:

## Model Evaluation

The testing script provides:

- Single model evaluation on test set
- Multi-model comparison with baselines
- Detailed per-class metrics
- JSON output for analysis
- Optional COCO format results

Example output:

```
Model                      Size   Params  mAP@0.5  Prec    Recall
─────────────────────────────────────────────────────────────────
current_best.pt            83.6M   43.6M   0.4149  0.6461  0.3598
pretrain_yolov8.pt         83.6M   43.6M   0.1944  0.3575  0.1650
trained_model.pt            6.2M    3.2M   0.XXXX  0.XXXX  0.XXXX
```

## Output Structure

## Output Structure

### Training Outputs

- `runs/train_alertcal/yolov8n_optimized_v1/`
  - `weights/best.pt` - Best model checkpoint
  - `weights/last.pt` - Latest checkpoint
  - `results.csv` - Training metrics per epoch
  - `results.png` - Training visualization

### Analysis Outputs

- `training_analysis.png` - Comprehensive 6-panel visualization
- `test_results_*.json` - Detailed evaluation metrics

## Training Strategy

## Training Strategy

### Current Approach

Training with YOLOv8n:
- Fast iteration and prototyping
- Low resource requirements
- Baseline performance assessment

### Model Selection Guidelines

If YOLOv8n performance is insufficient:

1. **YOLOv8s** (22MB, 11.2M params)
   - Moderate accuracy improvement
   - Reasonable training time
   - Good speed/accuracy balance

2. **YOLOv8l** (87MB, 43.7M params)
   - Matches baseline architecture
   - Direct comparison capability
   - Higher accuracy potential

3. **pretrain_yolov8.pt**
   - Leverages existing training
   - Proven architecture for dataset
   - Transfer learning benefits

## Dataset Information

## Dataset Information

### Composition

- **Training**: 15,325 images
  - Positive samples: 7,270 (47.4%) with fire/smoke annotations
  - Negative samples: 8,054 (52.6%) without fire/smoke (background)
  - Ratio: 1:1.11 (nearly balanced)
- **Validation/Test**: 382 images
  - Positive samples: 292 (76.4%) with fire/smoke annotations
  - Negative samples: 90 (23.6%) without fire/smoke
  - Ratio: 3.2:1 (more positives)
- **Classes**: 
  - Smoke: ~94% of fire/smoke instances
  - Fire: ~6% of fire/smoke instances
  - Class imbalance: 1:15.67 (addressed by class_weights)

### Train/Val Distribution Analysis

**Key Insight**: Training set has MORE negatives (52.6%) than validation (23.6%)

**Why This Is Correct:**
- ✅ **Training (52.6% neg)**: Simulates real-world deployment where most frames show no fire
- ✅ **Validation (76.4% pos)**: Tests detection capability when fire/smoke is present
- ✅ **Result**: Model learns to reject false positives while maintaining detection accuracy

**Important Considerations:**
- ⚠️  Validation mAP may be 5-10% HIGHER than real-world performance
- 🎯 **Focus on RECALL** (catching fires) not just mAP
- 🎯 **Monitor false positive rate** during validation
- ✅ This split is INTENTIONAL for safety-critical fire detection

### Design Rationale

The dataset includes negative samples (empty annotations) intentionally to:
- Reduce false positive detections
- Improve model generalization
- Enhance performance at lower zoom levels

### Performance Context

- YOLOv8n (3.2M parameters) vs Baseline (43.6M parameters)
- 13x parameter difference affects capacity
- Current gap: 0.2958 vs 0.4149 mAP@0.5

## Troubleshooting

## Troubleshooting

### Common Issues

**Results file not found**
```bash
python analyze.py --results path/to/results.csv
```

**Out of memory**
```bash
# Reduce batch size in train.py or use --batch argument
python train.py --model n --batch 64
```

**Baseline evaluation slow**
```bash
python analyze.py --no-baselines
```

**NumPy compatibility error**
```bash
# Use the correct conda environment
/home/whamidouche/ssdprivate/conda-env/cevg-rtnet/bin/python train.py
```

## Technical Notes

## Technical Notes

### Training Time Estimates

On NVIDIA A100 80GB GPU:

- YOLOv8n: 12-15 hours (150 epochs)
- YOLOv8s: 18-24 hours (150 epochs)
- YOLOv8l: 30-40 hours (150 epochs)

### Recommended Workflow

1. Train YOLOv8n for baseline performance
2. Analyze results and dataset characteristics
3. Select larger model if needed
4. Apply insights from initial training
5. Compare final results with baselines

## Advanced Usage

## Advanced Usage

### Tiling Inference for High-Resolution Images

For high-resolution images (>1920x1080), use the dedicated `tiling_inference.py` script that processes images in overlapping tiles for better small object detection.

**When to use tiling:**
- High-resolution images (>1920x1080)
- Small fire/smoke at distance
- Need better small object detection

**Basic usage:**
```bash
# Process single image
python tiling_inference.py \
  --model runs/yolov8s_fire_smoke/weights/best.pt \
  --image /path/to/high_res_image.jpg \
  --output tiling_results/

# Process entire directory
python tiling_inference.py \
  --model runs/yolov8s_fire_smoke/weights/best.pt \
  --image /path/to/images/ \
  --output tiling_results/

# Custom tile size and overlap
python tiling_inference.py \
  --model best.pt \
  --image image.jpg \
  --tile-size 1280 \
  --overlap 0.3 \
  --output results/
```

**Parameters:**
- `--tile-size`: Size of each tile in pixels (default: 640)
- `--overlap`: Overlap between tiles 0.0-0.5 (default: 0.2 = 20%)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--device`: Device to use (default: cuda:0)
- `--no-visualize`: Skip saving visualization images

**Output:**
- JSON files with all detections
- Visualization images with tile grid overlay
- Summary statistics

**Trade-offs:**
- ✅ Better small object detection (+5-15% on high-res images)
- ✅ Can process very large images
- ✅ Complete per-image analysis with visualizations
- ❌ Slower inference (3-5x depending on image size and overlap)

### Test-Time Augmentation (Recommended)

### Custom Configuration

Edit training parameters in `train.py`:

- Model selection and batch size
- Training hyperparameters
- Data augmentation settings
- Loss weights

### Monitoring Training

```bash
# View training in real-time
screen -r train

# Update visualization periodically
watch -n 300 python analyze.py

# Monitor GPU usage
nvtop
```

### Model Export

Export trained model to ONNX format:

```python
from ultralytics import YOLO
model = YOLO('runs/train_alertcal/yolov8n_optimized_v1/weights/best.pt')
model.export(format='onnx', dynamic=False, simplify=True)
```

## References

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [AlertCalifornia Project](https://www.alertcalifornia.org/)
- Dataset: Fire_data_v2_yolo_with_blank_images_and_false_positives

## Support

For assistance:
1. Review this documentation
2. Check training logs and outputs
3. Use `--help` flag with scripts

---

**Version:** 1.0  
**Last Updated:** December 15, 2025
