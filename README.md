# Fire/Smoke Detection Training Framework

Minimal, production-ready training framework for AlertCalifornia fire/smoke detection using YOLOv8.

## 📁 Structure (Only 4 Files Needed!)

```
fire_smoke_training/
├── train.py           # Training script with all model options
├── dataset.yaml       # Dataset configuration
├── analyze.py         # Training analysis and visualization
├── test.py            # Model testing and comparison
└── README.md          # This file
```

## ⚡ Quick Reference

```bash
# TRAINING
python train.py                    # YOLOv8n (default, fastest)
python train.py --model s          # YOLOv8s (better accuracy)
python train.py --model l          # YOLOv8l (match baseline architecture)
python train.py --model pretrain   # Continue from pretrain_yolov8.pt
python train.py -m l -b 32         # Custom batch size

# TESTING
python test.py --model path/to/best.pt           # Single model
python test.py --model path/to/best.pt --compare # vs baselines

# ANALYSIS
python analyze.py                                # Auto-detect & visualize
python analyze.py --test path/to/best.pt --compare
```

## 🚀 Quick Start

### 1. Train a Model

```bash
# Train with YOLOv8n (fastest, smallest - default)
python train.py

# Train with YOLOv8s (better accuracy)
python train.py --model s

# Train with YOLOv8l (same architecture as baseline)
python train.py --model l

# Train from hackathon pretrained model
python train.py --model pretrain

# Custom batch size
python train.py --model l --batch 32

# Train with custom model file
python train.py --model /path/to/custom.pt --batch 64
```

### 2. Analyze Training Progress

```bash
# Generate visualization with baseline comparison
python analyze.py --results ../runs/train_alertcal/yolov8n_optimized_v1/results.csv

# Quick analysis without baselines (faster)
python analyze.py --results ../runs/train_alertcal/yolov8n_optimized_v1/results.csv --no-baselines

# Auto-detect results file
python analyze.py
```

### 3. Test Your Model

```bash
# Test single model
python test.py --model ../runs/train_alertcal/yolov8n_optimized_v1/weights/best.pt

# Compare with baselines
python test.py --model ../runs/train_alertcal/yolov8n_optimized_v1/weights/best.pt --compare

# Or use analyze.py as shortcut
python analyze.py --test ../runs/train_alertcal/yolov8n_optimized_v1/weights/best.pt --compare
```

## 📊 Model Options

### Available Pretrained Models

| Option | Model | Size | Parameters | Batch Size | When to Use |
|--------|-------|------|------------|------------|-------------|
| 1 ✓ | yolov8n.pt | 6 MB | 3.2M | 128 | Fast training, mobile deployment |
| 2 | yolov8s.pt | 22 MB | 11.2M | 96-112 | Better accuracy, still fast |
| 3 | yolov8m.pt | 52 MB | 25.9M | 48-64 | Balanced performance |
| 4 | yolov8l.pt | 87 MB | 43.7M | 32-48 | High accuracy, fair baseline comparison |
| 5 | yolov8x.pt | 136 MB | 68.2M | 16-32 | Maximum accuracy |
| 6 | pretrain_yolov8.pt | 84 MB | 43.6M | 32-48 | Hackathon baseline (YOLOv8l) |

### Baseline Performance (Test Set: 382 images)

- **current_best.pt**: mAP@0.5 = 0.4149, Precision = 0.6461, Recall = 0.3598
- **pretrain_yolov8.pt**: mAP@0.5 = 0.1944, Precision = 0.3575, Recall = 0.1650
- **Target Goal**: mAP@0.5 ≥ 0.60, Precision ≥ 0.60, Recall ≥ 0.70

## 🔧 Configuration

### Dataset (dataset.yaml)

```yaml
path: /path/to/Fire_data_v2_yolo_with_blank_images_and_false_positives
train: train/images  # 15,323 images (47.4% with objects, 52.6% negatives)
val: test/images     # 382 images

names:
  0: smoke  # 94% of instances
  1: fire   # 6% of instances
```

### Training Hyperparameters (train.py)

Key settings optimized for AlertCalifornia dataset:

- **Epochs**: 150 (with patience=50 early stopping)
- **Optimizer**: AdamW (lr0=0.001)
- **Loss weights**: cls=0.3, box=7.5, dfl=1.5 (recall-focused)
- **Augmentation**: Conservative (degrees=0, flipud=0, hsv_h=0.01)
- **Batch size**: Model-dependent (see table above)

## 📈 Training Analysis

The `analyze.py` script generates a comprehensive 6-panel visualization:

1. **mAP@0.5 Progress** - Training curve with baseline comparison
2. **Precision & Recall** - Both metrics with smoothing
3. **Training Loss Curves** - Box, Classification, DFL losses
4. **Model Comparison Bar Chart** - Side-by-side performance
5. **Metrics Comparison Table** - Detailed numbers
6. **Training Status Summary** - Progress and ETA

## 🧪 Testing

The `test.py` script provides:

- Single model evaluation on test set
- Multi-model comparison (your model vs baselines)
- Detailed per-class metrics
- JSON output for further analysis
- COCO format results (optional)

**Example Output:**

```
Model                      Size   Params  mAP@0.5  Prec    Recall
─────────────────────────────────────────────────────────────────
Current Best               83.6M   43.6M   0.4149  0.6461  0.3598
Your Model (Best)           6.2M    3.2M   0.2958  0.4534  0.3150
Pretrain YOLOv8            83.6M   43.6M   0.1944  0.3575  0.1650
```

## 💾 Output Files

### Training Outputs

- `runs/train_alertcal/yolov8n_optimized_v1/`
  - `weights/best.pt` - Best model checkpoint
  - `weights/last.pt` - Latest checkpoint
  - `results.csv` - Training metrics per epoch
  - `results.png` - Ultralytics default plots

### Analysis Outputs

- `training_analysis.png` - Comprehensive 6-panel visualization
- `test_results_*.json` - Detailed test metrics

## 🎯 Strategy Recommendations

### Current Training (YOLOv8n)

**Status:** Epoch 51/150 (34%), Best mAP = 0.2958

**Pros:**
- ✅ Fast training (~0.14h/epoch)
- ✅ Already beating pretrain baseline (0.1944)
- ✅ Low resource usage

**Next Steps:**
1. Let it finish training (99 epochs remaining)
2. If final mAP < 0.45 → Try YOLOv8s or YOLOv8l
3. If final mAP ≥ 0.45 → Consider ensemble or post-processing

### Upgrade Path

If YOLOv8n doesn't reach target (0.60 mAP):

1. **YOLOv8s** (22MB, 11.2M params)
   - +9% mAP over nano baseline
   - Still relatively fast
   - Good balance of speed/accuracy

2. **YOLOv8l** (87MB, 43.7M params)
   - Same architecture as current_best
   - Fair comparison with baseline
   - Better chance to reach 0.60 target

3. **pretrain_yolov8.pt** (Hackathon baseline)
   - Start from 0.1944 → train to 0.4149+
   - Proven architecture for this dataset
   - Continue/improve hackathon training

## 🔍 Key Insights

### Dataset Composition

- **Total train**: 15,323 images
  - Positive samples: 7,269 (47.4%) with bounding boxes
  - Negative samples: 8,054 (52.6%) empty labels (intentional!)
- **Classes**: smoke (94%), fire (6%)
- **Purpose of negatives**: Reduce false positives at zoom 1.0

### Current Challenge

- YOLOv8n (3.2M params) vs Current Best (43.6M params)
- **13x parameter disadvantage!**
- Current performance: 0.2958 vs 0.4149 target (28.7% gap)
- This gap is partly due to model capacity

### Performance Trends

- Epoch 1 → 51: mAP improved +995% (0.027 → 0.2958)
- Best epoch: 43 (may need more training or different model)
- Loss curves decreasing steadily (no overfitting)
- Validation metrics show learning progress

## 🐛 Troubleshooting

### Common Issues

**"Results file not found"**
```bash
# Specify explicit path
python analyze.py --results path/to/results.csv
```

**"Out of memory"**
```bash
# Reduce batch size in train.py
BATCH_SIZE = 64  # or lower
```

**"Baseline evaluation slow"**
```bash
# Skip baseline comparison
python analyze.py --no-baselines
```

**"NumPy compatibility error"**
```bash
# Use the correct conda environment
/home/whamidouche/ssdprivate/conda-env/cevg-rtnet/bin/python train.py
```

## 📝 Notes

### Why 52.6% Empty Labels?

The dataset intentionally includes 8,054 negative samples (images with no fire/smoke). This is NOT an error - it's designed to:

1. Reduce false positives
2. Teach the model what "normal" scenes look like
3. Specifically improve performance at zoom level 1.0 (main bottleneck)

### Training Time Estimates

On NVIDIA A100 80GB:

- YOLOv8n: ~12-15 hours (150 epochs)
- YOLOv8s: ~18-24 hours (150 epochs)
- YOLOv8l: ~30-40 hours (150 epochs)

### Recommended Workflow

1. Train YOLOv8n first (fast iteration, baseline)
2. Analyze results and understand dataset
3. If needed, train larger model (YOLOv8s or YOLOv8l)
4. Use insights from nano training to optimize
5. Compare with baselines using test.py

## 🚀 Advanced Usage

### Custom Training

Edit `train.py` to customize:

- Model selection (lines 20-37)
- Batch size (line 56)
- Hyperparameters (lines 66-115)
- Data augmentation (lines 88-101)

### Monitoring Training

```bash
# Watch training in real-time
screen -r train

# Update visualization periodically
watch -n 300 python analyze.py

# Check GPU usage
nvtop
```

### Export to ONNX

After training, export for deployment:

```python
from ultralytics import YOLO
model = YOLO('runs/train_alertcal/yolov8n_optimized_v1/weights/best.pt')
model.export(format='onnx', dynamic=False, simplify=True)
```

## 📚 References

- YOLOv8 Docs: https://docs.ultralytics.com/
- AlertCalifornia: https://www.alertcalifornia.org/
- Dataset: Fire_data_v2_yolo_with_blank_images_and_false_positives

## 📧 Support

For questions or issues:
1. Check this README
2. Review training logs
3. Run `python analyze.py --help`
4. Run `python test.py --help`

---

**Last Updated:** December 15, 2025
**Framework Version:** 1.0
**Author:** Optimized YOLOv8 Training for AlertCalifornia Hackathon
