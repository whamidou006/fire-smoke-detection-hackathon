# Fire/Smoke Detection Training Framework

Production-ready training framework for fire/smoke detection using YOLOv8/YOLOv11 models on the Fire_data_v3_with_hard_examples dataset.

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/whamidou006/fire-smoke-detection-hackathon.git
cd fire-smoke-detection-hackathon/fire_smoke_training

# 2. Install dependencies
conda create -n firesmoke python=3.11 -y
conda activate firesmoke
pip install -r requirements.txt

# 3. Configure dataset path in dataset.yaml
# path: /path/to/Fire_data_v3_with_hard_examples_cleaned

# 4. Start training
python train.py --model 11l --config balanced
```

---

## Reproducing Results

### Balanced Configuration (640px, Recommended)
```bash
# Single GPU training
python train.py --model 11l --config balanced --batch 128 --imgsz 640

# Multi-GPU training (2x GPUs)
CUDA_VISIBLE_DEVICES=0,1 python train.py --model 11l --config balanced --batch 128 --device 0,1 --imgsz 640

# Evaluation
python test.py --model runs/yolo11l_balanced/weights/best.pt --imgsz 640
```

### High-Resolution Training (1024px)
```bash
# Single GPU training (requires more memory)
python train.py --model 11l --config balanced --batch 64 --imgsz 1024

# Multi-GPU training (2x GPUs, recommended)
CUDA_VISIBLE_DEVICES=0,1 python train.py --model 11l --config balanced --batch 64 --device 0,1 --imgsz 1024

# Evaluation
python test.py --model runs/yolo11l_balanced/weights/best.pt --imgsz 1024
```

**Note:** Higher resolution (1024px) captures more detail but requires more GPU memory. Reduce batch size accordingly.

---

## Project Structure

```
fire_smoke_training/
├── train.py              # Training script with hyperparameter configs
├── test.py               # Model evaluation and comparison
├── analyze.py            # Training visualization and analysis
├── dataset.yaml          # Dataset configuration
├── merge_hard_examples.py # Dataset preparation script
└── README.md             # This file
```

---

## Usage

### 1. Training

```bash
# Basic training
python train.py --model 11l --config balanced

# Multi-GPU training (2x GPUs)
python train.py --model 11x --batch 64 --device 0,1

# Custom configuration
python train.py --model 11l --batch 128 --config recall_aggressive
```

**Model Selection:**
- `11l` - YOLOv11l (recommended, 25.4M params)
- `11x` - YOLOv11x (best accuracy, 57.0M params)
- `l` - YOLOv8l (baseline comparison)
- `pretrain` - Pretrained YOLOv8l

**Hyperparameter Configs:**
- `balanced` ⭐ - Optimized recall/precision (recommended)
- `recall_focused` 🔥 - Maximize recall with focal loss (cls=1.0, fl_gamma=0.5)
- `recall_aggressive` - Maximize fire detection
- `reduce_fp` - Minimize false alarms
- `baseline` - Standard configuration

### 2. Evaluation

```bash
# Evaluate model (conf=0.01, iou=0.2)
python test.py --model runs/yolo11l_balanced/weights/best.pt

# Compare with baselines
python test.py --model runs/yolo11l_balanced/weights/best.pt --compare

# Test-Time Augmentation
python test.py --model best.pt --multi-scale
```

### 3. Analysis

```bash
# Visualize training progress
python analyze.py --results runs/yolo11l_balanced/results.csv

# Test and analyze
python analyze.py --test runs/yolo11l_balanced/weights/best.pt --compare
```

---

## Dataset Configuration

**Dataset:** Fire_data_v3_with_hard_examples_cleaned

```yaml
# dataset.yaml
path: /path/to/Fire_data_v3_with_hard_examples_cleaned
train: train/images  # 18,946 images
val: test/images     # 1,017 images

names:
  0: smoke  # 94% of annotations
  1: fire   # 6% of annotations
```

### Training Configuration

**Default Settings (balanced config):**
- Epochs: 150 (patience=0, no early stopping)
- Optimizer: AdamW (lr=1e-5)
- Loss weights: cls=0.27, box=7.8, dfl=1.6
- Augmentation: Conservative (no rotation/vertical flip)
- Checkpoints: Saved every 2 epochs

**Loss Balancing for Precision/Recall:**

If your model converges with high precision but low recall:
- Use `recall_focused` config: Doubles cls loss (1.0) and adds focal loss (fl_gamma=0.5)
- Focal loss focuses on hard-to-classify examples (missed detections)
- Higher cls weight penalizes false negatives more than false positives

```bash
# Example: Train with recall-focused configuration
python train.py --model 11x --config recall_focused --batch 64 --imgsz 640
```

**Available Configs:** `baseline`, `balanced`, `recall_focused`, `recall_aggressive`, `recall_moderate`, `reduce_fp`, `high_lr`

---

## Outputs

### Training
```
runs/yolo11l_balanced/
├── weights/
│   ├── best.pt          # Best checkpoint
│   ├── last.pt          # Latest checkpoint
│   └── best.onnx        # ONNX export (if exported)
├── results.csv          # Epoch metrics
└── results.png          # Training curves
```

### Analysis
- `training_analysis.png` - 6-panel visualization
- `test_results_*.json` - Detailed metrics

---

## Dataset Details

### Fire_data_v3_with_hard_examples

**Total:** 19,963 images (18,946 train + 1,017 test)

**Composition by Source:**

| Split | Source Dataset | Images | Smoke Annot. | Fire Annot. | Empty/Neg. | Cumulative |
|-------|---------------|--------|--------------|-------------|------------|------------|
| Train | Original v2 | 13,133 | 5,161 | 481 | 8,054 | 13,133 |
| | FN (missed fires) | 1,963 | 2,750 | 11 | 0 | 15,096 |
| | FP (hard negatives) | 3,850 | 0 | 0 | 3,850 | **18,946** |
| Test | Original v2 | 382 | 292 | 35 | 90 | 382 |
| | FN (missed fires) | 229 | 263 | 0 | 0 | 611 |
| | FP (hard negatives) | 406 | 0 | 0 | 406 | **1,017** |

**Class Distribution:**
- Smoke: 8,466 annotations (94.3%)
- Fire: 527 annotations (5.7%)

**Hard Examples Integrated:**
- False Negatives: 2,192 images (1,963 train + 229 test) - Missed detections now annotated
- False Positives: 4,256 images (3,850 train + 406 test) - False alarms as hard negatives

**Design Rationale:**
- More negatives in training (62.0% negative) simulate real-world deployment
- Hard example mining addresses specific failure modes from previous model iterations

---

## References

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [AlertCalifornia Project](https://www.alertcalifornia.org/)
- Dataset: Fire_data_v3_with_hard_examples

---

**Version:** 2.0  
**Last Updated:** December 20, 2025  
**Evaluation:** conf=0.01, iou=0.2 (standard thresholds)
