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

# Multi-GPU training
python train.py --model 11x --batch 64 --device 0,1,2,3

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

**Available Configs:** `baseline`, `balanced`, `recall_aggressive`, `recall_moderate`, `reduce_fp`, `high_lr`

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

| Split | Images | Positive | Negative | Annotations |
|-------|--------|----------|----------|-------------|
| Train | 18,946 | 7,911 (41.8%) | 11,035 (58.2%) | 7,911 |
| Test  | 1,017  | 555 (54.6%) | 462 (45.4%) | 555 |

**Class Distribution:**
- Smoke: 94% of annotations (7,939 total)
- Fire: 6% of annotations (527 total)

**Hard Examples Integrated:**
- False Negatives: 2,192 images (missed fires, now annotated)
- False Positives: 4,256 images (false alarms as hard negatives)

**Design Rationale:**
- More negatives in training (58.2%) simulate real-world deployment
- Hard example mining addresses specific failure modes

---

## References

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [AlertCalifornia Project](https://www.alertcalifornia.org/)
- Dataset: Fire_data_v3_with_hard_examples

---

**Version:** 2.0  
**Last Updated:** December 19, 2025  
**Evaluation:** conf=0.01, iou=0.2 (standard thresholds)
