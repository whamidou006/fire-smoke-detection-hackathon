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

## Automated Hyperparameter Tuning with GPT-5

The framework includes an automated hyperparameter optimization system powered by GPT-5 that iteratively trains models and refines configurations to maximize F1 score.

### Quick Start

```bash
# Test run: 5 epochs × 50 iterations (~30-35 hours)
python auto_tune_training.py --iterations 50 --epochs 5

# Production run: 150 epochs × 5 iterations (~10 days)
python auto_tune_training.py --config auto_tune_config.yaml
```

### How It Works

1. **Initial Training**: Starts with a baseline configuration (e.g., "balanced")
2. **GPT-5 Analysis**: Analyzes training metrics, identifies bottlenecks
3. **Optimization**: Recommends improved hyperparameters based on trends
4. **Iteration**: Trains with new config, repeats until convergence
5. **Best Selection**: Tracks best F1 score across all iterations

### Tunable Parameters (11 total)

**Loss Weights:**
- `cls` (0.1-2.0): Classification loss - higher boosts recall
- `box` (5.0-10.0): Bounding box regression loss
- `dfl` (1.0-2.0): Distribution focal loss
- `fl_gamma` (0.0-2.0): Focal loss gamma - combats class imbalance

**Learning Rates:**
- `lr0` (1e-6 to 1e-3): Initial learning rate
- `lrf` (1e-6 to 1e-4): Final learning rate factor

**Augmentation:**
- `mixup` (0.0-0.3): Mixup augmentation strength
- `copy_paste` (0.0-0.2): Copy-paste augmentation
- `mosaic` (0.0-1.0): Mosaic augmentation
- `scale` (0.1-0.5): Scale augmentation range

**Training Settings:**
- `imgsz` (640, 800, 1024, 1280): Input resolution

### Adaptive Batch Sizing

Batch size automatically adapts based on image resolution to prevent GPU OOM:
- `imgsz ≤ 640`: batch = 128
- `imgsz ≤ 1024`: batch = 64
- `imgsz > 1024`: batch = 32

### Command Options

```bash
# Full command with all parameters
python auto_tune_training.py \
  --config auto_tune_config.yaml \
  --model 11x \
  --initial-config balanced \
  --iterations 50 \
  --epochs 5 \
  --imgsz 640 \
  --device 0,1

# Run in background with logging
nohup python auto_tune_training.py --iterations 50 --epochs 5 \
  > auto_tune.log 2>&1 &

# Monitor progress
tail -f auto_tune.log

# Run in screen session
screen -S autotune
python auto_tune_training.py --iterations 50 --epochs 5
# Detach: Ctrl+A then D, Reattach: screen -r autotune
```

### Output Files

```
auto_tune_logs/
└── tuning_history.json  # Complete history with all configs and metrics
```

Extract best configuration:
```bash
python extract_best_config.py auto_tune_logs/tuning_history.json
```

### Time Estimates

**Test Run (5 epochs × 50 iterations):**
- Per iteration: ~30-40 minutes
- Total: ~30-35 hours (~1.5 days)

**Production Run (150 epochs × 5 iterations):**
- Per iteration: ~2 days
- Total: ~10 days

### Evaluation Thresholds

All evaluations use consistent thresholds optimized for fire detection:
- **conf=0.01**: Confidence threshold (filters weak predictions)
- **iou=0.2**: NMS IoU threshold (low value keeps more boxes for better recall)

These settings are critical for safety-critical fire detection where missing a fire is worse than a false alarm.

### Configuration Files

- `auto_tune_config.yaml`: GPT-5 settings, hyperparameter ranges, prompt template
- `train_configs.yaml`: Predefined training configurations (balanced, recall_focused, etc.)

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
