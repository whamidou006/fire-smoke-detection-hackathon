#!/usr/bin/env python3
"""
Fire/Smoke Detection Training - AlertCalifornia Dataset
Optimized for A100 80GB GPU with flexible model selection
"""
from ultralytics import YOLO
import torch
import sys
import argparse
from pathlib import Path
import subprocess
import os

# Model configurations: name -> (path, batch_size)
# Optimized batch sizes for A100 80GB: 128 for all models, 96 for yolov8x
MODEL_CONFIGS = {
    'n': ('yolov8n.pt', 128),
    's': ('yolov8s.pt', 128),
    'm': ('yolov8m.pt', 128),
    'l': ('yolov8l.pt', 128),
    'x': ('yolov8x.pt', 96),
    
    # YOLOv11 models (latest)
    '11n': ('yolo11n.pt', 128),
    '11s': ('yolo11s.pt', 128),
    '11m': ('yolo11m.pt', 128),
    '11l': ('yolo11l.pt', 128),
    '11x': ('yolo11x.pt', 96),
    
    # Pretrained model
    'pretrain': ('/home/whamidouche/ssdprivate/datasets/fire_hackathon/fire_hackathon/weights/pretrain_yolov8.pt', 128),
}

# ============================================================================
# HYPERPARAMETER CONFIGURATIONS
# ============================================================================
# Different experimental configurations to optimize recall vs precision trade-off

HYPERPARAMETER_CONFIGS = {
    'baseline': {
        'description': 'Current baseline configuration',
        'cls': 0.3,
        'box': 7.5,
        'dfl': 1.5,
        'lr0': 0.001,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'iou': 0.5,       # IoU threshold for NMS
        'conf': None,     # Confidence threshold (None = default)
    },
    
    'recall_moderate': {
        'description': 'Moderate recall boost with FP reduction',
        'cls': 0.25,      # ↓ Lower classification penalty → more detections
        'box': 8.0,       # ↑ Better localization → fewer false boxes
        'dfl': 1.6,       # ↑ Better box quality
        'lr0': 0.001,
        'mixup': 0.1,     # Add mixup for hard negatives
        'copy_paste': 0.0,
        'iou': 0.45,      # ↓ Lower IoU (keep more detections)
        'conf': None,
    },
    
    'recall_aggressive': {
        'description': 'Aggressive recall boost (catch more fires)',
        'cls': 0.2,       # ↓↓ Much lower classification penalty
        'box': 8.5,       # ↑↑ Stricter box quality
        'dfl': 1.8,       # ↑ Higher DFL
        'lr0': 0.0015,    # Slightly higher LR
        'mixup': 0.15,    # More mixup
        'copy_paste': 0.1,
        'iou': 0.4,       # ↓↓ Much lower IoU
        'conf': None,
    },
    
    'reduce_fp': {
        'description': 'Focus on reducing false positives',
        'cls': 0.4,       # ↑ Higher classification penalty
        'box': 7.0,       # Standard box weight
        'dfl': 1.5,
        'lr0': 0.001,
        'mixup': 0.2,     # More mixup for FP reduction
        'copy_paste': 0.0,
        'iou': 0.55,      # ↑ Higher IoU (stricter NMS)
        'conf': None,
    },
    
    'balanced': {
        'description': 'Balanced recall and precision (recommended)',
        'cls': 0.27,      # Slightly lower than baseline
        'box': 7.8,       # Slightly higher than baseline
        'dfl': 1.6,
        'lr0': 0.0012,    # Slightly higher LR
        'mixup': 0.1,
        'copy_paste': 0.05,
        'iou': 0.48,      # Slightly lower IoU
        'conf': None,
    },
    
    'high_lr': {
        'description': 'Higher learning rate for faster convergence',
        'cls': 0.25,
        'box': 8.0,
        'dfl': 1.6,
        'lr0': 0.002,     # ↑↑ Double LR
        'mixup': 0.1,
        'copy_paste': 0.0,
        'iou': 0.45,
        'conf': None,
    },
    
    'conservative_aug': {
        'description': 'Very conservative augmentation with minimal changes',
        'cls': 0.3,
        'box': 7.5,
        'dfl': 1.5,
        'lr0': 0.001,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'iou': 0.5,
        'conf': None,
        # Augmentation overrides (applied separately)
        'hsv_h': 0.005,   # Very minimal hue
        'hsv_s': 0.2,     # Reduced saturation
        'hsv_v': 0.2,     # Reduced brightness
        'scale': 0.1,     # Minimal scale
    },
}

def download_model(model_name):
    """
    Download YOLOv8 model using Python urllib if not already present.
    This handles SSL issues that occur with the default Ultralytics downloader.
    
    Args:
        model_name: Name of the model file (e.g., 'yolov8s.pt')
    
    Returns:
        Path to the downloaded model or None if download fails
    """
    # If it's a custom path (contains /), don't try to download
    if '/' in model_name:
        if not os.path.exists(model_name):
            print(f"❌ Custom model not found: {model_name}")
            return None
        return model_name
    
    # Check if model already exists locally (and not empty)
    script_dir = Path(__file__).parent
    local_path = script_dir / model_name
    
    if local_path.exists() and local_path.stat().st_size > 1024:  # At least 1KB
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model found locally: {local_path} ({file_size_mb:.1f}MB)")
        return str(local_path)
    
    # Check Ultralytics cache
    cache_dir = Path.home() / '.config' / 'Ultralytics'
    cached_path = cache_dir / model_name
    if cached_path.exists() and cached_path.stat().st_size > 1024:
        file_size_mb = cached_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model found in cache: {cached_path} ({file_size_mb:.1f}MB)")
        return str(cached_path)
    
    # Remove empty/corrupted file if exists
    if local_path.exists():
        print(f"⚠️  Removing corrupted file: {local_path}")
        local_path.unlink()
    
    # Download using Python urllib (more reliable with SSL)
    print(f"📥 Downloading {model_name}...")
    
    # Determine version for download URL
    if 'yolo11' in model_name or model_name.startswith('yolo11'):
        version = 'v8.3.0'  # YOLOv11 uses same release
    else:
        version = 'v8.3.0'  # YOLOv8
    
    url = f"https://github.com/ultralytics/assets/releases/download/{version}/{model_name}"
    
    try:
        import ssl
        import urllib.request
        
        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Download with progress
        print(f"Downloading from: {url}")
        with urllib.request.urlopen(url, context=ssl_context, timeout=300) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)", end='')
        
        print()  # New line after progress
        
        if local_path.exists() and local_path.stat().st_size > 1024:
            file_size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"✓ Successfully downloaded {model_name} ({file_size_mb:.1f}MB)")
            return str(local_path)
        else:
            print(f"❌ Downloaded file is invalid")
            if local_path.exists():
                local_path.unlink()
            return None
            
    except Exception as e:
        print(f"❌ Download error: {e}")
        if local_path.exists():
            local_path.unlink()
        return None


def main():
    """Main training function with all configurations"""
    
    # ============================================================================
    # ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description='Train YOLOv8/v11 on Fire/Smoke Detection Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                    # Train with YOLOv8n (default)
  python train.py --model s          # Train with YOLOv8s
  python train.py --model l          # Train with YOLOv8l
  python train.py --model 11s        # Train with YOLOv11s
  python train.py --model pretrain   # Continue from pretrain_yolov8.pt
  python train.py --model n --batch 64  # Custom batch size
  python train.py --model /path/to/custom.pt --batch 32  # Custom model
  
  # Hyperparameter configurations:
  python train.py --model s --config baseline            # Default config
  python train.py --model s --config recall_moderate     # Boost recall moderately
  python train.py --model s --config recall_aggressive   # Aggressive recall (catch more fires)
  python train.py --model s --config reduce_fp           # Reduce false positives
  python train.py --model s --config balanced            # Balanced approach (recommended)
  python train.py --model s --config high_lr             # Higher learning rate
  python train.py --model s --config conservative_aug    # Very conservative augmentation
  python train.py --model pretrain --config balanced --batch 128  # Full optimization

Available YOLOv8 models:
  n        YOLOv8n (6MB, 3.2M params, batch=128)
  s        YOLOv8s (22MB, 11.2M params, batch=128)
  m        YOLOv8m (52MB, 25.9M params, batch=128)
  l        YOLOv8l (87MB, 43.7M params, batch=128)
  x        YOLOv8x (136MB, 68.2M params, batch=96)

Available YOLOv11 models (latest):
  11n      YOLOv11n (5MB, 2.6M params, batch=128)
  11s      YOLOv11s (20MB, 9.4M params, batch=128)
  11m      YOLOv11m (48MB, 20.1M params, batch=128)
  11l      YOLOv11l (83MB, 25.3M params, batch=128)
  11x      YOLOv11x (131MB, 56.9M params, batch=96)

Other:
  pretrain Hackathon pretrained YOLOv8l (batch=128)
        """
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='n',
        help='Model to train: n/s/m/l/x/11n/11s/11m/11l/11x/pretrain or path to .pt file (default: n)'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=None,
        help='Batch size (default: auto-selected based on model)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='baseline',
        choices=list(HYPERPARAMETER_CONFIGS.keys()),
        help='Hyperparameter configuration: baseline/recall_moderate/recall_aggressive/reduce_fp/balanced/high_lr (default: baseline)'
    )
    
    args = parser.parse_args()
    
    # ============================================================================
    # GPU CHECK
    # ============================================================================
    print("\n" + "="*80)
    print("🔥 FIRE/SMOKE DETECTION TRAINING")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        sys.exit(1)
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ============================================================================
    # MODEL SELECTION
    # ============================================================================
    # Check if it's a predefined model or custom path
    if args.model in MODEL_CONFIGS:
        MODEL_PATH, default_batch = MODEL_CONFIGS[args.model]
        BATCH_SIZE = args.batch if args.batch is not None else default_batch
        
        # Handle model naming
        if args.model == 'pretrain':
            MODEL_NAME = 'yolov8l_pretrain'
        elif args.model.startswith('11'):
            MODEL_NAME = f'yolo{args.model}'  # e.g., '11s' -> 'yolo11s'
        else:
            MODEL_NAME = f'yolov8{args.model}'  # e.g., 'n' -> 'yolov8n'
    else:
        # Custom model path
        MODEL_PATH = args.model
        BATCH_SIZE = args.batch if args.batch is not None else 128  # Default batch size
        MODEL_NAME = Path(MODEL_PATH).stem
    
    print(f"\n📦 Model: {MODEL_NAME}")
    print(f"📊 Batch Size: {BATCH_SIZE}")
    
    # ============================================================================
    # HYPERPARAMETER CONFIGURATION SELECTION
    # ============================================================================
    hp_config = HYPERPARAMETER_CONFIGS[args.config]
    print(f"⚙️  Hyperparameter Config: {args.config}")
    print(f"   {hp_config['description']}")
    print(f"   Loss weights: cls={hp_config['cls']}, box={hp_config['box']}, dfl={hp_config['dfl']}")
    
    # ============================================================================
    # DOWNLOAD MODEL IF NEEDED
    # ============================================================================
    # Ensure model is downloaded before loading
    downloaded_path = download_model(MODEL_PATH)
    if downloaded_path is None:
        print(f"\n❌ Failed to download/locate model: {MODEL_PATH}")
        print(f"\nAvailable YOLOv8 models: n, s, m, l, x")
        print(f"Available YOLOv11 models: 11n, 11s, 11m, 11l, 11x")
        print(f"Other: pretrain")
        print(f"Or provide a custom path to an existing .pt file")
        sys.exit(1)
    
    # Use the downloaded/verified path
    MODEL_PATH = downloaded_path
    
    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    try:
        model = YOLO(MODEL_PATH)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        sys.exit(1)
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    
    # Start with base augmentation values
    aug_config = {
        'hsv_h': 0.01,       # Minimal hue shift (preserve fire colors)
        'hsv_s': 0.4,        # Moderate saturation (smoke density variations)
        'hsv_v': 0.3,        # Moderate brightness (day/night)
        'scale': 0.2,        # Moderate scale variation
    }
    
    # Override augmentation if conservative_aug config is selected
    if args.config == 'conservative_aug':
        aug_config.update({
            'hsv_h': hp_config.get('hsv_h', 0.01),
            'hsv_s': hp_config.get('hsv_s', 0.4),
            'hsv_v': hp_config.get('hsv_v', 0.3),
            'scale': hp_config.get('scale', 0.2),
        })
    
    config = {
        # Dataset
        'data': 'dataset.yaml',
        
        # Training duration (extended for better convergence)
        'epochs': 200,
        'patience': 50,
        
        # Image and batch
        'imgsz': 640,
        'batch': BATCH_SIZE,
        'device': 0,
        'workers': 0,  # Avoid shared memory issues
        
        # Multi-scale training (improves robustness across scales)
        'rect': False,   # Disable rectangular training for multi-scale
        
        # Project settings
        'project': 'runs',
        'name': f'{MODEL_NAME}_{args.config}',  # Include config name in run name
        'exist_ok': True,
        'save_period': 10,
        
        # Optimizer (AdamW with improved learning rate)
        'optimizer': 'AdamW',
        'lr0': hp_config['lr0'],        # From hyperparameter config
        'lrf': 0.01,         # Final LR factor (more gradual decay for better convergence)
        'weight_decay': 0.0005,  # Regularization (balanced)
        'warmup_epochs': 5,    # Longer warmup
        'warmup_momentum': 0.8,
        
        # Loss weights (from selected hyperparameter configuration)
        'cls': hp_config['cls'],   # Classification loss weight
        'box': hp_config['box'],   # Box regression loss weight
        'dfl': hp_config['dfl'],   # Distribution focal loss weight
        
        # NMS and confidence thresholds
        'iou': hp_config['iou'],   # IoU threshold for NMS
        # 'conf' is not a training parameter in Ultralytics, only for inference
        
        # Data augmentation (CONSERVATIVE for fire/smoke - based on train_alertcal_optimized.py)
        # Fire/smoke have distinctive characteristics that aggressive augmentation can distort:
        # - Fire colors (orange/red) shouldn't shift much
        # - Smoke rises upward (rotation breaks this pattern)
        # - Smoke density depends on saturation/brightness
        'hsv_h': aug_config['hsv_h'],       # Hue shift
        'hsv_s': aug_config['hsv_s'],       # Saturation variation
        'hsv_v': aug_config['hsv_v'],       # Brightness variation
        'degrees': 0.0,      # NO rotation (smoke orientation matters!)
        'translate': 0.05,   # Minimal translation
        'scale': aug_config['scale'],        # Scale variation
        'shear': 0.0,        # No shear (shape matters)
        'perspective': 0.0,  # No perspective (keep natural)
        'flipud': 0.0,       # No vertical flip (smoke rises)
        'fliplr': 0.5,       # Horizontal flip only
        'mosaic': 1.0,       # Keep mosaic (helps with context)
        'mixup': hp_config['mixup'],        # From hyperparameter config
        'copy_paste': hp_config['copy_paste'],   # From hyperparameter config
        'erasing': 0.0,      # No random erasing (fire/smoke shouldn't disappear)
        
        # Validation
        'val': True,
        'plots': True,
        'close_mosaic': 20,  # Disable mosaic last 20 epochs
        
        # Performance
        'amp': True,         # Automatic Mixed Precision
        'fraction': 1.0,     # Use full dataset
    }
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  • Dataset: AlertCalifornia Fire/Smoke (15,323 train, 382 val)")
    print(f"  • Epochs: {config['epochs']} (patience: {config['patience']})")
    print(f"  • Optimizer: {config['optimizer']} (lr={config['lr0']})")
    print(f"  • Config: {args.config} - {hp_config['description']}")
    print(f"  • Loss weights: cls={config['cls']}, box={config['box']}, dfl={config['dfl']}")
    print(f"  • NMS IoU threshold: {config['iou']}")
    print(f"  • Augmentation: mixup={config['mixup']}, copy_paste={config['copy_paste']}")
    print(f"  • HSV aug: h={config['hsv_h']}, s={config['hsv_s']}, v={config['hsv_v']}")
    print(f"  • Conservative augmentation (no rotation, no vertical flip)")
    
    # ============================================================================
    # START TRAINING
    # ============================================================================
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80 + "\n")
    
    try:
        results = model.train(**config)
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED!")
        print("="*80)
        
        # ========================================================================
        # FINAL VALIDATION
        # ========================================================================
        print("\n🔍 Running final validation...")
        metrics = model.val(data='dataset.yaml', workers=0)
        
        print("\n📊 FINAL METRICS:")
        print(f"  • mAP@0.5:     {metrics.box.map50:.4f}")
        print(f"  • mAP@0.5-0.95: {metrics.box.map:.4f}")
        print(f"  • Precision:    {metrics.box.mp:.4f}")
        print(f"  • Recall:       {metrics.box.mr:.4f}")
        
        # ========================================================================
        # EXPORT TO ONNX
        # ========================================================================
        print("\n📦 Exporting to ONNX...")
        try:
            onnx_path = model.export(format='onnx', dynamic=False, simplify=True)
            print(f"✅ ONNX export successful: {onnx_path}")
        except Exception as e:
            print(f"⚠️  ONNX export failed: {e}")
        
        # ========================================================================
        # SAVE PATHS
        # ========================================================================
        weights_dir = Path(config['project']) / config['name'] / 'weights'
        print(f"\n📁 MODEL OUTPUTS:")
        print(f"  • Best weights: {weights_dir / 'best.pt'}")
        print(f"  • Last weights: {weights_dir / 'last.pt'}")
        print(f"  • Results CSV:  {Path(config['project']) / config['name'] / 'results.csv'}")
        
        print("\n" + "="*80)
        print("🎉 ALL DONE!")
        print("="*80)
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
