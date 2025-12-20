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
        'lrf': 1e-6,
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
        'lrf': 1e-6,
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
        'lrf': 1e-6,
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
        'lrf': 1e-6,
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
        'lr0': 1.0e-05,    # Slightly higher LR
        'lrf': 1e-4,
        'mixup': 0.1,
        'copy_paste': 0.05,
        'iou': 0.2,      # Slightly lower IoU
        'conf': 0.01,
        'scale': 0.2,     # Enable multi-scale augmentation
    },
    
    'high_lr': {
        'description': 'Higher learning rate for faster convergence',
        'cls': 0.25,
        'box': 8.0,
        'dfl': 1.6,
        'lr0': 0.002,     # ↑↑ Double LR
        'lrf': 1e-6,
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
        'lrf': 1e-6,
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
  python train.py --model 11l --config balanced              # Recommended
  python train.py --model 11x --batch 64 --device 0,1       # Multi-GPU
  python train.py --model 11l --config recall_aggressive    # Custom config

Models: n, s, m, l, x (YOLOv8) | 11n, 11s, 11m, 11l, 11x (YOLOv11) | pretrain
Configs: baseline, balanced, recall_aggressive, recall_moderate, reduce_fp, high_lr

Use --help for full options
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
        '--imgsz', '-i',
        type=int,
        default=640,
        help='Image size for training (default: 640)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='baseline',
        choices=list(HYPERPARAMETER_CONFIGS.keys()),
        help='Hyperparameter configuration: baseline/recall_moderate/recall_aggressive/reduce_fp/balanced/high_lr (default: baseline)'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='0',
        help='CUDA device(s): 0 or 0,1,2,3 for multi-GPU (default: 0)'
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
    
    # Adjust batch size for multi-GPU (warn if potentially too large)
    num_devices = len(args.device.split(','))
    if num_devices > 1:
        batch_per_gpu = BATCH_SIZE / num_devices
        print(f"📊 Batch Size: {BATCH_SIZE} total ({batch_per_gpu:.0f} per GPU × {num_devices} GPUs)")
        if batch_per_gpu > 48 and args.model in ['l', 'x', '11l', '11x']:
            print(f"⚠️  WARNING: {batch_per_gpu:.0f} samples per GPU might cause OOM for large models!")
            print(f"   Consider reducing batch size (e.g., --batch {int(BATCH_SIZE * 0.5)} or --batch {int(BATCH_SIZE * 0.75)})")
    else:
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
        'scale': 0.2,       # Use scale from config if available
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
        'epochs': 150,
        'patience': 0,  # Disable early stopping (0 = train all epochs)
        
        # Image and batch
        'imgsz': args.imgsz,
        'batch': BATCH_SIZE,
        'device': args.device,  # Support single or multi-GPU
        'workers': 0,  # Avoid shared memory issues
        
        # Multi-scale training (improves robustness across scales)
        'rect': False,   # Disable rectangular training for multi-scale
        
        # Project settings
        'project': 'runs',
        'name': f'{MODEL_NAME}_{args.config}',  # Include config name in run name
        'exist_ok': True,
        'save_period': 2,  # Save checkpoint every 2 epochs
        
        # Optimizer (AdamW with improved learning rate)
        'optimizer': 'AdamW',
        'lr0': hp_config['lr0'],        # From hyperparameter config
        'lrf': hp_config['lrf'],        # From hyperparameter config
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
    print(f"  • Dataset: Fire_data_v3_with_hard_examples (18,946 train, 1,017 test)")
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
