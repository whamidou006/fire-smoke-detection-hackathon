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

# Model configurations: name -> (path, batch_size)
MODEL_CONFIGS = {
    'n': ('yolov8n.pt', 128),
    's': ('yolov8s.pt', 96),
    'm': ('yolov8m.pt', 56),
    'l': ('yolov8l.pt', 40),
    'x': ('yolov8x.pt', 24),
    'pretrain': ('/home/whamidouche/ssdprivate/datasets/fire_hackathon/fire_hackathon/weights/pretrain_yolov8.pt', 40),
}

def main():
    """Main training function with all configurations"""
    
    # ============================================================================
    # ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 on Fire/Smoke Detection Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                    # Train with YOLOv8n (default)
  python train.py --model s          # Train with YOLOv8s
  python train.py --model l          # Train with YOLOv8l
  python train.py --model pretrain   # Continue from pretrain_yolov8.pt
  python train.py --model n --batch 64  # Custom batch size
  python train.py --model /path/to/custom.pt --batch 32  # Custom model

Available models:
  n        YOLOv8n (6MB, 3.2M params, batch=128)
  s        YOLOv8s (22MB, 11.2M params, batch=96)
  m        YOLOv8m (52MB, 25.9M params, batch=56)
  l        YOLOv8l (87MB, 43.7M params, batch=40)
  x        YOLOv8x (136MB, 68.2M params, batch=24)
  pretrain Hackathon pretrained YOLOv8l (batch=40)
        """
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='n',
        help='Model to train: n/s/m/l/x/pretrain or path to .pt file (default: n)'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=None,
        help='Batch size (default: auto-selected based on model)'
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
        MODEL_NAME = f'yolov8{args.model}' if args.model != 'pretrain' else 'yolov8l_pretrain'
    else:
        # Custom model path
        MODEL_PATH = args.model
        BATCH_SIZE = args.batch if args.batch is not None else 64  # Default for custom
        MODEL_NAME = Path(MODEL_PATH).stem
    
    print(f"\n📦 Model: {MODEL_NAME}")
    print(f"📊 Batch Size: {BATCH_SIZE}")
    
    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    try:
        model = YOLO(MODEL_PATH)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        print(f"\nIf model not found, download with:")
        print(f"wget https://github.com/ultralytics/assets/releases/download/v8.3.0/{MODEL_PATH}")
        sys.exit(1)
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    config = {
        # Dataset
        'data': 'dataset.yaml',
        
        # Training duration (extended for better convergence)
        'epochs': 200,
        'patience': 75,
        
        # Image and batch
        'imgsz': 640,
        'batch': BATCH_SIZE,
        'device': 0,
        'workers': 0,  # Avoid shared memory issues
        
        # Project settings
        'project': 'runs',
        'name': f'{MODEL_NAME}_fire_smoke',
        'exist_ok': True,
        'save_period': 10,
        
        # Optimizer (AdamW with improved learning rate)
        'optimizer': 'AdamW',
        'lr0': 0.002,        # Increased for faster convergence
        'lrf': 0.001,        # Lower final LR for fine-tuning
        'weight_decay': 0.001,  # Stronger regularization
        'warmup_epochs': 10,    # Longer warmup
        'warmup_momentum': 0.8,
        
        # Loss weights (recall-focused - prioritize detection over localization)
        'cls': 0.5,   # Classification loss weight (increased from 0.3)
        'box': 5.0,   # Box regression loss weight (reduced from 7.5)
        'dfl': 1.5,   # Distribution focal loss weight
        
        # Data augmentation (enhanced for fire/smoke robustness)
        'hsv_h': 0.015,      # Hue variation (fire color variations)
        'hsv_s': 0.6,        # Saturation variation (smoke density, lighting)
        'hsv_v': 0.5,        # Value/brightness (day/night, shadows)
        'degrees': 5.0,      # Small rotation (smoke can tilt slightly)
        'translate': 0.1,    # Translation (more position variation)
        'scale': 0.5,        # Scaling (fire appears at various scales)
        'shear': 0.0,        # Shear (disabled - shape matters)
        'perspective': 0.001,# Perspective (camera angles)
        'flipud': 0.0,       # Vertical flip (disabled - smoke rises)
        'fliplr': 0.5,       # Horizontal flip (enabled)
        'mosaic': 1.0,       # Mosaic augmentation
        'mixup': 0.15,       # Mixup (helps with hard negatives)
        'copy_paste': 0.1,   # Copy-paste (paste fire into empty regions)
        'erasing': 0.4,      # Random erasing (occlusion robustness)
        
        # Validation
        'val': True,
        'plots': True,
        'close_mosaic': 20,  # Disable mosaic last 20 epochs
        
        # Performance
        'amp': True,         # Automatic Mixed Precision
        'fraction': 1.0,     # Use full dataset
    }
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Dataset: AlertCalifornia Fire/Smoke (15,323 train, 382 val)")
    print(f"  • Epochs: {config['epochs']} (patience: {config['patience']})")
    print(f"  • Optimizer: {config['optimizer']} (lr={config['lr0']})")
    print(f"  • Augmentation: Conservative (no rotation, vertical flip)")
    print(f"  • Loss weights: cls={config['cls']}, box={config['box']}")
    
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
