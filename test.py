"""
Fire/Smoke Detection Model Testing and Evaluation
Tests trained models and compares against baselines

Note: For tiling inference on high-resolution images, use tiling_inference.py
"""
from ultralytics import YOLO
import argparse
import os
from pathlib import Path


def test_model(model_path, data_yaml, batch=8, conf=0.01, iou=0.2, save_json=False, multi_scale=False):
    """
    Test a single model and return detailed metrics
    
    Args:
        model_path: Path to model weights (.pt file)
        data_yaml: Path to dataset YAML configuration
        batch: Batch size for testing
        conf: Confidence threshold
        iou: IoU threshold for NMS
        save_json: Whether to save results in COCO JSON format
        multi_scale: Enable multi-scale validation (Test-Time Augmentation)
    
    Returns:
        dict: Evaluation metrics
    """
    print("="*80)
    print(f"🔍 TESTING MODEL: {Path(model_path).name}")
    print("="*80)
    
    # Load model
    model = YOLO(model_path)
    
    # Get model info
    total_params = sum(p.numel() for p in model.model.parameters())
    file_size = os.path.getsize(model_path) / (1024**2)
    
    print(f"\n📦 Model Information:")
    print(f"  File:       {Path(model_path).name}")
    print(f"  Size:       {file_size:.1f} MB")
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Classes:    {len(model.names)} ({', '.join(model.names.values())})")
    
    # Run validation
    print(f"\n🔄 Running validation...")
    print(f"  Dataset:     {data_yaml}")
    print(f"  Batch size:  {batch}")
    print(f"  Confidence:  {conf}")
    print(f"  IoU:         {iou}")
    if multi_scale:
        print(f"  Multi-scale: ✅ ENABLED (Test-Time Augmentation)")
    
    results = model.val(
        data=data_yaml,
        batch=batch,
        imgsz=1024,
        conf=conf,
        iou=iou,
        workers=0,
        save_json=save_json,
        plots=True,
        verbose=True,
        augment=multi_scale  # Enable Test-Time Augmentation
    )
    
    # Extract metrics
    precision = float(results.box.mp)
    recall = float(results.box.mr)
    
    # Calculate F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    metrics = {
        'model_name': Path(model_path).name,
        'model_path': str(model_path),
        'file_size_mb': file_size,
        'parameters': total_params,
        'map50': float(results.box.map50),
        'map50_95': float(results.box.map),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'fitness': float(results.fitness),
    }
    
    # Per-class metrics if available
    if hasattr(results.box, 'maps'):
        for i, (class_name, class_map) in enumerate(zip(model.names.values(), results.box.maps)):
            metrics[f'{class_name}_map50'] = float(class_map)
    
    print(f"\n📊 Results Summary:")
    print(f"  mAP@0.5:      {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    print(f"  Fitness:      {metrics['fitness']:.4f}")
    
    return metrics


def compare_models(models_dict, data_yaml, batch=8, conf=0.01, iou=0.2, multi_scale=False):
    """
    Compare multiple models side-by-side
    
    Args:
        models_dict: Dictionary of {name: model_path}
        data_yaml: Path to dataset YAML
        batch: Batch size
        conf: Confidence threshold
        iou: IoU threshold
    
    Returns:
        dict: Results for all models
    """
    print("\n" + "="*80)
    print("🏆 MODEL COMPARISON")
    print("="*80)
    
    all_results = {}
    
    for name, model_path in models_dict.items():
        if not os.path.exists(model_path):
            print(f"\n⚠️  Model not found: {name} ({model_path})")
            continue
        
        try:
            results = test_model(model_path, data_yaml, batch, conf, iou, 
                               save_json=False, multi_scale=multi_scale)
            all_results[name] = results
        except Exception as e:
            print(f"\n❌ Error testing {name}: {e}")
            continue
    
    # Print comparison table
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("📊 COMPARISON TABLE")
        print("="*80)
        print(f"\n{'Model':<25} {'Size':>8} {'Params':>10} {'mAP@0.5':>10} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("-"*80)
        
        # Sort by mAP@0.5
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['map50'], reverse=True)
        
        for name, metrics in sorted_results:
            print(f"{name:<25} {metrics['file_size_mb']:>7.1f}M "
                  f"{metrics['parameters']/1e6:>9.1f}M "
                  f"{metrics['map50']:>10.4f} "
                  f"{metrics['precision']:>8.4f} "
                  f"{metrics['recall']:>8.4f} "
                  f"{metrics['f1_score']:>8.4f}")
        
        # Show best model
        best_name, best_metrics = sorted_results[0]
        print("\n" + "="*80)
        print(f"🏆 BEST MODEL: {best_name}")
        print(f"   mAP@0.5: {best_metrics['map50']:.4f}")
        print("="*80)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Test Fire/Smoke Detection Models')
    parser.add_argument('--model', type=str, help='Path to model weights (.pt file)')
    parser.add_argument('--data', type=str, default='dataset.yaml',
                        help='Path to dataset YAML (default: dataset.yaml)')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--conf', type=float, default=0.01,
                        help='Confidence threshold (default: 0.01, optimized for recall)')
    parser.add_argument('--iou', type=float, default=0.2,
                        help='IoU threshold for NMS (default: 0.2, optimized for recall)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with baseline models')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results in COCO JSON format')
    parser.add_argument('--multi-scale', action='store_true',
                        help='Enable multi-scale validation (Test-Time Augmentation)')
    
    args = parser.parse_args()
    
    # Default dataset path
    if not os.path.exists(args.data):
        args.data = os.path.join(os.path.dirname(__file__), 'dataset.yaml')
    
    if args.compare:
        # Compare mode: test your model + baselines
        print("\n🔄 COMPARISON MODE: Testing multiple models...")
        
        models_to_compare = {}
        
        # Your trained model
        if args.model and os.path.exists(args.model):
            models_to_compare['Your Model'] = args.model
        else:
            # Try to find latest trained model
            possible_paths = [
                'runs/train/weights/best.pt',
                'runs/train/weights/last.pt',
                '../runs/train_alertcal/yolov8n_optimized_v1/weights/best.pt',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    models_to_compare['Your Model'] = path
                    break
        
        # Baseline models
        baseline_dir = '/home/whamidouche/ssdprivate/datasets/fire_hackathon/fire_hackathon/weights'
        baseline_models = {
            'Current Best': os.path.join(baseline_dir, 'current_best.pt'),
            'Pretrain YOLOv8': os.path.join(baseline_dir, 'pretrain_yolov8.pt'),
        }
        
        for name, path in baseline_models.items():
            if os.path.exists(path):
                models_to_compare[name] = path
        
        if not models_to_compare:
            print("❌ No models found to compare!")
            return
        
        # Run comparison
        results = compare_models(models_to_compare, args.data, args.batch, 
                               args.conf, args.iou, args.multi_scale)
        
        # Save results
        import json
        output_file = 'test_results_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
    else:
        # Single model test mode
        if not args.model:
            print("❌ Please provide a model path with --model")
            print("\nExamples:")
            print("  python test.py --model runs/train/weights/best.pt")
            print("  python test.py --model runs/train/weights/best.pt --compare")
            return
        
        if not os.path.exists(args.model):
            print(f"❌ Model not found: {args.model}")
            return
        
        # Test single model
        results = test_model(args.model, args.data, args.batch, args.conf, args.iou, 
                            args.save_json, args.multi_scale)
        
        # Save results
        import json
        output_file = f"test_results_{Path(args.model).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    print("\n✅ Testing completed!")


if __name__ == '__main__':
    main()
