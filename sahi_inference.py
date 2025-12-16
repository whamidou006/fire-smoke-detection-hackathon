"""
SAHI (Slicing Aided Hyper Inference) for Fire/Smoke Detection - OPTIONAL

This script uses the SAHI library for production-grade tiling inference.
SAHI provides optimized slicing strategies and multiple merge algorithms (NMS, WBF, NMW).

⚠️  OPTIONAL DEPENDENCY - May cause conflicts with existing environment
    Install only if you need advanced merge strategies for 4K+ images

Installation (use at your own risk):
    pip install sahi
    
    If you encounter dependency conflicts, use a separate conda environment:
    conda create -n sahi python=3.10
    conda activate sahi
    pip install sahi ultralytics

Recommended for:
    - High-resolution images (4K+: 3840×2160 and above)
    - Aerial/satellite imagery
    - Small object detection at distance
    - Production deployments requiring robust tiling

NOT recommended for:
    - Standard resolution (≤1920×1080) - use test.py instead
    - Real-time inference requirements
    
Usage:
    # Single image
    python sahi_inference.py --model best.pt --image image.jpg
    
    # Directory of images
    python sahi_inference.py --model best.pt --image /path/to/images/ --output results/
    
    # Custom slicing parameters
    python sahi_inference.py --model best.pt --image image.jpg \
        --slice-size 1280 --overlap 0.3 --postprocess WBF
    
    # With ground truth evaluation
    python sahi_inference.py --model best.pt --image test_images/ \
        --labels test_labels/ --output results/
"""

import argparse
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional

# Try to import SAHI
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.postprocess.combine import (
        NMSPostprocess,
        NMWPostprocess,
        GreedyNMMPostprocess
    )
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    print("⚠️  SAHI not installed. Install with: pip install sahi")


def load_ground_truth_yolo(label_path: str, image_shape: tuple) -> List[Dict]:
    """
    Load ground truth from YOLO format txt file
    
    Args:
        label_path: Path to label file
        image_shape: (height, width) of image
    
    Returns:
        List of ground truth boxes
    """
    if not Path(label_path).exists():
        return []
    
    h, w = image_shape[:2]
    ground_truths = []
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # Convert from YOLO format (normalized) to absolute
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            ground_truths.append({
                'bbox': [x1, y1, x2, y2],
                'class': class_id
            })
    
    return ground_truths


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(predictions: List[Dict], ground_truths: List[Dict], 
                     iou_threshold: float = 0.5) -> Dict:
    """
    Calculate precision, recall, F1, and mAP@0.5
    
    Args:
        predictions: List of predictions with 'bbox', 'class', 'conf'
        ground_truths: List of ground truths with 'bbox', 'class'
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dict with metrics
    """
    if len(ground_truths) == 0:
        if len(predictions) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'map50': 1.0, 
                   'tp': 0, 'fp': 0, 'fn': 0}
        else:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'map50': 0.0,
                   'tp': 0, 'fp': len(predictions), 'fn': 0}
    
    if len(predictions) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'map50': 0.0,
               'tp': 0, 'fp': 0, 'fn': len(ground_truths)}
    
    # Match predictions to ground truths
    matched_gt = set()
    tp = 0
    fp = 0
    
    # Sort predictions by confidence
    sorted_preds = sorted(predictions, key=lambda x: x['conf'], reverse=True)
    
    for pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            
            if pred['class'] != gt['class']:
                continue
            
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(ground_truths) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map50': recall,  # Simplified
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def process_image_sahi(
    image_path: str,
    detection_model,
    slice_height: int,
    slice_width: int,
    overlap_ratio: float,
    postprocess_type: str,
    postprocess_threshold: float,
    output_dir: Optional[str] = None,
    labels_dir: Optional[str] = None,
    visualize: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Process image using SAHI
    
    Args:
        image_path: Path to image
        detection_model: SAHI detection model
        slice_height: Height of each slice
        slice_width: Width of each slice
        overlap_ratio: Overlap ratio (0.0-0.5)
        postprocess_type: 'NMS', 'NMW', or 'GREEDYNMM'
        postprocess_threshold: IoU threshold for postprocessing
        output_dir: Directory to save results
        labels_dir: Directory with ground truth labels
        visualize: Whether to save visualization
        verbose: Whether to print details
    
    Returns:
        Result dictionary
    """
    image_path = Path(image_path)
    
    # Run SAHI prediction
    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type=postprocess_type,
        postprocess_match_threshold=postprocess_threshold,
        postprocess_class_agnostic=False,
        verbose=0
    )
    
    # Convert predictions
    predictions = []
    class_counts = {}
    
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        class_name = pred.category.name
        class_id = pred.category.id
        confidence = pred.score.value
        
        predictions.append({
            'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
            'class': class_id,
            'class_name': class_name,
            'conf': confidence
        })
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Read image for dimensions
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    
    # Prepare result
    result_dict = {
        'image': image_path.name,
        'image_size': {'width': w, 'height': h},
        'num_detections': len(predictions),
        'class_counts': class_counts,
        'detections': predictions
    }
    
    # Calculate metrics if ground truth available
    if labels_dir:
        label_path = Path(labels_dir) / f"{image_path.stem}.txt"
        ground_truths = load_ground_truth_yolo(str(label_path), (h, w))
        
        if ground_truths or len(predictions) > 0:
            metrics = calculate_metrics(predictions, ground_truths)
            result_dict['metrics'] = metrics
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / f"{image_path.stem}_detections.json"
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        if verbose:
            print(f"💾 Saved: {json_path}")
        
        # Save visualization
        if visualize:
            vis_path = output_dir / f"{image_path.stem}_sahi.jpg"
            result.export_visuals(export_dir=str(output_dir), file_name=image_path.stem)
            if verbose:
                print(f"🖼️  Saved: {vis_path}")
    
    return result_dict


def main():
    parser = argparse.ArgumentParser(
        description='SAHI-based Tiling Inference for Fire/Smoke Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with defaults
  python sahi_inference.py --model best.pt --image image.jpg
  
  # Directory with custom slicing
  python sahi_inference.py --model best.pt --image images/ \\
      --slice-size 1280 --overlap 0.3 --postprocess WBF
  
  # With ground truth evaluation
  python sahi_inference.py --model best.pt --image test/images/ \\
      --labels test/labels/ --output results/
  
  # For 4K images (recommended)
  python sahi_inference.py --model best.pt --image 4k_image.jpg \\
      --slice-size 1920 --overlap 0.4 --postprocess NMW

Note: This is optimized for high-resolution images (4K+).
      For standard resolution (1920×1080), use test.py instead.
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO model weights')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image or directory')
    parser.add_argument('--output', type=str, default='sahi_results',
                       help='Output directory (default: sahi_results)')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to ground truth labels directory (optional)')
    parser.add_argument('--slice-size', type=int, default=640,
                       help='Size of each slice (default: 640)')
    parser.add_argument('--overlap', type=float, default=0.3,
                       help='Overlap ratio 0.0-0.5 (default: 0.3)')
    parser.add_argument('--conf', type=float, default=0.15,
                       help='Confidence threshold (default: 0.15)')
    parser.add_argument('--iou', type=float, default=0.4,
                       help='IoU threshold for postprocessing (default: 0.4)')
    parser.add_argument('--postprocess', type=str, default='NMS',
                       choices=['NMS', 'NMW', 'GREEDYNMM'],
                       help='Postprocessing method (default: NMS)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device (default: cuda:0)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    # Check SAHI availability
    if not SAHI_AVAILABLE:
        print("❌ SAHI library not found!")
        print("\n📦 Install with:")
        print("   pip install sahi")
        print("\n📚 Learn more: https://github.com/obss/sahi")
        return
    
    print("=" * 80)
    print("🔥 SAHI Inference for Fire/Smoke Detection")
    print("=" * 80)
    print(f"Model:        {Path(args.model).name}")
    print(f"Slice size:   {args.slice_size}×{args.slice_size}")
    print(f"Overlap:      {args.overlap*100:.0f}%")
    print(f"Postprocess:  {args.postprocess}")
    print(f"Confidence:   {args.conf}")
    print(f"IoU:          {args.iou}")
    print(f"Device:       {args.device}")
    print("=" * 80)
    
    # Initialize SAHI detection model
    print("\n⏳ Loading model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device
    )
    print("✅ Model loaded successfully")
    
    # Process image(s)
    image_path = Path(args.image)
    
    if image_path.is_file():
        # Single image
        print(f"\n📷 Processing: {image_path.name}")
        result = process_image_sahi(
            str(image_path),
            detection_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_ratio=args.overlap,
            postprocess_type=args.postprocess,
            postprocess_threshold=args.iou,
            output_dir=args.output,
            labels_dir=args.labels,
            visualize=not args.no_visualize,
            verbose=True
        )
        
        # Print summary
        print(f"\n{'='*80}")
        print("📊 RESULTS")
        print(f"{'='*80}")
        print(f"Total detections: {result['num_detections']}")
        for class_name, count in result['class_counts'].items():
            print(f"  {class_name}: {count}")
        
        if 'metrics' in result:
            m = result['metrics']
            print(f"\n📈 METRICS")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1 Score:  {m['f1']:.4f}")
            print(f"  mAP@0.5:   {m['map50']:.4f}")
    
    elif image_path.is_dir():
        # Multiple images
        image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
        print(f"\n📂 Processing {len(image_files)} images from {image_path}")
        
        all_results = []
        all_metrics = []
        has_ground_truth = args.labels is not None
        
        # Process with progress bar
        for img_file in tqdm(image_files, desc="Processing", unit="img"):
            result = process_image_sahi(
                str(img_file),
                detection_model,
                slice_height=args.slice_size,
                slice_width=args.slice_size,
                overlap_ratio=args.overlap,
                postprocess_type=args.postprocess,
                postprocess_threshold=args.iou,
                output_dir=args.output,
                labels_dir=args.labels,
                visualize=not args.no_visualize,
                verbose=False
            )
            all_results.append(result)
            
            if 'metrics' in result:
                all_metrics.append(result['metrics'])
        
        # Save summary
        summary_path = Path(args.output) / 'summary.json'
        summary_data = {
            'config': {
                'model': args.model,
                'slice_size': args.slice_size,
                'overlap': args.overlap,
                'postprocess': args.postprocess,
                'conf_threshold': args.conf,
                'iou_threshold': args.iou
            },
            'results': all_results,
            'has_metrics': has_ground_truth
        }
        
        if has_ground_truth and all_metrics:
            # Calculate overall metrics
            total_tp = sum(m['tp'] for m in all_metrics)
            total_fp = sum(m['fp'] for m in all_metrics)
            total_fn = sum(m['fn'] for m in all_metrics)
            
            overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
            
            summary_data['overall_metrics'] = {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'map50': overall_recall,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n💾 Saved summary: {summary_path}")
        
        # Print statistics
        total_detections = sum(r['num_detections'] for r in all_results)
        print(f"\n{'='*80}")
        print("📊 OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"Images processed:  {len(all_results)}")
        print(f"Total detections:  {total_detections}")
        print(f"Average per image: {total_detections/len(all_results):.1f}")
        
        if has_ground_truth and all_metrics:
            print(f"\n{'='*80}")
            print("📈 EVALUATION METRICS")
            print(f"{'='*80}")
            m = summary_data['overall_metrics']
            print(f"Precision:   {m['precision']:.4f}")
            print(f"Recall:      {m['recall']:.4f}")
            print(f"F1 Score:    {m['f1']:.4f}")
            print(f"mAP@0.5:     {m['map50']:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"  True Positives:  {m['tp']}")
            print(f"  False Positives: {m['fp']}")
            print(f"  False Negatives: {m['fn']}")
    
    else:
        print(f"❌ Invalid path: {args.image}")
        return
    
    print(f"\n{'='*80}")
    print("✅ Processing complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
