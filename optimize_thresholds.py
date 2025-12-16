#!/usr/bin/env python3
"""
Threshold Optimization for Fire/Smoke Detection

Finds optimal confidence and IoU thresholds to maximize mAP or F1-score.
Use this after training to tune inference thresholds for deployment.

Usage:
    python optimize_thresholds.py --model runs/yolov8s_fire_smoke/weights/best.pt
    python optimize_thresholds.py --model best.pt --metric f1  # Optimize for F1
    python optimize_thresholds.py --model best.pt --metric recall  # Prioritize recall
"""

from ultralytics import YOLO
import argparse
import numpy as np
from pathlib import Path
import json


def optimize_confidence_threshold(model_path, data_yaml='dataset.yaml', metric='map50', iou=0.2):
    """
    Find optimal confidence threshold by grid search
    
    Args:
        model_path: Path to model weights
        data_yaml: Dataset configuration
        metric: Metric to optimize ('map50', 'map', 'f1', 'precision', 'recall')
        iou: IoU threshold for NMS (fixed)
    
    Returns:
        dict: Results with optimal threshold
    """
    print("="*80)
    print("🎯 CONFIDENCE THRESHOLD OPTIMIZATION")
    print("="*80)
    print(f"\nModel: {Path(model_path).name}")
    print(f"Optimizing for: {metric.upper()}")
    print(f"IoU threshold: {iou}")
    
    model = YOLO(model_path)
    
    # Grid search confidence thresholds
    conf_thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
    
    results_list = []
    best_value = 0 if metric != 'precision' else 0  # For recall, we want max
    best_conf = None
    
    print(f"\n{'Conf':>6} {'mAP@0.5':>10} {'mAP':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*80)
    
    for conf in conf_thresholds:
        # Run validation with this threshold
        results = model.val(
            data=data_yaml,
            conf=conf,
            iou=iou,
            verbose=False,
            plots=False
        )
        
        # Extract metrics
        map50 = float(results.box.map50)
        map_val = float(results.box.map)
        precision = float(results.box.mp)
        recall = float(results.box.mr)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store results
        results_list.append({
            'conf': conf,
            'map50': map50,
            'map': map_val,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Print row
        print(f"{conf:>6.3f} {map50:>10.4f} {map_val:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
        
        # Track best
        metric_value = results_list[-1][metric]
        if metric_value > best_value:
            best_value = metric_value
            best_conf = conf
    
    print("\n" + "="*80)
    print("📊 OPTIMAL THRESHOLD")
    print("="*80)
    print(f"Best confidence threshold: {best_conf}")
    print(f"Best {metric.upper()}: {best_value:.4f}")
    
    # Find the result for best_conf
    best_result = [r for r in results_list if r['conf'] == best_conf][0]
    print(f"\nMetrics at optimal threshold (conf={best_conf}):")
    print(f"  mAP@0.5:   {best_result['map50']:.4f}")
    print(f"  mAP@0.5:0.95: {best_result['map']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  F1-Score:  {best_result['f1']:.4f}")
    
    return {
        'optimal_conf': best_conf,
        'optimal_metric_value': best_value,
        'all_results': results_list,
        'best_result': best_result
    }


def optimize_iou_threshold(model_path, data_yaml='dataset.yaml', conf=0.25):
    """
    Find optimal IoU threshold for NMS
    
    Args:
        model_path: Path to model weights
        data_yaml: Dataset configuration
        conf: Confidence threshold (fixed)
    
    Returns:
        dict: Results with optimal IoU threshold
    """
    print("\n" + "="*80)
    print("🎯 IoU THRESHOLD OPTIMIZATION")
    print("="*80)
    print(f"\nModel: {Path(model_path).name}")
    print(f"Confidence threshold: {conf}")
    
    model = YOLO(model_path)
    
    # Grid search IoU thresholds
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    results_list = []
    best_map = 0
    best_iou = None
    
    print(f"\n{'IoU':>6} {'mAP@0.5':>10} {'Precision':>10} {'Recall':>10}")
    print("-"*60)
    
    for iou in iou_thresholds:
        results = model.val(
            data=data_yaml,
            conf=conf,
            iou=iou,
            verbose=False,
            plots=False
        )
        
        map50 = float(results.box.map50)
        precision = float(results.box.mp)
        recall = float(results.box.mr)
        
        results_list.append({
            'iou': iou,
            'map50': map50,
            'precision': precision,
            'recall': recall
        })
        
        print(f"{iou:>6.2f} {map50:>10.4f} {precision:>10.4f} {recall:>10.4f}")
        
        if map50 > best_map:
            best_map = map50
            best_iou = iou
    
    print("\n" + "="*80)
    print("📊 OPTIMAL IoU THRESHOLD")
    print("="*80)
    print(f"Best IoU threshold: {best_iou}")
    print(f"Best mAP@0.5: {best_map:.4f}")
    
    return {
        'optimal_iou': best_iou,
        'optimal_map': best_map,
        'all_results': results_list
    }


def main():
    parser = argparse.ArgumentParser(
        description='Optimize detection thresholds for fire/smoke detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize confidence threshold for mAP@0.5
  python optimize_thresholds.py --model runs/yolov8s_fire_smoke/weights/best.pt
  
  # Optimize for F1-score (balance precision/recall)
  python optimize_thresholds.py --model best.pt --metric f1
  
  # Optimize for recall (catch all fires, accept more false positives)
  python optimize_thresholds.py --model best.pt --metric recall
  
  # Optimize for precision (reduce false alarms, might miss some fires)
  python optimize_thresholds.py --model best.pt --metric precision
  
  # Optimize both conf and IoU
  python optimize_thresholds.py --model best.pt --optimize-iou
        """
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model weights (.pt file)')
    parser.add_argument('--data', type=str, default='dataset.yaml',
                        help='Path to dataset YAML (default: dataset.yaml)')
    parser.add_argument('--metric', type=str, default='map50',
                        choices=['map50', 'map', 'f1', 'precision', 'recall'],
                        help='Metric to optimize (default: map50)')
    parser.add_argument('--optimize-iou', action='store_true',
                        help='Also optimize IoU threshold (slower)')
    parser.add_argument('--output', type=str, default='threshold_optimization.json',
                        help='Output file for results (default: threshold_optimization.json)')
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    # Optimize confidence threshold
    conf_results = optimize_confidence_threshold(
        args.model,
        args.data,
        args.metric,
        iou=0.2  # Default IoU for initial optimization
    )
    
    # Optionally optimize IoU threshold
    iou_results = None
    if args.optimize_iou:
        optimal_conf = conf_results['optimal_conf']
        iou_results = optimize_iou_threshold(
            args.model,
            args.data,
            conf=optimal_conf
        )
    
    # Combine results
    final_results = {
        'model': str(Path(args.model).name),
        'optimization_metric': args.metric,
        'confidence_optimization': conf_results,
        'iou_optimization': iou_results
    }
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Print recommendation
    print("\n" + "="*80)
    print("🎯 DEPLOYMENT RECOMMENDATION")
    print("="*80)
    print(f"\nUse these thresholds in production:")
    print(f"  conf = {conf_results['optimal_conf']}")
    if iou_results:
        print(f"  iou  = {iou_results['optimal_iou']}")
    else:
        print(f"  iou  = 0.2  (default, run --optimize-iou to tune)")
    
    print(f"\nExpected performance:")
    best = conf_results['best_result']
    print(f"  mAP@0.5:   {best['map50']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall:    {best['recall']:.4f}")
    print(f"  F1-Score:  {best['f1']:.4f}")
    
    print("\n✅ Threshold optimization completed!")


if __name__ == '__main__':
    main()
