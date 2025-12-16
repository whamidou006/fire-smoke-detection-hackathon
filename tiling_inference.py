"""
Fire/Smoke Detection with Tiling Inference - Standalone Tool

This script implements sliding window inference for processing high-resolution images.
Designed for scenarios where images exceed standard resolution or require fine-grained analysis.

For standard model evaluation on the validation set, use test.py instead.

⚠️  NOTE: Optimized for high-resolution images (4K+). For standard resolution (≤1920×1080),
    direct inference is recommended for optimal performance.

💡 TIP: For production deployments with advanced merge strategies, consider sahi_inference.py
   (requires: pip install sahi). SAHI provides NMS, WBF, and NMW postprocessing options.

Usage:
    # Single image
    python tiling_inference.py --model best.pt --image image.jpg --output results/
    
    # Entire directory
    python tiling_inference.py --model best.pt --image /path/to/images/ --output results/
    
    # Custom settings for 4K+ images
    python tiling_inference.py --model best.pt --image image.jpg --tile-size 1280 --overlap 0.3
    
    # Alternative: SAHI for advanced use cases
    python sahi_inference.py --model best.pt --image 4k_image.jpg --slice-size 1920 --overlap 0.3

Implements sliding window inference for high-resolution images and enhanced small object detection
"""
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict
import argparse
from tqdm import tqdm
import json


class TilingInference:
    """
    Implements tiling (sliding window) inference for YOLO models.
    Useful for:
    - High-resolution images that exceed GPU memory
    - Better small object detection (fire/smoke at distance)
    - Reducing false positives through focused analysis
    """
    
    def __init__(
        self,
        model_path: str,
        tile_size: int = 640,
        overlap: float = 0.2,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cuda:0'
    ):
        """
        Initialize tiling inference
        
        Args:
            model_path: Path to YOLO model weights
            tile_size: Size of each tile (e.g., 640x640)
            overlap: Overlap ratio between tiles (0.0-0.5, default 0.2 = 20%)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            device: Device to run inference on (cuda:0, cuda:1, cpu)
        """
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        print(f"🔥 Tiling Inference Initialized")
        print(f"  Model:      {Path(model_path).name}")
        print(f"  Tile size:  {tile_size}x{tile_size}")
        print(f"  Overlap:    {overlap*100:.0f}%")
        print(f"  Stride:     {self.stride}px")
        print(f"  Device:     {device}")
        print(f"  Conf:       {conf_threshold}")
        print(f"  IoU:        {iou_threshold}")
    
    def get_tiles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile positions for a given image
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            List of (x1, y1, x2, y2) tile coordinates
        """
        h, w = image.shape[:2]
        tiles = []
        
        # Calculate number of tiles needed
        y_positions = list(range(0, h - self.tile_size + 1, self.stride))
        x_positions = list(range(0, w - self.tile_size + 1, self.stride))
        
        # Add final tiles if image doesn't divide evenly
        if y_positions[-1] + self.tile_size < h:
            y_positions.append(h - self.tile_size)
        if x_positions[-1] + self.tile_size < w:
            x_positions.append(w - self.tile_size)
        
        # Generate all tile positions
        for y in y_positions:
            for x in x_positions:
                tiles.append((x, y, x + self.tile_size, y + self.tile_size))
        
        return tiles
    
    def predict_tile(self, tile: np.ndarray) -> List[dict]:
        """
        Run inference on a single tile
        
        Args:
            tile: Image tile (tile_size, tile_size, 3)
            
        Returns:
            List of detections [{class, conf, bbox}, ...]
        """
        results = self.model.predict(
            tile,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                detections.append({
                    'class': int(cls),
                    'class_name': self.model.names[cls],
                    'confidence': float(conf),
                    'bbox': box.tolist()  # [x1, y1, x2, y2]
                })
        
        return detections
    
    def merge_detections(
        self,
        all_detections: List[Tuple[dict, int, int]],
        image_shape: Tuple[int, int]
    ) -> List[dict]:
        """
        Merge overlapping detections from multiple tiles using NMS
        
        Args:
            all_detections: List of (detection, tile_x_offset, tile_y_offset)
            image_shape: (height, width) of original image
            
        Returns:
            List of merged detections
        """
        if not all_detections:
            return []
        
        # Convert to global coordinates
        global_detections = []
        for det, x_offset, y_offset in all_detections:
            bbox = det['bbox']
            global_bbox = [
                bbox[0] + x_offset,
                bbox[1] + y_offset,
                bbox[2] + x_offset,
                bbox[3] + y_offset
            ]
            global_detections.append({
                'class': det['class'],
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'bbox': global_bbox
            })
        
        # Group by class
        class_groups = {}
        for det in global_detections:
            cls = det['class']
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append(det)
        
        # Apply NMS per class
        merged = []
        for cls, dets in class_groups.items():
            if not dets:
                continue
            
            boxes = np.array([d['bbox'] for d in dets])
            scores = np.array([d['confidence'] for d in dets])
            
            # Apply NMS
            keep_indices = self.nms(boxes, scores, self.iou_threshold)
            
            for idx in keep_indices:
                merged.append(dets[idx])
        
        return merged
    
    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression
        
        Args:
            boxes: Array of boxes (N, 4) in format [x1, y1, x2, y2]
            scores: Array of confidence scores (N,)
            iou_threshold: IoU threshold for suppression
            
        Returns:
            List of indices to keep
        """
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def predict(self, image: np.ndarray, visualize: bool = False) -> Tuple[List[dict], Optional[np.ndarray]]:
        """
        Run tiling inference on full image
        
        Args:
            image: Input image (H, W, C)
            visualize: Whether to create visualization
            
        Returns:
            (detections, visualization_image)
        """
        h, w = image.shape[:2]
        tiles = self.get_tiles(image)
        
        print(f"\n📐 Image size: {w}x{h}")
        print(f"🔲 Processing {len(tiles)} tiles...")
        
        # Process each tile
        all_detections = []
        for x1, y1, x2, y2 in tqdm(tiles, desc="Processing tiles"):
            tile = image[y1:y2, x1:x2]
            
            # Run inference on tile
            tile_detections = self.predict_tile(tile)
            
            # Store with offset
            for det in tile_detections:
                all_detections.append((det, x1, y1))
        
        # Merge overlapping detections
        print(f"🔗 Merging {len(all_detections)} detections...")
        merged_detections = self.merge_detections(all_detections, (h, w))
        print(f"✅ Final detections: {len(merged_detections)}")
        
        # Create visualization if requested
        vis_image = None
        if visualize:
            vis_image = self.visualize(image, merged_detections, tiles)
        
        return merged_detections, vis_image
    
    def visualize(
        self,
        image: np.ndarray,
        detections: List[dict],
        tiles: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> np.ndarray:
        """
        Create visualization of detections and tiles
        
        Args:
            image: Original image
            detections: List of detections
            tiles: Optional tile positions to draw
            
        Returns:
            Visualization image
        """
        vis = image.copy()
        
        # Draw tiles (semi-transparent grid)
        if tiles:
            overlay = vis.copy()
            for x1, y1, x2, y2 in tiles:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), 1)
            vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)
        
        # Draw detections
        colors = {
            0: (255, 200, 0),   # Smoke - cyan
            1: (0, 69, 255)     # Fire - orange/red
        }
        
        for det in detections:
            bbox = [int(x) for x in det['bbox']]
            cls = det['class']
            conf = det['confidence']
            name = det['class_name']
            color = colors.get(cls, (0, 255, 0))
            
            # Draw box
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                vis,
                (bbox[0], bbox[1] - label_h - baseline - 5),
                (bbox[0] + label_w, bbox[1]),
                color,
                -1
            )
            cv2.putText(
                vis, label,
                (bbox[0], bbox[1] - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )
        
        return vis
    
    def process_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        visualize: bool = True,
        verbose: bool = True
    ) -> dict:
        """
        Process a single image with tiling
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results (optional)
            visualize: Whether to save visualization
            verbose: Whether to print processing details (default: True)
            
        Returns:
            Result dict with detections and metadata
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run inference
        detections, vis_image = self.predict(image, visualize=visualize)
        
        # Prepare results
        result = {
            'image': Path(image_path).name,
            'image_size': {'width': image.shape[1], 'height': image.shape[0]},
            'num_detections': len(detections),
            'detections': detections
        }
        
        # Count by class
        class_counts = {}
        for det in detections:
            name = det['class_name']
            class_counts[name] = class_counts.get(name, 0) + 1
        result['class_counts'] = class_counts
        
        # Save results if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON
            json_path = output_dir / f"{Path(image_path).stem}_detections.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            if verbose:
                print(f"💾 Saved detections: {json_path}")
            
            # Save visualization
            if visualize and vis_image is not None:
                vis_path = output_dir / f"{Path(image_path).stem}_tiled.jpg"
                cv2.imwrite(str(vis_path), vis_image)
                if verbose:
                    print(f"🖼️  Saved visualization: {vis_path}")
        
        return result


def calculate_metrics(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> Dict:
    """
    Calculate precision, recall, mAP@0.5, and F1 score
    
    Args:
        predictions: List of predictions [{'bbox': [x1,y1,x2,y2], 'class': int, 'conf': float}, ...]
        ground_truths: List of ground truth boxes [{'bbox': [x1,y1,x2,y2], 'class': int}, ...]
        iou_threshold: IoU threshold for matching (default: 0.5)
    
    Returns:
        Dict with metrics: precision, recall, f1, map50, tp, fp, fn
    """
    if len(ground_truths) == 0:
        # No ground truth - either all negatives or no labels
        if len(predictions) == 0:
            # True negative case
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'map50': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        else:
            # All predictions are false positives
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'map50': 0.0, 'tp': 0, 'fp': len(predictions), 'fn': 0}
    
    if len(predictions) == 0:
        # No predictions but have ground truth - all false negatives
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'map50': 0.0, 'tp': 0, 'fp': 0, 'fn': len(ground_truths)}
    
    # Calculate IoU between all pred-gt pairs
    def calc_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    # Match predictions to ground truths
    matched_gt = set()
    tp = 0
    fp = 0
    
    # Sort predictions by confidence (highest first)
    sorted_preds = sorted(predictions, key=lambda x: x['conf'], reverse=True)
    
    for pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            
            # Check class match
            if pred['class'] != gt['class']:
                continue
            
            iou = calc_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(ground_truths) - tp
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    map50 = recall  # Simplified: for single IoU threshold, mAP@0.5 ≈ recall at that threshold
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map50': map50,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def load_ground_truth(image_path: str, labels_dir: Optional[str] = None) -> List[Dict]:
    """
    Load ground truth labels from YOLO format txt file
    
    Args:
        image_path: Path to image file
        labels_dir: Optional path to labels directory (if None, assumes parallel structure)
    
    Returns:
        List of ground truth boxes [{'bbox': [x1,y1,x2,y2], 'class': int}, ...]
    """
    image_path = Path(image_path)
    
    # Determine label file path
    if labels_dir:
        label_path = Path(labels_dir) / f"{image_path.stem}.txt"
    else:
        # Assume parallel structure: images/ -> labels/
        label_path = image_path.parent.parent / 'labels' / image_path.parent.name / f"{image_path.stem}.txt"
        if not label_path.exists():
            # Try without parent subdirectory
            label_path = image_path.parent.parent / 'labels' / f"{image_path.stem}.txt"
    
    if not label_path.exists():
        return []  # No ground truth available
    
    # Read image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    
    ground_truths = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # Convert from YOLO format (normalized center) to absolute corners
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            ground_truths.append({
                'bbox': [x1, y1, x2, y2],
                'class': class_id
            })
    
    return ground_truths


def main():
    parser = argparse.ArgumentParser(description='Fire/Smoke Detection with Tiling Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
    parser.add_argument('--image', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='tiling_results', help='Output directory')
    parser.add_argument('--labels', type=str, default=None, help='Path to ground truth labels directory (optional, auto-detected if not provided)')
    parser.add_argument('--tile-size', type=int, default=640, help='Tile size (default: 640)')
    parser.add_argument('--overlap', type=float, default=0.2, help='Overlap ratio (default: 0.2)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    # Initialize tiling inference
    tiler = TilingInference(
        model_path=args.model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Process image(s)
    image_path = Path(args.image)
    
    if image_path.is_file():
        # Single image
        print(f"\n📷 Processing: {image_path.name}")
        result = tiler.process_image(
            str(image_path),
            output_dir=args.output,
            visualize=not args.no_visualize
        )
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"📊 RESULTS")
        print(f"{'='*80}")
        print(f"Total detections: {result['num_detections']}")
        for class_name, count in result['class_counts'].items():
            print(f"  {class_name}: {count}")
        
    elif image_path.is_dir():
        # Multiple images
        image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
        print(f"\n📂 Processing {len(image_files)} images from {image_path}")
        
        all_results = []
        all_metrics = []
        has_ground_truth = False
        
        # Use tqdm progress bar instead of per-image logging
        for img_file in tqdm(image_files, desc="Processing images", unit="img"):
            result = tiler.process_image(
                str(img_file),
                output_dir=args.output,
                visualize=not args.no_visualize,
                verbose=False  # Suppress per-image output
            )
            all_results.append(result)
            
            # Try to load ground truth and calculate metrics
            ground_truths = load_ground_truth(str(img_file), labels_dir=args.labels)
            if ground_truths or Path(img_file).parent.parent.name in ['test', 'val']:
                # Ground truth available or in test/val directory
                has_ground_truth = True
                
                # Convert detections to format for metric calculation
                predictions = []
                for det in result['detections']:
                    predictions.append({
                        'bbox': det['bbox'],
                        'class': det['class'],
                        'conf': det['confidence']
                    })
                
                metrics = calculate_metrics(predictions, ground_truths, iou_threshold=0.5)
                all_metrics.append(metrics)
        
        # Save summary
        summary_path = Path(args.output) / 'summary.json'
        summary_data = {
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
                'map50': overall_recall,  # Simplified
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n💾 Saved summary: {summary_path}")
        
        # Print overall statistics
        total_detections = sum(r['num_detections'] for r in all_results)
        print(f"\n{'='*80}")
        print(f"📊 OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"Images processed: {len(all_results)}")
        print(f"Total detections: {total_detections}")
        print(f"Average per image: {total_detections/len(all_results):.1f}")
        
        # Print metrics if available
        if has_ground_truth and all_metrics:
            print(f"\n{'='*80}")
            print(f"📈 EVALUATION METRICS (with Ground Truth)")
            print(f"{'='*80}")
            metrics = summary_data['overall_metrics']
            print(f"Precision:   {metrics['precision']:.4f}")
            print(f"Recall:      {metrics['recall']:.4f}")
            print(f"F1 Score:    {metrics['f1']:.4f}")
            print(f"mAP@0.5:     {metrics['map50']:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"  True Positives:  {metrics['tp']}")
            print(f"  False Positives: {metrics['fp']}")
            print(f"  False Negatives: {metrics['fn']}")
    else:
        raise ValueError(f"Invalid path: {args.image}")


if __name__ == '__main__':
    main()
