#!/usr/bin/env python3
"""
Training Analysis & Visualization
Generates comprehensive plots comparing training progress with baselines
Can also run model testing by calling test.py
Supports watch mode for continuous updates during training

Baseline Evaluation Thresholds:
- Default: conf=0.01, iou=0.2 (optimized for maximum recall)
- Use --conf and --iou flags for custom thresholds
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse
import subprocess
import time
from datetime import datetime
from ultralytics import YOLO


def load_baseline_metrics(dataset_yaml, conf_threshold=0.01, iou_threshold=0.2, imgsz=640):
    """Load and evaluate baseline models
    
    Args:
        dataset_yaml: Path to dataset YAML file
        conf_threshold: Confidence threshold (default: 0.01)
        iou_threshold: IoU threshold (default: 0.2)
        imgsz: Image size for validation (default: 640)
    """
    baselines = {}
    
    baseline_models = {
        'Current Best': '/home/whamidouche/ssdprivate/datasets/fire_hackathon/fire_hackathon/weights/current_best.pt',
        'Pretrain YOLOv8': '/home/whamidouche/ssdprivate/datasets/fire_hackathon/fire_hackathon/weights/pretrain_yolov8.pt'
    }
    
    print("📊 Evaluating baseline models...")
    print(f"   Using conf={conf_threshold}, iou={iou_threshold}")
    
    for name, model_path in baseline_models.items():
        if not Path(model_path).exists():
            print(f"  ⚠️  {name} not found at {model_path}")
            continue
        
        try:
            print(f"  🔄 {name}...", end=' ')
            model = YOLO(model_path)
            results = model.val(
                data=dataset_yaml,
                batch=8,
                imgsz=imgsz,
                verbose=False,
                conf=conf_threshold,  # Use provided threshold
                iou=iou_threshold,    # Use provided threshold
                workers=0
            )
            
            baselines[name] = {
                'map50': float(results.box.map50),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            }
            print(f"✓ mAP@0.5={baselines[name]['map50']:.4f}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return baselines


def create_visualization(results_csv, output_path, baselines, show_baselines=True):
    """Create comprehensive 6-panel training visualization"""
    
    # Load training results
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # Calculate F1 score for each epoch
    precision = df['metrics/precision(B)']
    recall = df['metrics/recall(B)']
    df['f1_score'] = 2 * (precision * recall) / (precision + recall)
    df['f1_score'] = df['f1_score'].fillna(0)  # Handle division by zero
    
    current_epoch = int(df['epoch'].iloc[-1])
    current_map = df['metrics/mAP50(B)'].iloc[-1]
    current_f1 = df['f1_score'].iloc[-1]
    
    # Find best based on F1 score (primary) and mAP (secondary)
    best_f1 = df['f1_score'].max()
    best_f1_epoch = int(df[df['f1_score'] == best_f1]['epoch'].values[0])
    best_f1_map = df[df['f1_score'] == best_f1]['metrics/mAP50(B)'].values[0]
    
    # Also track best mAP for comparison
    best_map = df['metrics/mAP50(B)'].max()
    best_map_epoch = int(df[df['metrics/mAP50(B)'] == best_map]['epoch'].values[0])
    
    print(f"\n📈 Training Progress:")
    print(f"  • Current: Epoch {current_epoch}, F1={current_f1:.4f}, mAP@0.5={current_map:.4f}")
    print(f"  • Best F1: Epoch {best_f1_epoch}, F1={best_f1:.4f}, mAP@0.5={best_f1_map:.4f}")
    if best_map_epoch != best_f1_epoch:
        print(f"  • Best mAP: Epoch {best_map_epoch}, mAP@0.5={best_map:.4f} (different from best F1)")
    
    # Create figure with 4 rows instead of 3
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Training Performance Analysis - Fire/Smoke Detection\nBest Model: Epoch {best_f1_epoch} (F1={best_f1:.4f}, mAP={best_f1_map:.4f})',
                 fontsize=16, fontweight='bold')
    
    # Colors for baselines
    colors = {'Current Best': 'red', 'Pretrain YOLOv8': 'orange'}
    
    # ========================================================================
    # Plot 1: mAP@0.5 Progress (Main plot - spans 2 columns)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Raw and smoothed data
    ax1.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', alpha=0.3, label='Raw', linewidth=1)
    
    # Smoothed data - handle NaN by dropping them
    window = 5
    rolling = df['metrics/mAP50(B)'].rolling(window=window, center=True).mean()
    # Only plot where rolling average is valid (no NaN)
    valid_mask = ~rolling.isna()
    ax1.plot(df['epoch'][valid_mask], rolling[valid_mask], 'b-', linewidth=2.5, label=f'{window}-epoch average')
    
    # Baseline lines
    if show_baselines and baselines:
        for name, metrics in baselines.items():
            ax1.axhline(y=metrics['map50'], color=colors[name], linestyle='--',
                       linewidth=2, alpha=0.7, label=f'{name} ({metrics["map50"]:.3f})')
    
    # Target and current marker
    ax1.axhline(y=0.60, color='g', linestyle=':', linewidth=2, alpha=0.5, label='Target (0.60)')
    ax1.axvline(x=best_f1_epoch, color='green', linestyle='--', linewidth=2.5, alpha=0.7,
               label=f'Best Model (ep {best_f1_epoch}, F1={best_f1:.3f})')
    ax1.axvline(x=current_epoch, color='purple', linestyle=':', alpha=0.5,
               label=f'Current (ep {current_epoch})')
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('mAP@0.5', fontsize=11)
    ax1.set_title('Validation mAP@0.5 Progress vs Baselines', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    max_y = max(0.7, df['metrics/mAP50(B)'].max() * 1.1)
    if baselines:
        max_y = max(max_y, max([m['map50'] for m in baselines.values()]) * 1.1)
    ax1.set_ylim(0, max_y)
    
    # ========================================================================
    # Plot 2: Precision & Recall
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    
    ax2.plot(df['epoch'], df['metrics/precision(B)'], 'g-', alpha=0.5, linewidth=1, label='Precision')
    ax2.plot(df['epoch'], df['metrics/recall(B)'], 'r-', alpha=0.5, linewidth=1, label='Recall')
    
    # Smoothed - handle NaN values
    prec_rolling = df['metrics/precision(B)'].rolling(window, center=True).mean()
    rec_rolling = df['metrics/recall(B)'].rolling(window, center=True).mean()
    
    prec_valid = ~prec_rolling.isna()
    rec_valid = ~rec_rolling.isna()
    
    ax2.plot(df['epoch'][prec_valid], prec_rolling[prec_valid], 'g-', linewidth=2)
    ax2.plot(df['epoch'][rec_valid], rec_rolling[rec_valid], 'r-', linewidth=2)
    
    # Baseline lines
    if show_baselines and baselines:
        for name, metrics in baselines.items():
            ax2.axhline(y=metrics['precision'], color=colors[name], linestyle='--', alpha=0.5, linewidth=1.5)
            ax2.axhline(y=metrics['recall'], color=colors[name], linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.set_title('Precision & Recall', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # ========================================================================
    # Plot 3: Loss Curves (spans all 3 columns)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, :])
    
    ax3.plot(df['epoch'], df['train/box_loss'], 'b-', alpha=0.7, linewidth=2, label='Box Loss')
    ax3.plot(df['epoch'], df['train/cls_loss'], 'r-', alpha=0.7, linewidth=2, label='Classification Loss')
    ax3.plot(df['epoch'], df['train/dfl_loss'], 'g-', alpha=0.7, linewidth=2, label='DFL Loss')
    ax3.axvline(x=current_epoch, color='orange', linestyle=':', alpha=0.5, label=f'Current (ep {current_epoch})')
    ax3.axvline(x=best_f1_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2, label=f'Best Model (ep {best_f1_epoch})')
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Training Loss Curves (Decreasing = Learning)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 3.5: Learning Rate Schedule (NEW - spans all 3 columns)
    # ========================================================================
    ax3_5 = fig.add_subplot(gs[2, :])
    
    # Check which learning rate columns are available
    lr_columns = [col for col in df.columns if col.startswith('lr/')]
    
    if lr_columns:
        # Plot all learning rate parameter groups
        colors_lr = ['blue', 'red', 'green']
        for idx, lr_col in enumerate(lr_columns):
            color = colors_lr[idx % len(colors_lr)]
            label = lr_col.replace('lr/', 'LR ').upper()
            ax3_5.plot(df['epoch'], df[lr_col], color=color, alpha=0.7, linewidth=2, label=label)
        
        # Mark best epoch
        ax3_5.axvline(x=best_f1_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2, 
                     label=f'Best Model (ep {best_f1_epoch})')
        ax3_5.axvline(x=current_epoch, color='orange', linestyle=':', alpha=0.5, 
                     label=f'Current (ep {current_epoch})')
        
        ax3_5.set_xlabel('Epoch', fontsize=11)
        ax3_5.set_ylabel('Learning Rate', fontsize=11)
        ax3_5.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax3_5.legend(fontsize=10, loc='best')
        ax3_5.grid(True, alpha=0.3)
        ax3_5.set_yscale('log')  # Log scale for better visualization
    else:
        # If no LR columns, show a message
        ax3_5.text(0.5, 0.5, 'Learning Rate data not available in results.csv', 
                  ha='center', va='center', fontsize=12, transform=ax3_5.transAxes)
        ax3_5.axis('off')
    
    # ========================================================================
    # Plot 4: Bar Chart Comparison (moved to row 4)
    # ========================================================================
    ax4 = fig.add_subplot(gs[3, 0])
    
    models = ['Your Model\n(Current)', 'Your Model\n(Best)']
    scores = [current_map, best_map]
    bar_colors = ['skyblue', 'cornflowerblue']
    
    if baselines:
        for name, metrics in baselines.items():
            models.append(name.replace(' ', '\n'))
            scores.append(metrics['map50'])
            bar_colors.append(colors[name])
    
    bars = ax4.bar(models, scores, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=0.60, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target')
    ax4.set_ylabel('mAP@0.5', fontsize=10)
    ax4.set_title('mAP@0.5 Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(0.65, max(scores) * 1.1))
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend(fontsize=8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ========================================================================
    # Plot 5: Metrics Table (moved to row 4)
    # ========================================================================
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis('off')
    
    text = "METRICS COMPARISON\n" + "="*50 + "\n\n"
    text += f"{'Model':<18} {'F1':<8} {'mAP':<8} {'Prec':<8} {'Recall':<8}\n"
    text += "-"*55 + "\n"
    
    # Your model - use F1-based best
    best_f1_idx = df['f1_score'].idxmax()
    current_prec = df['metrics/precision(B)'].iloc[-1]
    current_rec = df['metrics/recall(B)'].iloc[-1]
    best_prec = df['metrics/precision(B)'].iloc[best_f1_idx]
    best_rec = df['metrics/recall(B)'].iloc[best_f1_idx]
    
    text += f"{'Your (Now)':<18} {current_f1:<8.4f} {current_map:<8.4f} {current_prec:<8.4f} {current_rec:<8.4f}\n"
    text += f"{'Your (Best F1)':<18} {best_f1:<8.4f} {best_f1_map:<8.4f} {best_prec:<8.4f} {best_rec:<8.4f}\n"
    
    # Baselines
    if baselines:
        for name, metrics in baselines.items():
            short_name = name.split()[0]
            # Calculate F1 for baselines
            baseline_f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
            text += f"{short_name:<18} {baseline_f1:<8.4f} {metrics['map50']:<8.4f} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f}\n"
    
    text += "\n" + "-"*55 + "\n"
    text += "TARGET             0.6600   0.6000   0.6000   0.7000\n"
    text += "(Balanced performance across all metrics)\n"
    
    ax5.text(0.05, 0.95, text, transform=ax5.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # ========================================================================
    # Plot 6: Summary Statistics (moved to row 4)
    # ========================================================================
    ax6 = fig.add_subplot(gs[3, 2])
    ax6.axis('off')
    
    # Calculate improvements based on F1 score
    first_epoch_prec = df['metrics/precision(B)'].iloc[0]
    first_epoch_rec = df['metrics/recall(B)'].iloc[0]
    first_f1 = 2 * (first_epoch_prec * first_epoch_rec) / (first_epoch_prec + first_epoch_rec) if (first_epoch_prec + first_epoch_rec) > 0 else 0
    f1_improvement = (best_f1 - first_f1) / first_f1 * 100 if first_f1 > 0 else 0
    map_improvement = (best_f1_map - df['metrics/mAP50(B)'].iloc[0]) / df['metrics/mAP50(B)'].iloc[0] * 100
    
    # Baseline comparison (use F1 score)
    comparison_lines = []
    if baselines:
        for name, metrics in baselines.items():
            baseline_f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
            if best_f1 > baseline_f1:
                diff = (best_f1 - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0
                comparison_lines.append(f"✓ Beating {name.split()[0]} by {diff:.1f}% (F1)")
            else:
                diff = (baseline_f1 - best_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0
                comparison_lines.append(f"✗ {name.split()[0]} ahead by {diff:.1f}% (F1)")
    
    stats = f"""
TRAINING STATUS

Epoch: {current_epoch}/150 ({current_epoch/150*100:.1f}%)

🏆 BEST MODEL: Epoch {best_f1_epoch}
  F1:   {best_f1:.4f} ⭐
  mAP:  {best_f1_map:.4f}
  Prec: {best_prec:.4f}
  Rec:  {best_rec:.4f}

Current (ep {current_epoch}):
  F1:   {current_f1:.4f}
  mAP:  {current_map:.4f}

Improvement from start:
  F1:   +{f1_improvement:.1f}%
  mAP:  +{map_improvement:.1f}%

Gap to Target (F1=0.66):
  {(0.66-best_f1)/0.66*100:.1f}% remaining

BASELINE COMPARISON
"""
    
    for line in comparison_lines:
        stats += f"\n{line}"
    
    stats += f"""

LOSSES (ep 1→{current_epoch})
Box: {df['train/box_loss'].iloc[0]:.3f}→{df['train/box_loss'].iloc[-1]:.3f}
Cls: {df['train/cls_loss'].iloc[0]:.3f}→{df['train/cls_loss'].iloc[-1]:.3f}
"""
    
    ax6.text(0.05, 0.95, stats, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")
    
    return {
        'current_epoch': current_epoch,
        'current_map': current_map,
        'current_f1': current_f1,
        'best_epoch': best_f1_epoch,  # Best based on F1 score
        'best_map': best_f1_map,
        'best_f1': best_f1,
        'best_map_epoch': best_map_epoch,  # For reference
        'best_map_overall': best_map,  # For reference
        'total_epochs': len(df)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training results and generate visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py --results runs/yolo11l_balanced/results.csv           # Analyze training
  python analyze.py --results runs/train/results.csv --watch              # Watch mode
  python analyze.py --test runs/train/weights/best.pt --compare           # Test model

Use --help for full options
        """
    )
    parser.add_argument('--results', type=str, help='Path to results.csv file')
    parser.add_argument('--output', type=str, default='training_analysis.png', help='Output plot filename')
    parser.add_argument('--dataset', type=str, default='dataset.yaml', help='Dataset YAML for baseline evaluation')
    parser.add_argument('--no-baselines', action='store_true', help='Skip baseline evaluation')
    parser.add_argument('--conf', type=float, default=0.01, help='Confidence threshold for baseline evaluation (default: 0.01)')
    parser.add_argument('--iou', type=float, default=0.2, help='IoU threshold for baseline evaluation (default: 0.2)')
    parser.add_argument('--imgsz', '-i', type=int, default=640, help='Image size for validation (default: 640)')
    parser.add_argument('--test', type=str, metavar='MODEL', help='Run model testing (calls test.py)')
    parser.add_argument('--compare', action='store_true', help='Compare with baselines when testing')
    parser.add_argument('--multi-scale', action='store_true', help='Enable multi-scale validation when testing')
    parser.add_argument('--watch', action='store_true', help='Continuously update visualization during training')
    parser.add_argument('--interval', type=int, default=30, help='Update interval in seconds for watch mode (default: 30)')
    
    args = parser.parse_args()
    
    # If --test is provided, run test.py
    if args.test:
        print("\n🔄 Running model testing via test.py...")
        print("="*80)
        
        test_script = Path(__file__).parent / 'test.py'
        if not test_script.exists():
            print(f"❌ test.py not found at {test_script}")
            sys.exit(1)
        
        cmd = [sys.executable, str(test_script), '--model', args.test, '--data', args.dataset]
        if args.compare:
            cmd.append('--compare')
        if args.multi_scale:
            cmd.append('--multi-scale')
        
        try:
            subprocess.run(cmd, check=True)
            print("\n✅ Testing completed!")
            return
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Testing failed: {e}")
            sys.exit(1)
    
    # Analysis mode - require results file
    if not args.results:
        # Try to find most recent results automatically
        runs_dir = Path('runs')
        if runs_dir.exists():
            # Find all results.csv files
            results_files = list(runs_dir.glob('*/results.csv'))
            if results_files:
                # Sort by modification time, get most recent
                latest = max(results_files, key=lambda p: p.stat().st_mtime)
                args.results = str(latest)
                print(f"ℹ️  Auto-detected latest results: {args.results}")
            else:
                # Try legacy paths
                default_paths = [
                    '../runs/train_alertcal/yolov8n_optimized_v1/results.csv',
                    'runs/train/results.csv',
                    '../runs/train/results.csv',
                ]
                
                for path in default_paths:
                    if Path(path).exists():
                        args.results = path
                        break
        
        if not args.results:
            print("❌ ERROR: No results file found. Please specify with --results")
            print("\nExamples:")
            print("  python analyze.py --results runs/train/results.csv")
            print("  python analyze.py --test runs/train/weights/best.pt")
            sys.exit(1)
    
    # Validate inputs
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌ ERROR: Results file not found: {args.results}")
        sys.exit(1)
    
    # Watch mode - continuous updates
    if args.watch:
        print("="*80)
        print("👁️  WATCH MODE - Continuous Training Monitoring")
        print("="*80)
        print(f"\nResults: {args.results}")
        print(f"Output:  {args.output}")
        print(f"Interval: {args.interval}s")
        print("\nPress Ctrl+C to stop\n")
        
        last_modified = 0
        iteration = 0
        baselines = {}  # Initialize baselines outside loop
        baselines_loaded = False
        
        # Load baselines once at the start
        if not args.no_baselines:
            print("📊 Loading baseline models (one-time evaluation)...")
            try:
                baselines = load_baseline_metrics(args.dataset, args.conf, args.iou, args.imgsz)
                baselines_loaded = True
                print("✓ Baselines loaded successfully\n")
            except Exception as e:
                print(f"⚠️  Baseline evaluation failed: {e}")
                print("   Continuing without baselines...\n")
        
        try:
            while True:
                current_modified = results_path.stat().st_mtime
                
                # Only update if file has changed
                if current_modified != last_modified:
                    iteration += 1
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    print(f"[{timestamp}] Update #{iteration}: Generating visualization...")
                    
                    try:
                        stats = create_visualization(
                            results_csv=args.results,
                            output_path=args.output,
                            baselines=baselines,  # Use pre-loaded baselines
                            show_baselines=baselines_loaded
                        )
                        
                        print(f"  ✓ Epoch {stats['current_epoch']}/{stats['total_epochs']}, "
                              f"F1={stats['current_f1']:.4f}, mAP={stats['current_map']:.4f}, "
                              f"Best F1={stats['best_f1']:.4f} @ Epoch {stats['best_epoch']}")
                        
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
                    
                    last_modified = current_modified
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n\n✋ Watch mode stopped by user")
            sys.exit(0)
    
    # Single-run mode
    print("="*80)
    print("📊 TRAINING ANALYSIS")
    print("="*80)
    print(f"\nResults: {args.results}")
    print(f"Output:  {args.output}")
    
    # Load baselines
    baselines = {}
    if not args.no_baselines:
        try:
            baselines = load_baseline_metrics(args.dataset, args.conf, args.iou, args.imgsz)
        except Exception as e:
            print(f"⚠️  Baseline evaluation failed: {e}")
            print("   Continuing without baselines...")
    
    # Create visualization
    try:
        stats = create_visualization(
            results_csv=args.results,
            output_path=args.output,
            baselines=baselines,
            show_baselines=not args.no_baselines
        )
        
        print("\n" + "="*80)
        print("🎉 ANALYSIS COMPLETE")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"  • Current: Epoch {stats['current_epoch']}, F1={stats['current_f1']:.4f}, mAP={stats['current_map']:.4f}")
        print(f"  • Best F1: Epoch {stats['best_epoch']}, F1={stats['best_f1']:.4f}, mAP={stats['best_map']:.4f}")
        
        if baselines:
            print(f"\n🎯 Baseline Comparison (F1 Score):")
            for name, metrics in baselines.items():
                baseline_f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
                status = "✅" if stats['best_f1'] > baseline_f1 else "⚠️"
                diff = abs(stats['best_f1'] - baseline_f1)
                print(f"  {status} {name}: F1={baseline_f1:.4f} (Δ {diff:.4f})")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
