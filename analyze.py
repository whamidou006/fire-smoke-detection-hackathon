#!/usr/bin/env python3
"""
Training Analysis & Visualization
Generates comprehensive plots comparing training progress with baselines
Can also run model testing by calling test.py
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
from ultralytics import YOLO


def load_baseline_metrics(dataset_yaml):
    """Load and evaluate baseline models"""
    baselines = {}
    
    baseline_models = {
        'Current Best': '/home/whamidouche/ssdprivate/datasets/fire_hackathon/fire_hackathon/weights/current_best.pt',
        'Pretrain YOLOv8': '/home/whamidouche/ssdprivate/datasets/fire_hackathon/fire_hackathon/weights/pretrain_yolov8.pt'
    }
    
    print("📊 Evaluating baseline models...")
    
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
                imgsz=640,
                verbose=False,
                conf=0.01,
                iou=0.2,
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
    
    # If evaluation fails, use known values
    if not baselines:
        print("  ℹ️  Using cached baseline metrics")
        baselines = {
            'Current Best': {'map50': 0.4149, 'precision': 0.6461, 'recall': 0.3598},
            'Pretrain YOLOv8': {'map50': 0.1944, 'precision': 0.3575, 'recall': 0.1650}
        }
    
    return baselines


def create_visualization(results_csv, output_path, baselines, show_baselines=True):
    """Create comprehensive 6-panel training visualization"""
    
    # Load training results
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    current_epoch = int(df['epoch'].iloc[-1])
    current_map = df['metrics/mAP50(B)'].iloc[-1]
    best_map = df['metrics/mAP50(B)'].max()
    best_epoch = int(df[df['metrics/mAP50(B)'] == best_map]['epoch'].values[0])
    
    print(f"\n📈 Training Progress:")
    print(f"  • Current: Epoch {current_epoch}, mAP@0.5={current_map:.4f}")
    print(f"  • Best:    Epoch {best_epoch}, mAP@0.5={best_map:.4f}")
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Training Performance Analysis - Fire/Smoke Detection\nwith Baseline Comparisons',
                 fontsize=16, fontweight='bold')
    
    # Colors for baselines
    colors = {'Current Best': 'red', 'Pretrain YOLOv8': 'orange'}
    
    # ========================================================================
    # Plot 1: mAP@0.5 Progress (Main plot - spans 2 columns)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Raw and smoothed data
    ax1.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', alpha=0.3, label='Raw')
    window = 5
    rolling = df['metrics/mAP50(B)'].rolling(window=window, center=True).mean()
    ax1.plot(df['epoch'], rolling, 'b-', linewidth=2.5, label=f'{window}-epoch average')
    
    # Baseline lines
    if show_baselines and baselines:
        for name, metrics in baselines.items():
            ax1.axhline(y=metrics['map50'], color=colors[name], linestyle='--',
                       linewidth=2, alpha=0.7, label=f'{name} ({metrics["map50"]:.3f})')
    
    # Target and current marker
    ax1.axhline(y=0.60, color='g', linestyle=':', linewidth=2, alpha=0.5, label='Target (0.60)')
    ax1.axvline(x=current_epoch, color='purple', linestyle=':', alpha=0.5,
               label=f'Current (epoch {current_epoch})')
    
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
    
    ax2.plot(df['epoch'], df['metrics/precision(B)'], 'g-', alpha=0.5, label='Precision')
    ax2.plot(df['epoch'], df['metrics/recall(B)'], 'r-', alpha=0.5, label='Recall')
    
    # Smoothed
    ax2.plot(df['epoch'], df['metrics/precision(B)'].rolling(window, center=True).mean(), 'g-', linewidth=2)
    ax2.plot(df['epoch'], df['metrics/recall(B)'].rolling(window, center=True).mean(), 'r-', linewidth=2)
    
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
    ax3.axvline(x=current_epoch, color='orange', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Training Loss Curves (Decreasing = Learning)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 4: Bar Chart Comparison
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])
    
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
    # Plot 5: Metrics Table
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    text = "METRICS COMPARISON\n" + "="*40 + "\n\n"
    text += f"{'Model':<18} {'mAP':<8} {'Prec':<8} {'Recall':<8}\n"
    text += "-"*45 + "\n"
    
    # Your model
    best_idx = df['metrics/mAP50(B)'].idxmax()
    text += f"{'Your (Now)':<18} {current_map:<8.4f} {df['metrics/precision(B)'].iloc[-1]:<8.4f} {df['metrics/recall(B)'].iloc[-1]:<8.4f}\n"
    text += f"{'Your (Best)':<18} {best_map:<8.4f} {df['metrics/precision(B)'].iloc[best_idx]:<8.4f} {df['metrics/recall(B)'].iloc[best_idx]:<8.4f}\n"
    
    # Baselines
    if baselines:
        for name, metrics in baselines.items():
            short_name = name.split()[0]
            text += f"{short_name:<18} {metrics['map50']:<8.4f} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f}\n"
    
    text += "\n" + "-"*45 + "\n"
    text += "TARGET           0.6000   0.6000   0.7000\n"
    
    ax5.text(0.05, 0.95, text, transform=ax5.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # ========================================================================
    # Plot 6: Summary Statistics
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    improvement = (best_map - df['metrics/mAP50(B)'].iloc[0]) / df['metrics/mAP50(B)'].iloc[0] * 100
    
    # Baseline comparison
    comparison_lines = []
    if baselines:
        for name, metrics in baselines.items():
            if best_map > metrics['map50']:
                diff = (best_map - metrics['map50']) / metrics['map50'] * 100
                comparison_lines.append(f"✓ Beating {name.split()[0]} by {diff:.1f}%")
            else:
                diff = (metrics['map50'] - best_map) / metrics['map50'] * 100
                comparison_lines.append(f"✗ {name.split()[0]} ahead by {diff:.1f}%")
    
    stats = f"""
TRAINING STATUS

Epoch: {current_epoch}/150 ({current_epoch/150*100:.1f}%)
Best:  ep {best_epoch} ({best_map:.4f})

Gap to Target (0.60):
  {(0.60-best_map)/0.60*100:.1f}% remaining

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
        'best_epoch': best_epoch,
        'best_map': best_map
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training results and generate visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze training results
  python analyze.py --results runs/train/results.csv
  
  # Skip baseline comparison (faster)
  python analyze.py --results runs/train/results.csv --no-baselines
  
  # Run model testing
  python analyze.py --test runs/train/weights/best.pt
  
  # Compare multiple models
  python analyze.py --test runs/train/weights/best.pt --compare
        """
    )
    parser.add_argument('--results', type=str, help='Path to results.csv file')
    parser.add_argument('--output', type=str, default='training_analysis.png', help='Output plot filename')
    parser.add_argument('--dataset', type=str, default='dataset.yaml', help='Dataset YAML for baseline evaluation')
    parser.add_argument('--no-baselines', action='store_true', help='Skip baseline evaluation')
    parser.add_argument('--test', type=str, metavar='MODEL', help='Run model testing (calls test.py)')
    parser.add_argument('--compare', action='store_true', help='Compare with baselines when testing')
    
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
        
        try:
            subprocess.run(cmd, check=True)
            print("\n✅ Testing completed!")
            return
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Testing failed: {e}")
            sys.exit(1)
    
    # Analysis mode - require results file
    if not args.results:
        # Try to find results automatically
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
    
    print("="*80)
    print("📊 TRAINING ANALYSIS")
    print("="*80)
    print(f"\nResults: {args.results}")
    print(f"Output:  {args.output}")
    
    # Load baselines
    baselines = {}
    if not args.no_baselines:
        try:
            baselines = load_baseline_metrics(args.dataset)
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
        print(f"  • Current: Epoch {stats['current_epoch']}, mAP={stats['current_map']:.4f}")
        print(f"  • Best:    Epoch {stats['best_epoch']}, mAP={stats['best_map']:.4f}")
        
        if baselines:
            print(f"\n🎯 Baseline Comparison:")
            for name, metrics in baselines.items():
                status = "✅" if stats['best_map'] > metrics['map50'] else "⚠️"
                diff = abs(stats['best_map'] - metrics['map50'])
                print(f"  {status} {name}: {metrics['map50']:.4f} (Δ {diff:.4f})")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
