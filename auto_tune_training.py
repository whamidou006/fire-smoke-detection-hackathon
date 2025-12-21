#!/usr/bin/env python3
"""
Automated Training Hyperparameter Tuning with GPT-5 Reasoning

Iteratively trains models and uses GPT-5 to analyze results and suggest
configuration improvements for better F1 score performance.

Usage:
    python auto_tune_training.py --model 11l --initial-config balanced --iterations 5
    python auto_tune_training.py --config auto_tune_config.yaml
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Add LLM judge utilities to path
llm_judge_path = "/home/whamidouche/ssdprivate/git-local/low-resource-llm.pipeline/multilingual-llm-evaluation"
if llm_judge_path not in sys.path:
    sys.path.insert(0, llm_judge_path)

from llm_judge_utils import initialize_endpoint_clients, generate_endpoint_response

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "auto_tune_config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dict
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def parse_results_csv(results_path: str, config: Dict) -> Dict:
    """
    Parse training results CSV and extract key metrics.
    
    Args:
        results_path: Path to results.csv file
        config: Configuration dict with CSV column mappings
        
    Returns:
        Dict with final metrics and training history
    """
    if not os.path.exists(results_path):
        logger.error(f"Results file not found: {results_path}")
        return {}
    
    # Get column mappings from config
    csv_cols = config.get('metrics', {}).get('csv_columns', {})
    
    metrics = {
        'final_epoch': 0,
        'precision': 0.0,
        'recall': 0.0,
        'mAP50': 0.0,
        'mAP50_95': 0.0,
        'f1_score': 0.0,
        'train_box_loss': 0.0,
        'train_cls_loss': 0.0,
        'train_dfl_loss': 0.0,
        'val_box_loss': 0.0,
        'val_cls_loss': 0.0,
        'val_dfl_loss': 0.0,
        'history': []
    }
    
    try:
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if not rows:
                return metrics
            
            # Store full history for trend analysis
            for row in rows:
                metrics['history'].append({
                    'epoch': int(row.get(csv_cols.get('epoch', 'epoch'), 0)),
                    'precision': float(row.get(csv_cols.get('precision', 'metrics/precision(B)'), 0)),
                    'recall': float(row.get(csv_cols.get('recall', 'metrics/recall(B)'), 0)),
                    'mAP50': float(row.get(csv_cols.get('mAP50', 'metrics/mAP50(B)'), 0)),
                    'mAP50_95': float(row.get(csv_cols.get('mAP50_95', 'metrics/mAP50-95(B)'), 0)),
                    'train_box_loss': float(row.get(csv_cols.get('train_box_loss', 'train/box_loss'), 0)),
                    'train_cls_loss': float(row.get(csv_cols.get('train_cls_loss', 'train/cls_loss'), 0)),
                    'train_dfl_loss': float(row.get(csv_cols.get('train_dfl_loss', 'train/dfl_loss'), 0)),
                    'val_box_loss': float(row.get(csv_cols.get('val_box_loss', 'val/box_loss'), 0)),
                    'val_cls_loss': float(row.get(csv_cols.get('val_cls_loss', 'val/cls_loss'), 0)),
                    'val_dfl_loss': float(row.get(csv_cols.get('val_dfl_loss', 'val/dfl_loss'), 0)),
                })
            
            # Get final epoch metrics
            last_row = rows[-1]
            metrics['final_epoch'] = int(last_row.get(csv_cols.get('epoch', 'epoch'), 0))
            metrics['precision'] = float(last_row.get(csv_cols.get('precision', 'metrics/precision(B)'), 0))
            metrics['recall'] = float(last_row.get(csv_cols.get('recall', 'metrics/recall(B)'), 0))
            metrics['mAP50'] = float(last_row.get(csv_cols.get('mAP50', 'metrics/mAP50(B)'), 0))
            metrics['mAP50_95'] = float(last_row.get(csv_cols.get('mAP50_95', 'metrics/mAP50-95(B)'), 0))
            metrics['train_box_loss'] = float(last_row.get(csv_cols.get('train_box_loss', 'train/box_loss'), 0))
            metrics['train_cls_loss'] = float(last_row.get(csv_cols.get('train_cls_loss', 'train/cls_loss'), 0))
            metrics['train_dfl_loss'] = float(last_row.get(csv_cols.get('train_dfl_loss', 'train/dfl_loss'), 0))
            metrics['val_box_loss'] = float(last_row.get(csv_cols.get('val_box_loss', 'val/box_loss'), 0))
            metrics['val_cls_loss'] = float(last_row.get(csv_cols.get('val_cls_loss', 'val/cls_loss'), 0))
            metrics['val_dfl_loss'] = float(last_row.get(csv_cols.get('val_dfl_loss', 'val/dfl_loss'), 0))
            
            # Calculate F1 score
            if metrics['precision'] > 0 or metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            
            logger.info(f"✅ Parsed {len(rows)} epochs from results")
            logger.info(f"Final metrics - P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
            
    except Exception as e:
        logger.error(f"Error parsing results CSV: {e}")
    
    return metrics


def format_metrics_summary(metrics: Dict, iteration: int, config_name: str, yaml_config: Dict) -> str:
    """
    Format metrics into human-readable summary for GPT-5.
    
    Args:
        metrics: Metrics dict from parse_results_csv
        iteration: Current iteration number
        config_name: Configuration name used
        yaml_config: YAML configuration dict
        
    Returns:
        Formatted summary string
    """
    if not metrics:
        return "No metrics available - training may have failed."
    
    # Analyze trends from history
    history = metrics.get('history', [])
    trend_analysis = ""
    
    trend_cfg = yaml_config.get('trend_analysis', {})
    if trend_cfg.get('enabled', True) and len(history) >= 10:
        early_n = trend_cfg.get('early_epochs', 10)
        late_n = trend_cfg.get('late_epochs', 10)
        
        # Compare first N epochs vs last N epochs
        early_metrics = history[:early_n]
        late_metrics = history[-late_n:]
        
        early_precision = sum(e['precision'] for e in early_metrics) / len(early_metrics)
        late_precision = sum(e['precision'] for e in late_metrics) / len(late_metrics)
        
        early_recall = sum(e['recall'] for e in early_metrics) / len(early_metrics)
        late_recall = sum(e['recall'] for e in late_metrics) / len(late_metrics)
        
        trend_analysis = f"""
Training Trend Analysis:
- Precision: {early_precision:.4f} (early) → {late_precision:.4f} (late) [{'+' if late_precision > early_precision else ''}{late_precision - early_precision:.4f}]
- Recall: {early_recall:.4f} (early) → {late_recall:.4f} (late) [{'+' if late_recall > early_recall else ''}{late_recall - early_recall:.4f}]
"""
    
    summary = f"""
=== Training Iteration {iteration} Results ===
Configuration: {config_name}
Total Epochs: {metrics['final_epoch']}

Final Performance Metrics:
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1 Score: {metrics['f1_score']:.4f} ⭐
- mAP@50: {metrics['mAP50']:.4f}
- mAP@50-95: {metrics['mAP50_95']:.4f}

Final Loss Values:
- Training: box={metrics['train_box_loss']:.4f}, cls={metrics['train_cls_loss']:.4f}, dfl={metrics['train_dfl_loss']:.4f}
- Validation: box={metrics['val_box_loss']:.4f}, cls={metrics['val_cls_loss']:.4f}, dfl={metrics['val_dfl_loss']:.4f}
{trend_analysis}
Problem Analysis:
- Precision > Recall? {metrics['precision'] > metrics['recall']} (model is conservative, missing detections)
- Recall > Precision? {metrics['recall'] > metrics['precision']} (model is aggressive, false positives)
- Balanced? {abs(metrics['precision'] - metrics['recall']) < 0.05}
"""
    
    return summary


def create_gpt5_tuning_prompt(
    current_metrics: Dict,
    previous_configs: List[Dict],
    iteration: int,
    config_name: str,
    yaml_config: Dict
) -> str:
    """
    Create prompt for GPT-5 to suggest configuration improvements.
    
    Args:
        current_metrics: Current training metrics
        previous_configs: List of previous configs and their results
        iteration: Current iteration number
        config_name: Current configuration name
        yaml_config: YAML configuration dict
        
    Returns:
        Formatted prompt for GPT-5
    """
    metrics_summary = format_metrics_summary(current_metrics, iteration, config_name, yaml_config)
    
    # Format previous iterations history
    history_text = ""
    if previous_configs:
        history_text = "\n=== Previous Iterations ===\n"
        for i, prev in enumerate(previous_configs, 1):
            history_text += f"""
Iteration {i}:
- Config: {prev['config_name']}
- F1: {prev['f1_score']:.4f}, Precision: {prev['precision']:.4f}, Recall: {prev['recall']:.4f}
- Loss weights: cls={prev['config'].get('cls', 'N/A')}, box={prev['config'].get('box', 'N/A')}, dfl={prev['config'].get('dfl', 'N/A')}
- FL gamma: {prev['config'].get('fl_gamma', 'N/A')}
"""
    
    # Get prompt template from config
    prompt_template = yaml_config.get('gpt5', {}).get('tuning_prompt_template', '')
    
    if not prompt_template:
        logger.error("No tuning_prompt_template found in config")
        return ""
    
    # Format template with actual values
    prompt = prompt_template.format(
        metrics_summary=metrics_summary,
        history_text=history_text,
        current_config=json.dumps(current_metrics.get('config', {}), indent=2)
    )
    
    return prompt


def parse_gpt5_recommendations(response: str) -> Optional[Dict]:
    """
    Parse GPT-5 response to extract recommended configuration changes.
    
    Args:
        response: GPT-5 response text
        
    Returns:
        Dict with recommended config values or None if parsing fails
    """
    if not response or response.startswith("Error:"):
        logger.error(f"Invalid GPT-5 response: {response}")
        return None
    
    try:
        config = {}
        in_recommended_section = False
        
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith("Recommended_Changes:"):
                in_recommended_section = True
                continue
            
            if in_recommended_section:
                # Stop at next section
                if line.startswith("Rationale:") or line.startswith("Expected_Improvement:"):
                    break
                
                # Parse key: value pairs
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Skip empty or placeholder values
                    if value.lower() in ('', 'n/a', 'none', 'same', 'unchanged', '[value]'):
                        continue
                    
                    # Convert to appropriate type
                    try:
                        # Try float first
                        if '.' in value or 'e-' in value.lower():
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                    except ValueError:
                        # Keep as string if not a number
                        config[key] = value
        
        if config:
            logger.info(f"✅ Parsed {len(config)} recommendations from GPT-5")
            return config
        else:
            logger.warning("⚠️ No recommendations found in GPT-5 response")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing GPT-5 recommendations: {e}")
        return None


def run_training(
    model: str,
    config_name: str,
    custom_config: Optional[Dict] = None,
    batch_size: int = 128,
    imgsz: int = 640,
    device: str = "0",
    epochs: int = 150
) -> Tuple[bool, str]:
    """
    Run training with specified configuration.
    
    Args:
        model: Model name (e.g., '11l', '11x')
        config_name: Base config name or 'custom' for custom config
        custom_config: Optional custom hyperparameter dict
        batch_size: Batch size
        imgsz: Image size
        device: GPU device
        epochs: Number of epochs
        
    Returns:
        Tuple of (success, results_path)
    """
    # Build command
    cmd = [
        'python', 'train.py',
        '--model', model,
        '--config', config_name,
        '--batch', str(batch_size),
        '--imgsz', str(imgsz),
        '--device', device,
        '--epochs', str(epochs)
    ]
    
    logger.info(f"🚀 Starting training: {' '.join(cmd)}")
    
    # If custom config provided, temporarily modify train.py or pass as json
    # For simplicity, we'll write it to a temp config file
    if custom_config:
        config_file = 'auto_tune_config.json'
        with open(config_file, 'w') as f:
            json.dump(custom_config, f, indent=2)
        logger.info(f"Custom config saved to {config_file}")
    
    try:
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("✅ Training completed successfully")
            
            # Find results path
            results_path = f"runs/{model}_{config_name}/results.csv"
            if os.path.exists(results_path):
                return True, results_path
            else:
                logger.error(f"Results file not found: {results_path}")
                return False, ""
        else:
            logger.error(f"❌ Training failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout[-500:]}")
            logger.error(f"STDERR: {result.stderr[-500:]}")
            return False, ""
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Training timeout (2 hours)")
        return False, ""
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        return False, ""


def auto_tune_training(
    model: str,
    initial_config: str,
    iterations: int,
    batch_size: int,
    imgsz: int,
    device: str,
    epochs: int,
    output_dir: str,
    yaml_config: Dict
):
    """
    Main auto-tuning loop using GPT-5 for configuration optimization.
    
    Args:
        model: Model name
        initial_config: Initial configuration name
        iterations: Number of tuning iterations
        batch_size: Batch size
        imgsz: Image size
        device: GPU device
        epochs: Epochs per training run
        output_dir: Output directory for logs
        yaml_config: YAML configuration dict
    """
    logger.info("=" * 80)
    logger.info("🔧 Starting Automated Training Hyperparameter Tuning with GPT-5")
    logger.info("=" * 80)
    
    # Initialize API clients
    logger.info("Initializing GPT-5 API client...")
    clients = initialize_endpoint_clients()
    
    if not clients.get("azure_client"):
        logger.error("❌ Failed to initialize Azure client for GPT-5")
        return
    
    # Track history
    history = []
    best_f1 = 0.0
    best_config = None
    best_iteration = 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    history_filename = yaml_config.get('output', {}).get('history_filename', 'tuning_history.json')
    history_file = os.path.join(output_dir, history_filename)
    
    # Get GPT-5 config
    gpt5_config = yaml_config.get('gpt5', {})
    
    for iteration in range(1, iterations + 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📊 ITERATION {iteration}/{iterations}")
        logger.info(f"{'=' * 80}\n")
        
        # Determine configuration for this iteration
        if iteration == 1:
            config_name = initial_config
            custom_config = None
            logger.info(f"Using initial configuration: {config_name}")
        else:
            # Use GPT-5 recommendations from previous iteration
            config_name = f"auto_tuned_iter{iteration}"
            custom_config = history[-1].get('recommended_config')
            logger.info(f"Using GPT-5 recommended configuration from iteration {iteration-1}")
        
        # Run training
        success, results_path = run_training(
            model=model,
            config_name=config_name,
            custom_config=custom_config,
            batch_size=batch_size,
            imgsz=imgsz,
            device=device,
            epochs=epochs
        )
        
        if not success:
            logger.error(f"❌ Training failed at iteration {iteration}")
            break
        
        # Parse results
        metrics = parse_results_csv(results_path, yaml_config)
        if not metrics:
            logger.error(f"❌ Failed to parse results at iteration {iteration}")
            break
        
        # Track best F1 score
        current_f1 = metrics['f1_score']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_config = metrics.get('config', custom_config)
            best_iteration = iteration
            logger.info(f"🎯 New best F1 score: {best_f1:.4f}")
        
        # Add current config to metrics for GPT-5 context
        metrics['config'] = custom_config or {}
        
        # Request GPT-5 recommendations for next iteration
        if iteration < iterations:
            logger.info("\n🤖 Requesting GPT-5 recommendations for next iteration...")
            
            gpt5_prompt = create_gpt5_tuning_prompt(
                current_metrics=metrics,
                previous_configs=history,
                iteration=iteration,
                config_name=config_name,
                yaml_config=yaml_config
            )
            
            # Build GPT-5 model config from YAML
            gpt5_model_config = {
                "model": gpt5_config.get('model', 'gpt-5'),
                "generation": gpt5_config.get('generation', {})
            }
            
            gpt5_response = generate_endpoint_response(
                prompt=gpt5_prompt,
                model_config=gpt5_model_config,
                clients=clients,
                system_prompt=gpt5_config.get('system_prompt')
            )
            
            # Parse recommendations
            recommended_config = parse_gpt5_recommendations(gpt5_response)
            
            if not recommended_config:
                logger.warning("⚠️ Failed to parse GPT-5 recommendations, using previous config")
                recommended_config = custom_config or {}
            
            # Save to history
            history.append({
                'iteration': iteration,
                'config_name': config_name,
                'f1_score': current_f1,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'mAP50': metrics['mAP50'],
                'config': custom_config or {},
                'recommended_config': recommended_config,
                'gpt5_response': gpt5_response if yaml_config.get('output', {}).get('save_gpt5_responses', True) else None
            })
            
            # Save history to file
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"✅ GPT-5 recommendations saved for iteration {iteration + 1}")
        else:
            # Last iteration - just save results
            history.append({
                'iteration': iteration,
                'config_name': config_name,
                'f1_score': current_f1,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'mAP50': metrics['mAP50'],
                'config': custom_config or {}
            })
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
    
    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("📈 AUTO-TUNING SUMMARY")
    logger.info(f"{'=' * 80}\n")
    logger.info(f"Total iterations: {len(history)}")
    logger.info(f"Best F1 score: {best_f1:.4f} (iteration {best_iteration})")
    logger.info(f"\nPerformance progression:")
    for h in history:
        logger.info(f"  Iteration {h['iteration']}: F1={h['f1_score']:.4f}, P={h['precision']:.4f}, R={h['recall']:.4f}")
    
    if best_config:
        logger.info(f"\n🏆 Best configuration (iteration {best_iteration}):")
        logger.info(json.dumps(best_config, indent=2))
    
    logger.info(f"\n✅ Full tuning history saved to: {history_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated training hyperparameter tuning with GPT-5 reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration file (recommended)
  python auto_tune_training.py --config auto_tune_config.yaml
  
  # Override config with command-line arguments
  python auto_tune_training.py --config auto_tune_config.yaml --model 11x --iterations 3
  
  # Auto-tune YOLOv11l starting from balanced config for 5 iterations
  python auto_tune_training.py --model 11l --initial-config balanced --iterations 5
  
  # Auto-tune YOLOv11x with high-resolution and 3 iterations
  python auto_tune_training.py --model 11x --initial-config balanced --iterations 3 --imgsz 1024 --batch 64
  
  # Quick tuning with fewer epochs per iteration
  python auto_tune_training.py --model 11l --initial-config recall_focused --iterations 3 --epochs 50
"""
    )
    
    parser.add_argument('--config', type=str, default='auto_tune_config.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (11l, 11x, etc.) - overrides config')
    parser.add_argument('--initial-config', type=str, default=None,
                        help='Initial configuration (balanced, recall_focused, etc.) - overrides config')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of tuning iterations - overrides config')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size - overrides config')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='Image size (640 or 1024) - overrides config')
    parser.add_argument('--device', type=str, default=None,
                        help='GPU device (0 or 0,1 for multi-GPU) - overrides config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Epochs per training run - overrides config')
    parser.add_argument('--output-dir', type=str, default='auto_tune_logs',
                        help='Output directory for tuning logs')
    
    args = parser.parse_args()
    
    # Load YAML configuration
    yaml_config = load_config(args.config)
    if not yaml_config:
        logger.error("Failed to load configuration file")
        return
    
    # Get default values from config
    defaults = yaml_config.get('default_args', {})
    
    # Command-line args override config file
    model = args.model or defaults.get('model', '11l')
    initial_config = args.initial_config or defaults.get('initial_config', 'balanced')
    iterations = args.iterations or defaults.get('iterations', 5)
    batch_size = args.batch or defaults.get('batch_size', 128)
    imgsz = args.imgsz or defaults.get('imgsz', 640)
    device = args.device or defaults.get('device', '0')
    epochs = args.epochs or yaml_config.get('training', {}).get('default_epochs', 150)
    
    logger.info(f"Configuration: model={model}, config={initial_config}, iterations={iterations}")
    logger.info(f"Training: batch={batch_size}, imgsz={imgsz}, device={device}, epochs={epochs}")
    
    auto_tune_training(
        model=model,
        initial_config=initial_config,
        iterations=iterations,
        batch_size=batch_size,
        imgsz=imgsz,
        device=device,
        epochs=epochs,
        output_dir=args.output_dir,
        yaml_config=yaml_config
    )


if __name__ == '__main__':
    main()
