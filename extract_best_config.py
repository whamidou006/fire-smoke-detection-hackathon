#!/usr/bin/env python3
"""
Extract best configuration from tuning_history.json

Usage:
    python extract_best_config.py [tuning_history.json]
"""
import json
import sys

history_file = sys.argv[1] if len(sys.argv) > 1 else "tuning_history.json"

with open(history_file) as f:
    history = json.load(f)

# Find best by F1 score
best = max(history, key=lambda x: x['f1_score'])

print(f"🏆 Best Configuration (Iteration {best['iteration']})")
print(f"   F1: {best['f1_score']:.4f}")
print(f"   Precision: {best['precision']:.4f}")
print(f"   Recall: {best['recall']:.4f}")
print(f"\n📋 Hyperparameters:")
print(json.dumps(best['config'], indent=2))
