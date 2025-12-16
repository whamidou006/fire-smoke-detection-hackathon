#!/usr/bin/env python3
"""
Merge Hard Examples (FN/FP) with Existing Dataset

This script merges false negative and false positive images from current_best model
with your existing training dataset to improve model performance.

Strategy:
1. False Negatives (2,521): Model missed these fires → Add to training with annotations
2. False Positives (4,256): Model detected fire incorrectly → Add as hard negatives

Expected improvement: +10-20% mAP (hard example mining is very effective!)

Usage:
    python merge_hard_examples.py --check  # Dry run to see what will be done
    python merge_hard_examples.py --merge  # Actually merge the datasets
"""

import os
import shutil
from pathlib import Path
import json
from collections import defaultdict
from PIL import Image

# Paths
BASE_DATA_DIR = Path("/home/whamidouche/ssdprivate/datasets/fire_hackathon/fire_hackathon/data")
FN_DIR = BASE_DATA_DIR / "export_111225_fn"
FP_DIR = BASE_DATA_DIR / "export_111225_fp"
CURRENT_DATASET = BASE_DATA_DIR / "Fire_data_v2_yolo_with_blank_images_and_false_positives"

OUTPUT_DATASET = BASE_DATA_DIR / "Fire_data_v3_with_hard_examples"


def check_structure():
    """Check dataset structure and count images"""
    print("="*80)
    print("🔍 CHECKING DATASET STRUCTURE")
    print("="*80)
    
    # Check FN (False Negatives)
    fn_images = list(FN_DIR.glob("images/**/*.jpg")) + list(FN_DIR.glob("images/**/*.png"))
    fn_train_json = FN_DIR / "111225_export_train.json"
    fn_test_json = FN_DIR / "111225_export_test.json"
    
    print(f"\n📁 False Negatives (FN): {FN_DIR}")
    print(f"   Images: {len(fn_images)}")
    print(f"   Train annotations: {'✓' if fn_train_json.exists() else '✗'}")
    print(f"   Test annotations:  {'✓' if fn_test_json.exists() else '✗'}")
    
    # Check FP (False Positives)
    fp_images = list(FP_DIR.glob("images/**/*.jpg")) + list(FP_DIR.glob("images/**/*.png"))
    fp_train_csv = FP_DIR / "train.csv"
    fp_test_csv = FP_DIR / "test.csv"
    
    print(f"\n📁 False Positives (FP): {FP_DIR}")
    print(f"   Images: {len(fp_images)}")
    print(f"   Train split: {'✓' if fp_train_csv.exists() else '✗'}")
    print(f"   Test split:  {'✓' if fp_test_csv.exists() else '✗'}")
    print(f"   Note: FP are blanks (no annotations)")
    
    # Check current dataset
    current_train = CURRENT_DATASET / "train" / "images"
    current_test = CURRENT_DATASET / "test" / "images"
    
    if current_train.exists():
        train_images = list(current_train.glob("*.jpg")) + list(current_train.glob("*.png"))
        print(f"\n📁 Current Dataset: {CURRENT_DATASET}")
        print(f"   Train images: {len(train_images)}")
    
    if current_test.exists():
        test_images = list(current_test.glob("*.jpg")) + list(current_test.glob("*.png"))
        print(f"   Test images:  {len(test_images)}")
    
    print("\n" + "="*80)
    print("📊 MERGING STRATEGY")
    print("="*80)
    print(f"""
1. False Negatives (FN): {len(fn_images)} images
   - Model MISSED these fires (low recall)
   - Action: Convert COCO annotations → YOLO format
   - Add to training set with fire/smoke labels
   - Expected gain: +5-10% recall
   
2. False Positives (FP): {len(fp_images)} images  
   - Model detected fire when there was NONE (low precision)
   - Action: Add as blank/negative samples
   - No annotations needed (empty .txt files)
   - Expected gain: +5-10% precision
   
3. Keep existing dataset: {len(train_images)} train + {len(test_images)} test
   - Don't remove anything
   - Just add hard examples

Total new training images: {len(fn_images) + len(fp_images):,}
New total training set: ~{len(train_images) + len(fn_images) + len(fp_images):,} images
""")
    
    return {
        'fn_images': len(fn_images),
        'fp_images': len(fp_images),
        'current_train': len(train_images) if current_train.exists() else 0,
        'current_test': len(test_images) if current_test.exists() else 0
    }


def coco_to_yolo(coco_bbox, img_width, img_height):
    """
    Convert COCO bbox format [x, y, width, height] to YOLO format [x_center, y_center, width, height]
    All values normalized to [0, 1]
    
    Args:
        coco_bbox: [x_min, y_min, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        [x_center, y_center, width, height] normalized
    """
    x, y, w, h = coco_bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return [x_center, y_center, width, height]


def merge_datasets(dry_run=True):
    """
    Merge FN and FP datasets with current dataset
    
    Args:
        dry_run: If True, only print what would be done without actually copying
    """
    print("\n" + "="*80)
    if dry_run:
        print("🔍 DRY RUN MODE (no files will be modified)")
    else:
        print("🚀 MERGING DATASETS")
    print("="*80)
    
    if not dry_run:
        # Create output directories
        output_train_images = OUTPUT_DATASET / "train" / "images"
        output_train_labels = OUTPUT_DATASET / "train" / "labels"
        output_test_images = OUTPUT_DATASET / "test" / "images"
        output_test_labels = OUTPUT_DATASET / "test" / "labels"
        
        for dir_path in [output_train_images, output_train_labels, output_test_images, output_test_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created output directory: {OUTPUT_DATASET}")
    
    # Step 1: Copy existing dataset
    print(f"\n📦 Step 1: Copying existing dataset...")
    current_train_images = CURRENT_DATASET / "train" / "images"
    current_train_labels = CURRENT_DATASET / "train" / "labels"
    
    if current_train_images.exists():
        train_files = list(current_train_images.glob("*.jpg")) + list(current_train_images.glob("*.png"))
        print(f"   Found {len(train_files)} training images")
        
        if not dry_run:
            for img_file in train_files:
                shutil.copy2(img_file, output_train_images / img_file.name)
                
                # Copy label if exists
                label_file = current_train_labels / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, output_train_labels / label_file.name)
            print(f"   ✓ Copied {len(train_files)} images + labels")
    
    # Copy test set
    current_test_images = CURRENT_DATASET / "test" / "images"
    current_test_labels = CURRENT_DATASET / "test" / "labels"
    
    if current_test_images.exists():
        test_files = list(current_test_images.glob("*.jpg")) + list(current_test_images.glob("*.png"))
        print(f"   Found {len(test_files)} test images")
        
        if not dry_run:
            for img_file in test_files:
                shutil.copy2(img_file, output_test_images / img_file.name)
                
                label_file = current_test_labels / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, output_test_labels / label_file.name)
            print(f"   ✓ Copied {len(test_files)} images + labels")
    
    # Step 2: Add False Negatives with annotations
    print(f"\n🔥 Step 2: Adding False Negatives (fires model missed)...")
    
    # Process FN TRAIN split
    fn_train_json = FN_DIR / "111225_export_train.json"
    fn_train_count = 0
    
    if fn_train_json.exists():
        with open(fn_train_json, 'r') as f:
            fn_data = json.load(f)
        
        # Build image id to filename map
        image_map = {img['id']: img for img in fn_data['images']}
        
        # Group annotations by image
        annotations_by_image = defaultdict(list)
        for ann in fn_data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
        
        for img_id, annotations in annotations_by_image.items():
            img_info = image_map.get(img_id)
            if not img_info:
                continue
            
            # Find actual image file (in subdirectories)
            # file_name might have full path, extract just the filename
            img_filename = os.path.basename(img_info['file_name'])
            img_files = list(FN_DIR.glob(f"images/**/{img_filename}"))
            
            if not img_files:
                continue
            
            img_path = img_files[0]
            fn_train_count += 1
            
            if not dry_run:
                # Copy image to TRAIN
                dest_img = output_train_images / f"fn_{img_path.name}"
                shutil.copy2(img_path, dest_img)
                
                # Convert annotations to YOLO format
                yolo_lines = []
                img_width = img_info['width']
                img_height = img_info['height']
                
                # Skip if image dimensions are invalid
                if img_width <= 0 or img_height <= 0:
                    # Get actual dimensions from image file
                    try:
                        with Image.open(img_path) as img:
                            img_width, img_height = img.size
                    except:
                        continue
                
                for ann in annotations:
                    category_id = ann['category_id'] - 1  # COCO is 1-indexed, YOLO is 0-indexed
                    bbox = ann['bbox']
                    yolo_bbox = coco_to_yolo(bbox, img_width, img_height)
                    yolo_lines.append(f"{category_id} {' '.join(map(str, yolo_bbox))}\n")
                
                # Write YOLO label file
                label_path = output_train_labels / f"fn_{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.writelines(yolo_lines)
        
        print(f"   {'Would add' if dry_run else '✓ Added'} {fn_train_count} FN train images with annotations")
    
    # Process FN TEST split
    fn_test_json = FN_DIR / "111225_export_test.json"
    fn_test_count = 0
    
    if fn_test_json.exists():
        with open(fn_test_json, 'r') as f:
            fn_test_data = json.load(f)
        
        # Build image id to filename map
        test_image_map = {img['id']: img for img in fn_test_data['images']}
        
        # Group annotations by image
        test_annotations_by_image = defaultdict(list)
        for ann in fn_test_data['annotations']:
            test_annotations_by_image[ann['image_id']].append(ann)
        
        for img_id, annotations in test_annotations_by_image.items():
            img_info = test_image_map.get(img_id)
            if not img_info:
                continue
            
            # Find actual image file
            img_filename = os.path.basename(img_info['file_name'])
            img_files = list(FN_DIR.glob(f"images/**/{img_filename}"))
            
            if not img_files:
                continue
            
            img_path = img_files[0]
            fn_test_count += 1
            
            if not dry_run:
                # Copy image to TEST
                dest_img = output_test_images / f"fn_{img_path.name}"
                shutil.copy2(img_path, dest_img)
                
                # Convert annotations to YOLO format
                yolo_lines = []
                img_width = img_info['width']
                img_height = img_info['height']
                
                if img_width <= 0 or img_height <= 0:
                    try:
                        with Image.open(img_path) as img:
                            img_width, img_height = img.size
                    except:
                        continue
                
                for ann in annotations:
                    category_id = ann['category_id'] - 1
                    bbox = ann['bbox']
                    yolo_bbox = coco_to_yolo(bbox, img_width, img_height)
                    yolo_lines.append(f"{category_id} {' '.join(map(str, yolo_bbox))}\n")
                
                # Write YOLO label file
                label_path = output_test_labels / f"fn_{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.writelines(yolo_lines)
        
        print(f"   {'Would add' if dry_run else '✓ Added'} {fn_test_count} FN test images with annotations")
    
    # Step 3: Add False Positives (hard negatives)
    print(f"\n📭 Step 3: Adding False Positives (hard negatives)...")
    
    # Read FP train split CSV
    fp_train_csv = FP_DIR / "train.csv"
    fp_train_count = 0
    fp_train_files = set()
    
    if fp_train_csv.exists():
        import csv
        with open(fp_train_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV has 'original_path' column with relative path like "images/225155048/..."
                img_path = row.get('original_path', '')
                if img_path:
                    img_filename = os.path.basename(img_path)
                    fp_train_files.add(img_filename)
    
    # Read FP test split CSV
    fp_test_csv = FP_DIR / "test.csv"
    fp_test_count = 0
    fp_test_files = set()
    
    if fp_test_csv.exists():
        import csv
        with open(fp_test_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row.get('original_path', '')
                if img_path:
                    img_filename = os.path.basename(img_path)
                    fp_test_files.add(img_filename)
    
    # Process all FP images
    fp_images = list(FP_DIR.glob("images/**/*.jpg")) + list(FP_DIR.glob("images/**/*.png"))
    
    for img_path in fp_images:
        img_filename = img_path.name
        
        if not dry_run:
            # Check if it's in train or test split
            if img_filename in fp_train_files:
                # Add to TRAIN
                dest_img = output_train_images / f"fp_{img_filename}"
                shutil.copy2(img_path, dest_img)
                
                # Create empty label file (blank/negative sample)
                label_path = output_train_labels / f"fp_{img_path.stem}.txt"
                label_path.touch()
                fp_train_count += 1
                
            elif img_filename in fp_test_files:
                # Add to TEST
                dest_img = output_test_images / f"fp_{img_filename}"
                shutil.copy2(img_path, dest_img)
                
                # Create empty label file
                label_path = output_test_labels / f"fp_{img_path.stem}.txt"
                label_path.touch()
                fp_test_count += 1
        else:
            # Dry run - just count
            if img_filename in fp_train_files:
                fp_train_count += 1
            elif img_filename in fp_test_files:
                fp_test_count += 1
    
    print(f"   {'Would add' if dry_run else '✓ Added'} {fp_train_count} FP train images as hard negatives")
    print(f"   {'Would add' if dry_run else '✓ Added'} {fp_test_count} FP test images as hard negatives")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    total_new_train = fn_train_count + fp_train_count
    total_new_test = fn_test_count + fp_test_count
    total_train = len(train_files) + total_new_train if current_train_images.exists() else total_new_train
    total_test = len(test_files) + total_new_test if current_test_images.exists() else total_new_test
    
    print(f"""
TRAINING SET:
  Original:             {len(train_files) if current_train_images.exists() else 0:,} images
  + FN train:           {fn_train_count:,} images (fires model missed)
  + FP train:           {fp_train_count:,} images (false alarms)
  ----------------------------------------
  New training set:     {total_train:,} images (+{(total_new_train/len(train_files)*100) if current_train_images.exists() and len(train_files) > 0 else 0:.1f}%)

TEST SET:
  Original:             {len(test_files) if current_test_images.exists() else 0:,} images
  + FN test:            {fn_test_count:,} images (fires model missed)
  + FP test:            {fp_test_count:,} images (false alarms)
  ----------------------------------------
  New test set:         {total_test:,} images (+{(total_new_test/len(test_files)*100) if current_test_images.exists() and len(test_files) > 0 else 0:.1f}%)
""")
    
    if not dry_run:
        print(f"✅ Dataset merged successfully!")
        print(f"📁 Output: {OUTPUT_DATASET}")
        print(f"\nNext steps:")
        print(f"1. Update dataset.yaml to point to: {OUTPUT_DATASET}")
        print(f"2. Retrain model: python train.py --model s --batch 96")
        print(f"3. Expected improvement: +10-20% mAP from hard examples!")
    else:
        print(f"\n💡 Run with --merge flag to actually perform the merge")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge hard examples with existing dataset')
    parser.add_argument('--check', action='store_true', help='Check dataset structure (default)')
    parser.add_argument('--merge', action='store_true', help='Actually merge the datasets')
    
    args = parser.parse_args()
    
    if args.merge:
        stats = check_structure()
        print("\n⚠️  This will create a new dataset with hard examples merged.")
        response = input("Continue? (yes/no): ")
        if response.lower() == 'yes':
            merge_datasets(dry_run=False)
        else:
            print("❌ Cancelled")
    else:
        # Default: just check
        check_structure()
        print("\n💡 Run with --merge to actually merge the datasets")


if __name__ == '__main__':
    main()
