import os
import json
import random
import pandas as pd
from PIL import Image
from io import BytesIO
import numpy as np
from collections import defaultdict
import uuid

data_base_dir = "./eval_exp_visual/data/stanford_cars/data/train"
parquet_files = [
    os.path.join(data_base_dir, "train-00000-of-00002.parquet"),
    os.path.join(data_base_dir, "train-00001-of-00002.parquet")
]
output_dir = "./eval_exp_visual/processed_beir_for_train/StanfordCars_train"
output_train_file = os.path.join(output_dir, "train.jsonl")
output_images_dir = os.path.join(output_dir, "images")

os.makedirs(output_images_dir, exist_ok=True)

all_images = []
all_labels = []

print("Loading and processing parquet files...")
for parquet_file in parquet_files:
    print(f"Processing {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processed {idx} rows from {parquet_file}")
        
        image_bytes = row['image']['bytes']
        label = row['label']
        
        all_images.append(image_bytes)
        all_labels.append(label)

total_images = len(all_images)
print(f"Total images loaded: {total_images}")
print(f"Number of unique labels: {len(set(all_labels))}")

print("Saving extracted images to disk...")
image_paths = [] 
label_to_image_paths = defaultdict(list) 

for i, (image_bytes, label) in enumerate(zip(all_images, all_labels)):
    if i % 100 == 0:
        print(f"Saving image {i}/{total_images}")
    
    label_dir = os.path.join(output_images_dir, f"class_{label:03d}")
    os.makedirs(label_dir, exist_ok=True)
    
    image_filename = f"image_{i:05d}_{uuid.uuid4().hex[:8]}.jpg"
    image_path = os.path.join(label_dir, image_filename)
    
    try:
        img = Image.open(BytesIO(image_bytes))
        img.save(image_path)
        
        rel_path = os.path.join("images", f"class_{label:03d}", image_filename)
        image_paths.append(rel_path)
        label_to_image_paths[label].append(rel_path)
    except Exception as e:
        print(f"Error saving image {i}: {e}")
        continue

print(f"Saved {len(image_paths)} images to disk")

print("Generating contrastive learning samples...")
train_samples = []
used_query_indices = set()  

for label, paths in label_to_image_paths.items():
    if len(paths) < 2:
        print(f"Skipping label {label} with only {len(paths)} images")
        continue
    
    for query_path in paths:
        if query_path in used_query_indices:
            continue
        
        used_query_indices.add(query_path)
        
        pos_candidates = [path for path in paths if path != query_path]
        
        if not pos_candidates:
            continue
        
        pos_path = random.choice(pos_candidates)
        
        neg_paths = []
        other_labels = [l for l in label_to_image_paths.keys() if l != label]
        
        sampled_labels = random.sample(other_labels, min(15, len(other_labels)))
        for neg_label in sampled_labels:
            if label_to_image_paths[neg_label]: 
                neg_path = random.choice(label_to_image_paths[neg_label])
                neg_paths.append(neg_path)
        
        while len(neg_paths) < 15 and other_labels:
            additional_negs = []
            more_labels = random.sample(other_labels, min(15 - len(neg_paths), len(other_labels)))
            for neg_label in more_labels:
                if label_to_image_paths[neg_label]:
                    available_paths = [p for p in label_to_image_paths[neg_label] if p not in neg_paths]
                    if not available_paths:
                        available_paths = label_to_image_paths[neg_label]
                    
                    neg_path = random.choice(available_paths)
                    additional_negs.append(neg_path)
            
            if not additional_negs:
                break  # Can't find more negatives
            
            neg_paths.extend(additional_negs)
        
        neg_paths = neg_paths[:15]
        
        sample = {
            "query": query_path,
            "pos": [pos_path],
            "neg": neg_paths
        }
        
        train_samples.append(sample)

print(f"Generated {len(train_samples)} training samples")

print(f"Writing training samples to {output_train_file}...")
with open(output_train_file, 'w') as f:
    for sample in train_samples:
        f.write(json.dumps(sample) + '\n')

print(f"Training samples written to {output_train_file}")

if train_samples:
    print("\nExample training sample:")
    print(json.dumps(train_samples[0], indent=2))
