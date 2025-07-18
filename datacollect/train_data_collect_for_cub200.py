import os
import json
import random
import shutil
from collections import defaultdict

base_dir = "./eval_exp_visual/data/cub200/CUB_200_2011"
images_file = os.path.join(base_dir, "images.txt")
labels_file = os.path.join(base_dir, "image_class_labels.txt")
split_file = os.path.join(base_dir, "train_test_split.txt")
source_images_dir = os.path.join(base_dir, "images")
output_dir = "./eval_exp_visual/processed_beir_for_train/CUB_200_train"
output_train_file = os.path.join(output_dir, "train.jsonl")
output_images_dir = os.path.join(output_dir, "images")
os.makedirs(output_images_dir, exist_ok=True)

train_ids = set()
with open(split_file, 'r') as f:
    for line in f:
        img_id, is_train = map(int, line.strip().split())
        if is_train == 1:
            train_ids.add(img_id)

print(f"Number of training images: {len(train_ids)}")

image_paths = {}
with open(images_file, 'r') as f:
    for line in f:
        img_id, img_path = line.strip().split()
        img_id = int(img_id)
        if img_id in train_ids:
            image_paths[img_id] = os.path.join("images", img_path)

image_labels = {}
with open(labels_file, 'r') as f:
    for line in f:
        img_id, class_id = map(int, line.strip().split())
        img_id = int(img_id)
        if img_id in train_ids:
            image_labels[img_id] = class_id

train_class_to_images = defaultdict(list)
for img_id in train_ids:
    if img_id in image_labels:
        class_id = image_labels[img_id]
        train_class_to_images[class_id].append(img_id)

train_samples = []
used_query_ids = set()

for class_id, img_ids in train_class_to_images.items():
    if len(img_ids) < 2:
        continue
    
    for query_id in img_ids:
        if query_id in used_query_ids:
            continue
        
        used_query_ids.add(query_id)
        
        pos_candidates = [id for id in img_ids if id != query_id]
        
        if not pos_candidates:
            continue
            
        pos_id = random.choice(pos_candidates)
        
        neg_ids = []
        other_classes = [c for c in train_class_to_images.keys() if c != class_id]
        
        for neg_class in random.sample(other_classes, min(15, len(other_classes))):
            if train_class_to_images[neg_class]:
                neg_id = random.choice(train_class_to_images[neg_class])
                neg_ids.append(neg_id)
        
        while len(neg_ids) < 15 and other_classes:
            additional_negs = []
            for neg_class in random.sample(other_classes, min(15 - len(neg_ids), len(other_classes))):
                if train_class_to_images[neg_class]:
                    available_imgs = [id for id in train_class_to_images[neg_class] if id not in neg_ids]
                    if not available_imgs:
                        available_imgs = train_class_to_images[neg_class]
                    
                    neg_id = random.choice(available_imgs)
                    additional_negs.append(neg_id)
            
            if not additional_negs:
                break
                
            neg_ids.extend(additional_negs)
        
        neg_ids = neg_ids[:15]
        
        sample = {
            "query": image_paths[query_id],
            "pos": [image_paths[pos_id]], 
            "neg": [image_paths[neg_id] for neg_id in neg_ids]
        }
        
        train_samples.append(sample)

print(f"Generated {len(train_samples)} training samples")

copied_images = set()
all_image_paths = set()

for sample in train_samples:
    all_image_paths.add(sample["query"])
    all_image_paths.update(sample["pos"])
    all_image_paths.update(sample["neg"])

for rel_img_path in all_image_paths:
    orig_img_path = rel_img_path.replace("images/", "", 1)
    source_path = os.path.join(source_images_dir, orig_img_path)
    dest_path = os.path.join(output_images_dir, orig_img_path)
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(source_path):
        shutil.copy2(source_path, dest_path)
        copied_images.add(rel_img_path)
    else:
        print(f"Warning: Source image not found: {source_path}")

print(f"Copied {len(copied_images)} image files to {output_images_dir}")

with open(output_train_file, 'w') as f:
    for sample in train_samples:
        f.write(json.dumps(sample) + '\n')

print(f"Training samples written to {output_train_file}")

if train_samples:
    print("\nExample training sample:")
    print(json.dumps(train_samples[0], indent=2))