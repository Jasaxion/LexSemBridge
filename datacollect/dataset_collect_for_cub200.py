import os
import json
import shutil
import random
from collections import defaultdict

def process_cub200(cub_dir, output_dir):
    """
    Process CUB200 dataset and convert to BEIR format
    
    Args:
        cub_dir: Root directory of CUB200 dataset
        output_dir: Output directory for converted dataset
    """
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    for dir_path in [train_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, "eval_data", "queries"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "eval_data", "passages"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "eval_data", "qrels"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "imgs"), exist_ok=True)
    
    classes = {}
    with open(os.path.join(cub_dir, "classes.txt"), "r") as f:
        for line in f:
            class_id, class_name = line.strip().split(" ", 1)
            classes[int(class_id)] = class_name
    
    image_labels = {}
    with open(os.path.join(cub_dir, "image_class_labels.txt"), "r") as f:
        for line in f:
            image_id, label_id = map(int, line.strip().split())
            image_labels[image_id] = label_id
    
    image_paths = {}
    with open(os.path.join(cub_dir, "images.txt"), "r") as f:
        for line in f:
            image_id, image_path = line.strip().split(" ", 1)
            image_paths[int(image_id)] = image_path
    
    train_test_split = {}
    with open(os.path.join(cub_dir, "train_test_split.txt"), "r") as f:
        for line in f:
            image_id, is_train = map(int, line.strip().split())
            train_test_split[image_id] = is_train
    
    train_images_by_class = defaultdict(list)
    test_images_by_class = defaultdict(list)
    
    for image_id, image_path in image_paths.items():
        class_id = image_labels[image_id]
        if train_test_split[image_id] == 1:  # Training set
            train_images_by_class[class_id].append((image_id, image_path))
        else:  # Test set
            test_images_by_class[class_id].append((image_id, image_path))
    
    for split_name, images_by_class, split_dir in [
        ("train", train_images_by_class, train_dir),
        ("test", test_images_by_class, test_dir)
    ]:
        queries = {}
        passages = {}
        qrels = []
        
        for class_id, images in images_by_class.items():
            # Shuffle images to randomize selection
            random.shuffle(images)
            
            mid = len(images) // 2
            query_images = images[:mid]
            passage_images = images[mid:]
            
            for img_id, img_path in query_images:
                # Store just the image filename in queries, not the full path
                img_filename = f"{img_id}.jpg"
                queries[str(img_id)] = img_filename
                
                # Create qrels: match this query with all passages of same class
                related_passages = {}
                for p_id, _ in passage_images:
                    related_passages[str(p_id)] = 1
                
                qrels.append({str(img_id): related_passages})
                
                # Copy the query image file to imgs directory
                src_img_path = os.path.join(cub_dir, "images", img_path)
                dst_img_path = os.path.join(split_dir, "imgs", img_filename)
                
                # Copy the image file
                try:
                    shutil.copy2(src_img_path, dst_img_path)
                except Exception as e:
                    print(f"Error copying {src_img_path}: {e}")
            
            # Create passages for this class
            for img_id, img_path in passage_images:
                # Store just the image filename in passages, not the full path
                img_filename = f"{img_id}.jpg"
                passages[str(img_id)] = {"text": img_filename}
                
                # Copy the passage image file to imgs directory
                src_img_path = os.path.join(cub_dir, "images", img_path)
                dst_img_path = os.path.join(split_dir, "imgs", img_filename)
                
                # Copy the image file
                try:
                    shutil.copy2(src_img_path, dst_img_path)
                except Exception as e:
                    print(f"Error copying {src_img_path}: {e}")
        
        # Convert queries dict to list format as required by BEIR
        queries_list = []
        for query_id, query_path in queries.items():
            queries_list.append({query_id: query_path})
        
        # Write JSON files
        with open(os.path.join(split_dir, "eval_data", "queries", "queries.json"), "w") as f:
            json.dump(queries_list, f, indent=4)
        
        with open(os.path.join(split_dir, "eval_data", "passages", "passages.json"), "w") as f:
            json.dump(passages, f, indent=4)
        
        with open(os.path.join(split_dir, "eval_data", "qrels", "qrels.json"), "w") as f:
            json.dump(qrels, f, indent=4)
        
        print(f"Processed {split_name} dataset:")
        print(f"  - Queries: {len(queries_list)}")
        print(f"  - Passages: {len(passages)}")
        print(f"  - QRels: {len(qrels)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CUB200 dataset to BEIR format")
    parser.add_argument("--cub_dir", required=False, default="./eval_exp_visual/data/cub200/CUB_200_2011", help="Path to CUB_200_2011 directory")
    parser.add_argument("--output_dir", required=False, default="./eval_exp_visual/processed_beir/CUB_200", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Process the dataset
    process_cub200(args.cub_dir, args.output_dir)
    print("Conversion completed successfully!")