#!/usr/bin/env python3
"""
Stanford Cars dataset converted to BEIR format
Convert the Stanford Cars dataset from parquet format to the standard BEIR format for image retrieval evaluation.
Usage:
    python dataset_collect_for_scar.py --input_dir ./eval_exp_visual/data/stanford_cars/data/test --output_dir ./eval_exp_visual/processed_beir/StanfordCars/test --image_output_dir ./eval_exp_visual/processed_beir/StanfordCars/test/images
"""

import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from PIL import Image
import io
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Convert the Stanford Cars dataset to BEIR format")
    parser.add_argument("--input_dir", type=str, required=True,
                      help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="BEIR format data output directory")
    parser.add_argument("--image_output_dir", type=str, required=True,
                      help="Output image file directory")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed, used for dataset splitting")
    return parser.parse_args()

def load_parquet_data(input_dir):
    print("loading...")
    
    dataframes = []
    parquet_files = sorted(list(Path(input_dir).glob("*.parquet")))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}.")
    
    for file_path in parquet_files:
        print(f"Load {file_path}...")
        df = pd.read_parquet(file_path)
        dataframes.append(df)
    
    if len(dataframes) > 1:
        df = pd.concat(dataframes, ignore_index=True)
    else:
        df = dataframes[0]
    
    print(f"Loaded {len(df)} pieces of data")
    return df

def extract_images_and_labels(df, image_output_dir):
    print("Extracting images and labels...")
    
    os.makedirs(image_output_dir, exist_ok=True)
    
    unique_labels = df['label'].unique()
    for label in unique_labels:
        os.makedirs(os.path.join(image_output_dir, f"class_{label}"), exist_ok=True)
    
    image_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        label = row['label']
        image_bytes = row['image']['bytes']
        
        image_id = f"{label}_{idx}"
        image_filename = f"{image_id}.jpg"
        image_path = os.path.join(f"class_{label}", image_filename)
        full_image_path = os.path.join(image_output_dir, image_path)
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.save(full_image_path)
            
            image_data.append({
                "id": image_id,
                "path": image_path,
                "label": int(label)
            })
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
    
    print(f"Extracted {len(image_data)} images")
    return image_data

def split_by_class(image_data, seed=42):
    print("Splitting the dataset by category...")
    
    random.seed(seed)
    
    images_by_class = {}
    for item in image_data:
        label = item["label"]
        if label not in images_by_class:
            images_by_class[label] = []
        images_by_class[label].append(item)
    
    queries = []
    passages = {}
    qrels = []
    
    for label, items in images_by_class.items():
        random.shuffle(items)
        
        split_idx = len(items) // 2
        query_items = items[:split_idx]
        passage_items = items[split_idx:]
        
        for item in query_items:
            query_entry = {item["id"]: item["path"]}
            queries.append(query_entry)
            
            qrel_entry = {}
            relevant_passages = {}
            
            for p_item in passage_items:
                relevant_passages[p_item["id"]] = 1  # 标记为相关
            
            qrel_entry[item["id"]] = relevant_passages
            qrels.append(qrel_entry)
        
        for item in passage_items:
            passages[item["id"]] = {"text": item["path"]}
    
    print(f"Created {len(queries)} queries, {len(passages)} passages, {len(qrels)} qrels entries")
    return queries, passages, qrels

def create_beir_format(queries, passages, qrels, output_dir):
    print("Creating BEIR format dataset...")
    
    eval_data_dir = os.path.join(output_dir, "eval_data")
    os.makedirs(os.path.join(eval_data_dir, "queries"), exist_ok=True)
    os.makedirs(os.path.join(eval_data_dir, "passages"), exist_ok=True)
    os.makedirs(os.path.join(eval_data_dir, "qrels"), exist_ok=True)
    
    with open(os.path.join(eval_data_dir, "queries", "queries.json"), "w") as f:
        json.dump(queries, f, indent=2)
    
    with open(os.path.join(eval_data_dir, "passages", "passages.json"), "w") as f:
        json.dump(passages, f, indent=2)
    
    with open(os.path.join(eval_data_dir, "qrels", "qrels.json"), "w") as f:
        json.dump(qrels, f, indent=2)
    
    print(f"BEIR format dataset has been saved to {eval_data_dir}")

def main():
    args = parse_args()
    
    df = load_parquet_data(args.input_dir)
    
    image_data = extract_images_and_labels(df, args.image_output_dir)
    
    queries, passages, qrels = split_by_class(image_data, args.seed)
    
    create_beir_format(queries, passages, qrels, args.output_dir)
    
    print("Conversion completed!")
    print(f"Image file is located at:{args.image_output_dir}")
    print(f"BEIR format data is located at: {os.path.join(args.output_dir, 'eval_data')}")

if __name__ == "__main__":
    main()