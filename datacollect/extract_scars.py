#!/usr/bin/env python3
import os
import shutil
import glob
from pathlib import Path

def copy_image_files(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    
    copied_count = 0
    
    for ext in image_extensions:
        for img_file in Path(source_dir).glob(f'**/{ext}'):
            target_file = os.path.join(target_dir, img_file.name)
            
            shutil.copy2(img_file, target_file)
            copied_count += 1
    
    return copied_count

if __name__ == "__main__":
    SOURCE_DIR = "./eval_exp_visual/processed_beir/StanfordCars/test/images"
    TARGET_DIR = "./eval_exp_visual/processed_beir/StanfordCars/test/snippet_test/tmp_images"
    
    copied_files = copy_image_files(SOURCE_DIR, TARGET_DIR)
    
    print(f"All image files have been copied to {TARGET_DIR}")
    print(f"Copied {copied_files} files")