import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import datasets
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, PreTrainedTokenizer

from .arguments import DataArguments


class TrainDatasetForVisualEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            image_processor: Optional[AutoImageProcessor] = None
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                if file.endswith('.jsonl') or file.endswith('.json'):
                    temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                        split='train')
                    if len(temp_dataset) > args.max_example_num_per_dataset:
                        temp_dataset = temp_dataset.select(
                            random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                    train_datasets.append(temp_dataset)
            
            if train_datasets:
                self.dataset = datasets.concatenate_datasets(train_datasets)
            else:
                raise ValueError(f"No .json or .jsonl files found in {args.train_data}")
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.image_processor = image_processor
        self.args = args
        self.total_len = len(self.dataset)
        self.image_root_dir = args.image_root_dir

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        # Get the query and passage paths
        query = self.dataset[item]['query']
        
        # For image paths, we need the actual paths rather than applying instructions
        query_path = os.path.join(self.image_root_dir, query)

        passages = []
        
        # Get a positive example
        assert isinstance(self.dataset[item]['pos'], list)
        pos = random.choice(self.dataset[item]['pos'])
        pos_path = os.path.join(self.image_root_dir, pos)
        passages.append(pos_path)

        # Get negative examples
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        
        # Convert negative examples to full paths
        for neg in negs:
            neg_path = os.path.join(self.image_root_dir, neg)
            passages.append(neg_path)

        return query_path, passages


@dataclass
class VisualEmbedCollator:
    """
    Collator that processes visual data from image paths to input features for the visual model.
    """
    image_processor: AutoImageProcessor
    query_max_len: int = 224
    passage_max_len: int = 224

    def __call__(self, features):
        query_paths = [f[0] for f in features]
        passage_paths = [f[1] for f in features]

        if isinstance(passage_paths[0], list):
            passage_paths = sum(passage_paths, [])

        # Load and process query images
        query_images = []
        for path in query_paths:
            try:
                img = Image.open(path).convert('RGB')
                query_images.append(img)
            except Exception as e:
                print(f"Error loading query image {path}: {e}")
                # Use a blank image as fallback
                query_images.append(Image.new('RGB', (self.query_max_len, self.query_max_len), color='black'))

        # Load and process passage images
        passage_images = []
        for path in passage_paths:
            try:
                img = Image.open(path).convert('RGB')
                passage_images.append(img)
            except Exception as e:
                print(f"Error loading passage image {path}: {e}")
                # Use a blank image as fallback
                passage_images.append(Image.new('RGB', (self.passage_max_len, self.passage_max_len), color='black'))

        # Process images using the image processor
        q_features = self.image_processor(
            images=query_images,
            return_tensors="pt",
            size={"height": self.query_max_len, "width": self.query_max_len}
        )
        
        p_features = self.image_processor(
            images=passage_images,
            return_tensors="pt",
            size={"height": self.passage_max_len, "width": self.passage_max_len}
        )
        
        # Create attention masks if they're not already provided by the processor
        if 'attention_mask' not in q_features:
            # Create default attention masks (all 1s)
            batch_size = q_features['pixel_values'].size(0)
            q_features['attention_mask'] = torch.ones((batch_size, 196), dtype=torch.long)  # 196 patches for 14x14 ViT
            
        if 'attention_mask' not in p_features:
            batch_size = p_features['pixel_values'].size(0)
            p_features['attention_mask'] = torch.ones((batch_size, 196), dtype=torch.long)

        return {"query": q_features, "passage": p_features}