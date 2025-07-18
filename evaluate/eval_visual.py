#!/usr/bin/env python3
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# "microsoft/beit-base-patch16-224"
model_list = [
    "./output_model/beit-cub-original",
    "./output_model/beit-cub-clr",
    "./output_model/beit-cub-llr",
    "./output_model/beit-cub-slr",

    "./output_model/beit-scar-original",
    "./output_model/beit-scar-slr",
    "./output_model/beit-scar-llr",
    "./output_model/beit-scar-clr",
]

tasks = [
    {
        "task_name": "Classification",
        "dataset_name": "CUB_200",
        "eval_data_path": "./processed_beir/CUB_200/test/eval_data",
        "image_root": "./processed_beir/CUB_200/test/images",
    },
    {
        "task_name": "Snippet-10",
        "dataset_name": "CUB_200",
        "eval_data_path": "./processed_beir/CUB_200/snippet-cub200/snippet-query-10/eval_data",
        "image_root": "./processed_beir/CUB_200/snippet-cub200/snippet-query-10/images",
        "passages_image_root": "./processed_beir/CUB_200/test/images",
    },
    {
        "task_name": "Snippet-20",
        "dataset_name": "CUB_200",
        "eval_data_path": "./processed_beir/CUB_200/snippet-cub200/snippet-query-20/eval_data",
        "image_root": "./processed_beir/CUB_200/snippet-cub200/snippet-query-20/images",
        "passages_image_root": "./processed_beir/CUB_200/test/images",
    },
    {
        "task_name": "Snippet-30",
        "dataset_name": "CUB_200",
        "eval_data_path": "./processed_beir/CUB_200/snippet-cub200/snippet-query-30/eval_data",
        "image_root": "./processed_beir/CUB_200/snippet-cub200/snippet-query-30/images",
        "passages_image_root": "./processed_beir/CUB_200/test/images",
    },
    {
        "task_name": "Snippet-50",
        "dataset_name": "CUB_200",
        "eval_data_path": "./processed_beir/CUB_200/snippet-cub200/snippet-query-50/eval_data",
        "image_root": "./processed_beir/CUB_200/snippet-cub200/snippet-query-50/images",
        "passages_image_root": "./processed_beir/CUB_200/test/images",
    },
    {
        "task_name": "Snippet-80",
        "dataset_name": "CUB_200",
        "eval_data_path": "./processed_beir/CUB_200/snippet-cub200/snippet-query-80/eval_data",
        "image_root": "./processed_beir/CUB_200/snippet-cub200/snippet-query-80/images",
        "passages_image_root": "./processed_beir/CUB_200/test/images",
    },

    {
        "task_name": "Classification",
        "dataset_name": "StanfordCars",
        "eval_data_path": "./processed_beir/StanfordCars/test/eval_data",
        "image_root": "./processed_beir/StanfordCars/test/images",
    },
    {
        "task_name": "Snippet-10",
        "dataset_name": "StanfordCars",
        "eval_data_path": "./processed_beir/StanfordCars/snippet-StanfordCars/snippet-query-10/eval_data",
        "image_root": "./processed_beir/StanfordCars/snippet-StanfordCars/snippet-query-10/images",
        "passages_image_root": "./processed_beir/StanfordCars/test/images",
    },
    {
        "task_name": "Snippet-20",
        "dataset_name": "StanfordCars",
        "eval_data_path": "./processed_beir/StanfordCars/snippet-StanfordCars/snippet-query-20/eval_data",
        "image_root": "./processed_beir/StanfordCars/snippet-StanfordCars/snippet-query-20/images",
        "passages_image_root": "./processed_beir/StanfordCars/test/images",
    },
    {
        "task_name": "Snippet-30",
        "dataset_name": "StanfordCars",
        "eval_data_path": "./processed_beir/StanfordCars/snippet-StanfordCars/snippet-query-30/eval_data",
        "image_root": "./processed_beir/StanfordCars/snippet-StanfordCars/snippet-query-30/images",
        "passages_image_root": "./processed_beir/StanfordCars/test/images",
    },
    {
        "task_name": "Snippet-50",
        "dataset_name": "StanfordCars",
        "eval_data_path": "./processed_beir/StanfordCars/snippet-StanfordCars/snippet-query-50/eval_data",
        "image_root": "./processed_beir/StanfordCars/snippet-StanfordCars/snippet-query-50/images",
        "passages_image_root": "./processed_beir/StanfordCars/test/images",
    },
    {
        "task_name": "Snippet-80",
        "dataset_name": "StanfordCars",
        "eval_data_path": "./processed_beir/StanfordCars/snippet-StanfordCars/snippet-query-80/eval_data",
        "image_root": "./processed_beir/StanfordCars/snippet-StanfordCars/snippet-query-80/images",
        "passages_image_root": "./processed_beir/StanfordCars/test/images",
    }
]


def load_dataset(dataset_path):
    print(f"Loading dataset: {dataset_path}")
    
    with open(os.path.join(dataset_path, "queries", "queries.json"), "r") as f:
        queries_list = json.load(f)
        
    with open(os.path.join(dataset_path, "passages", "passages.json"), "r") as f:
        passages = json.load(f)
    
    with open(os.path.join(dataset_path, "qrels", "qrels.json"), "r") as f:
        qrels_list = json.load(f)
    
    queries = {}
    for q in queries_list:
        for qid, query_text in q.items():
            queries[qid] = query_text
    
    qrels = {}
    for q in qrels_list:
        for qid, rel_docs in q.items():
            qrels[qid] = rel_docs
    
    print(f"Final Load {len(queries)} queries, {len(passages)} passages, {len(qrels)} qrels")
    
    return queries, passages, qrels

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu
        device_count = torch_npu.npu.device_count()
        if device_count > 0:
            return "npu"
    except (ImportError, AttributeError):
        pass
    return "cpu"

class ImageEncoder:    
    def __init__(self, model_name):
        self.device = get_device()
        print(f"Device: {self.device}")
        
        # load model
        print(f"Current Load model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        print("Fished loading model and processor")
    
    def encode_image(self, image_path):
        """Encode an image to a feature vector"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
    
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get Embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 提取特征向量
                if hasattr(outputs, "pooler_output"):
                    # ViT
                    embedding = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    # CLS token
                    embedding = outputs.last_hidden_state[:, 0]
                else:
                    # Average pooling
                    embedding = torch.mean(outputs[0], dim=1)
                
                # Normalize the embedding
                embedding = F.normalize(embedding, p=2, dim=1)
                
                return embedding.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Encoding {image_path} Error: {e}")
            # If there's an error, return a zero vector
            return np.zeros(self.model.config.hidden_size)


def calculate_metrics(qrels, results, k_values=[1, 3, 5, 10]):
    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    mrr = {f"MRR@{k}": 0.0 for k in k_values}
    recall = {f"Recall@{k}": 0.0 for k in k_values}
    
    query_count = 0
    
    for query_id, relevant_docs in qrels.items():
        if query_id not in results:
            continue
        
        query_count += 1
        ranked_docs = [doc_id for doc_id, _ in results[query_id]]
        
        for k in k_values:
            dcg = 0.0
            for i, doc_id in enumerate(ranked_docs[:k]):
                if doc_id in relevant_docs:
                    dcg += relevant_docs[doc_id] / np.log2(i + 2)
            
            idcg = 0.0
            sorted_rel_docs = sorted(relevant_docs.items(), key=lambda x: x[1], reverse=True)
            for i, (_, rel) in enumerate(sorted_rel_docs[:k]):
                idcg += rel / np.log2(i + 2)
            
            if idcg > 0:
                ndcg[f"NDCG@{k}"] += dcg / idcg
        
        for k in k_values:
            for i, doc_id in enumerate(ranked_docs[:k]):
                if doc_id in relevant_docs:
                    mrr[f"MRR@{k}"] += 1.0 / (i + 1)
                    break
        
        for k in k_values:
            retrieved_relevant = set(ranked_docs[:k]).intersection(relevant_docs.keys())
            if relevant_docs:
                recall[f"Recall@{k}"] += len(retrieved_relevant) / len(relevant_docs)
    
    if query_count > 0:
        for k in k_values:
            ndcg[f"NDCG@{k}"] /= query_count
            mrr[f"MRR@{k}"] /= query_count
            recall[f"Recall@{k}"] /= query_count
    
    return ndcg, mrr, recall


def main():    
    for model in model_list:
        for task in tasks:
            task_name = task["task_name"]
            dataset_name = task["dataset_name"]
            dataset_path = task["eval_data_path"]
            image_root = task["image_root"]
            passages_image_root = task.get("passages_image_root", image_root)
            queries, passages, qrels = load_dataset(dataset_path)

            query_ids = list(queries.keys())
            query_paths = [os.path.join(image_root, queries[qid]) for qid in query_ids]

            passage_ids = list(passages.keys())
            if dataset_name == 'StanfordCars' and task_name != "Classification":
                passage_paths = [os.path.join(passages_image_root, "class_"+str(passages[pid]["text"].split("_")[0]), passages[pid]["text"]) for pid in passage_ids]
            else:
                passage_paths = [os.path.join(passages_image_root, passages[pid]["text"]) for pid in passage_ids]

            encoder = ImageEncoder(model)

            print("Encoding Query...")
            query_embeddings = {}
            for i, query_id in enumerate(tqdm(query_ids)):
                path = query_paths[i]
                query_embeddings[query_id] = encoder.encode_image(path)

            print("Encoding Candidate...")
            passage_embeddings = {}
            for i, passage_id in enumerate(tqdm(passage_ids)):
                path = passage_paths[i]
                passage_embeddings[passage_id] = encoder.encode_image(path)

            print("Searching...")
            results = {}
            for query_id, query_embedding in tqdm(query_embeddings.items()):
                # Compute similarity
                scores = []
                for passage_id, passage_embedding in passage_embeddings.items():
                    similarity = np.dot(query_embedding, passage_embedding)
                    scores.append((passage_id, similarity))

                # Sort by similarity
                scores.sort(key=lambda x: x[1], reverse=True)
                results[query_id] = scores

            print("Calculating metrics...")
            k_values = [1, 3, 5, 10]
            ndcg, mrr, recall = calculate_metrics(qrels, results, k_values)

            print(f"\nResult for dataset {dataset_name} with task {task_name}:")
            for k in k_values:
                print(f"NDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}")
                print(f"MRR@{k}: {mrr[f'MRR@{k}']:.4f}")
                print(f"Recall@{k}: {recall[f'Recall@{k}']:.4f}")

            if not os.path.exists("./results"):
                os.makedirs("./results")
            with open("./results/ir_results.csv", 'a+', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # write header if file is empty
                if csvfile.tell() == 0:
                    header = ['dataset_name', 'task_name', 'model']
                    for k in k_values:
                        header.extend([f'mrr@{k}', f'ndcg@{k}', f'recall@{k}'])
                    csv_writer.writerow(header)
                # header = ['dataset_name', 'task_name', 'model']
                # for k in k_values:
                #     header.extend([f'mrr@{k}', f'ndcg@{k}', f'recall@{k}'])
                # csv_writer.writerow(header)
                
                # write data
                row_data = [dataset_name, task_name, model]
                for k in k_values:
                    row_data.extend([
                        f"{mrr[f'MRR@{k}']:.4f}", 
                        f"{ndcg[f'NDCG@{k}']:.4f}", 
                        f"{recall[f'Recall@{k}']:.4f}"
                    ])
                csv_writer.writerow(row_data)

if __name__ == "__main__":
    main()