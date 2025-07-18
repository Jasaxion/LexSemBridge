import json
import torch
import numpy as np
import argparse
import os
import csv
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import fcntl
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.retrieval.evaluation import EvaluateRetrieval

def get_evaluation_structure():
    base_dir = "./LexSemBridge_eval"
    return {
        "PartOfPassageSearch": {
            "NQ": {
                "corpus": f"{base_dir}/eval_data/PartOfPassageSearch/NQ/nq_pop_text5000.json",
                "queries": {
                    "pop16": f"{base_dir}/eval_data/PartOfPassageSearch/NQ/nq_pop_16_queries.json",
                    "pop32": f"{base_dir}/eval_data/PartOfPassageSearch/NQ/nq_pop_32_queries.json",
                    "pop64": f"{base_dir}/eval_data/PartOfPassageSearch/NQ/nq_pop_64_queries.json",
                    "pop128": f"{base_dir}/eval_data/PartOfPassageSearch/NQ/nq_pop_128_queries.json",
                    "pop256": f"{base_dir}/eval_data/PartOfPassageSearch/NQ/nq_pop_256_queries.json",
                },
                "qrels": f"{base_dir}/eval_data/PartOfPassageSearch/NQ/nq_pop_qrels.json"
            },
            "FEVER": {
                "corpus": f"{base_dir}/eval_data/PartOfPassageSearch/FEVER/fever_pop_text.json",
                "queries": {
                    "pop16": f"{base_dir}/eval_data/PartOfPassageSearch/FEVER/fever_pop_16_queries.json",
                    "pop32": f"{base_dir}/eval_data/PartOfPassageSearch/FEVER/fever_pop_32_queries.json",
                    "pop64": f"{base_dir}/eval_data/PartOfPassageSearch/FEVER/fever_pop_64_queries.json",
                    "pop128": f"{base_dir}/eval_data/PartOfPassageSearch/FEVER/fever_pop_128_queries.json",
                    "pop256": f"{base_dir}/eval_data/PartOfPassageSearch/FEVER/fever_pop_256_queries.json",
                },
                "qrels": f"{base_dir}/eval_data/PartOfPassageSearch/FEVER/fever_pop_qrels.json"
            },
            "HotpotQA": {
                "corpus": f"{base_dir}/eval_data/PartOfPassageSearch/HotpotQA/hotpotqa_pop_text5000.json",
                "queries": {
                    "pop16": f"{base_dir}/eval_data/PartOfPassageSearch/HotpotQA/hotpotqa_pop_16_queries.json",
                    "pop32": f"{base_dir}/eval_data/PartOfPassageSearch/HotpotQA/hotpotqa_pop_32_queries.json",
                    "pop64": f"{base_dir}/eval_data/PartOfPassageSearch/HotpotQA/hotpotqa_pop_64_queries.json",
                    "pop128": f"{base_dir}/eval_data/PartOfPassageSearch/HotpotQA/hotpotqa_pop_128_queries.json",
                    "pop256": f"{base_dir}/eval_data/PartOfPassageSearch/HotpotQA/hotpotqa_pop_256_queries.json",
                },
                "qrels": f"{base_dir}/eval_data/PartOfPassageSearch/HotpotQA/hotpotqa_pop_qrels.json"
            }
        },
        "KeywordsSearch": {
            "hotpotqa_keywords":{
                "corpus": f"{base_dir}/eval_data/hotpotqa5000/keyword/hotpotqa_pop_text5000.json",
                "queries": f"{base_dir}/eval_data/hotpotqa5000/keyword/hotpotqa_keyword_queries.json",
                "qrels": f"{base_dir}/eval_data/hotpotqa5000/keyword/hotpotqa_keyword_qrels.json"
            },
            "fever_keywords":{
                "corpus": f"{base_dir}/eval_data/fever1500/keyword/fever_pop_text.json",
                "queries": f"{base_dir}/eval_data/fever1500/keyword/fever_keyword_queries.json",
                "qrels": f"{base_dir}/eval_data/fever1500/keyword/fever_keyword_qrels.json"
            },
            "nq_keywords":{
                "corpus": f"{base_dir}/eval_data/nq5000/keyword/nq_keyword_text5000.json",
                "queries": f"{base_dir}/eval_data/nq5000/keyword/nq_keyword_queries.json",
                "qrels": f"{base_dir}/eval_data/nq5000/keyword/nq_keyword_qrels.json"
            }
        },
        "SemanticSearch": {
            "NQ": {
                "corpus": f"{base_dir}/eval_data/SemanticSearch/NQ/nq_text5000.json",
                "queries": f"{base_dir}/eval_data/SemanticSearch/NQ/nq_queries5000.json",
                "qrels": f"{base_dir}/eval_data/SemanticSearch/NQ/nq_qrels5000.json"
            },
            "FEVER": {
                "corpus": f"{base_dir}/eval_data/SemanticSearch/FEVER/fever_text1500.json",
                "queries": f"{base_dir}/eval_data/SemanticSearch/FEVER/fever_queries.json",
                "qrels": f"{base_dir}/eval_data/SemanticSearch/FEVER/fever_qrels.json"
            },
            "HotpotQA": {
                "corpus": f"{base_dir}/eval_data/SemanticSearch/HotpotQA/hotpotqa_text5000.json",
                "queries": f"{base_dir}/eval_data/SemanticSearch/HotpotQA/hotpotqa_queries5000.json",
                "qrels": f"{base_dir}/eval_data/SemanticSearch/HotpotQA/hotpotqa_qrels5000.json"
            }
        }
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluation configuration for SPLADE retrieval model')
    
    parser.add_argument('--tasks', nargs='+', choices=['PartOfPassageSearch', 'KeywordsSearch', 'SemanticSearch'],
                      help='Specify tasks to evaluate')
    parser.add_argument('--datasets', nargs='+',
                      help='Specify datasets to evaluate')
    parser.add_argument('--pop_sizes', nargs='+', choices=['pop16', 'pop32', 'pop64', 'pop128', 'pop256'],
                      help='Specify POP sizes to evaluate')
    parser.add_argument('--gpu', type=str, default="0",
                      help='Specify GPU device number (default: 0)')
    parser.add_argument('--output', type=str, default='./results_splade.csv',
                      help='Specify output file path (default: ./results_splade.csv)')
    parser.add_argument('--model', type=str, default="naver/splade_v2_max",
                      help='Specify SPLADE model path')
    
    return parser.parse_args()

def filter_evaluation_structure(
    eval_structure: Dict,
    selected_tasks: Optional[List[str]] = None,
    selected_datasets: Optional[List[str]] = None,
    selected_pop_sizes: Optional[List[str]] = None
) -> Dict:
    """Filter the evaluation structure based on selected tasks, datasets, and POP sizes."""
    filtered_structure = {}
    
    # Filter tasks
    for task, datasets in eval_structure.items():
        if selected_tasks and task not in selected_tasks:
            continue
            
        filtered_structure[task] = {}
        
        # Filter datasets
        for dataset_name, paths in datasets.items():
            if selected_datasets and dataset_name not in selected_datasets:
                continue
                
            # Handle POP sizes for PartOfPassageSearch
            if task == "PartOfPassageSearch" and isinstance(paths["queries"], dict):
                if selected_pop_sizes:
                    filtered_queries = {size: path for size, path in paths["queries"].items()
                                     if size in selected_pop_sizes}
                    if not filtered_queries:
                        continue
                    paths = paths.copy()
                    paths["queries"] = filtered_queries
                    
            filtered_structure[task][dataset_name] = paths
            
    return filtered_structure

class SpladeModel:
    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def encode_queries(self, queries: List[str], batch_size: int = 32, **kwargs):
        return self.encode_text(queries, batch_size=batch_size)
        
    def encode_corpus(self, corpus: List[dict], batch_size: int = 32, **kwargs):
        # Extract text from corpus documents
        texts = [doc.get("text", "") for doc in corpus]
        return self.encode_text(texts, batch_size=batch_size)

    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into SPLADE sparse vectors.
        
        Args:
            texts: List[str], list of text strings to encode
            batch_size: int, batch size for encoding
            
        Returns:
            np.ndarray of sparse vectors
        """
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch_texts,
                                  padding=True,
                                  truncation=True,
                                  max_length=512,
                                  return_tensors="pt").to(self.device)
            
            # Get SPLADE sparse vectors
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Apply SPLADE pooling (log(1 + ReLU(MLM))) 
                logits = outputs.logits
                sparse_vec = torch.log(1 + torch.relu(logits))
                # Max pooling over sequence length
                sparse_vec = torch.max(sparse_vec, dim=1)[0]
                embeddings.append(sparse_vec.cpu().numpy())

        return np.vstack(embeddings)

class SpladeEvaluator:
    def __init__(self, model_path: str, batch_size: int = 32):
        self.model = SpladeModel(model_path)
        self.dense_retriever = DenseRetrievalExactSearch(
            self.model,
            batch_size=batch_size
        )
        self.retriever = EvaluateRetrieval(self.dense_retriever, score_function="dot")
        
    def load_and_transform_data(self, queries_path: str, corpus_path: str, qrels_path: str):
        # Load data
        with open(queries_path) as f:
            queries = json.load(f)
        with open(corpus_path) as f:
            corpus = json.load(f)
        with open(qrels_path) as f:
            qrels = json.load(f)
            
        # Transform queries - handle both list and dict formats
        beir_queries = {}
        if isinstance(queries, list):
            # Handle list of dictionaries
            for query_dict in queries:
                for qid, query_text in query_dict.items():
                    beir_queries[qid] = query_text
        else:
            # Handle direct dictionary
            beir_queries = queries
            
        # Transform corpus - handle both list and dict formats    
        beir_corpus = {}
        if isinstance(corpus, list):
            # Handle list of dictionaries
            for doc_dict in corpus:
                for doc_id, doc in doc_dict.items():
                    text = doc["text"] if isinstance(doc, dict) else doc
                    beir_corpus[doc_id] = {
                        "text": text,
                        "title": ""
                    }
        else:
            # Handle direct dictionary
            for doc_id, doc in corpus.items():
                text = doc["text"] if isinstance(doc, dict) else doc
                beir_corpus[doc_id] = {
                    "text": text,
                    "title": ""
                }
                
        # Transform qrels - handle both list and dict formats
        beir_qrels = {}
        if isinstance(qrels, list):
            # Handle list of dictionaries
            for qrel_dict in qrels:
                for query_id, relevance in qrel_dict.items():
                    beir_qrels[query_id] = relevance
        else:
            # Handle direct dictionary
            beir_qrels = qrels
            
        return beir_queries, beir_corpus, beir_qrels
    
    def evaluate(self, queries: Dict, corpus: Dict, qrels: Dict):
        """
        Evaluate the retrieval performance using BEIR's evaluation metrics.
        
        Args:
            queries: Dict of query_id -> query_text
            corpus: Dict of doc_id -> {"text": text, "title": title}
            qrels: Dict of query_id -> {doc_id: relevance_score}
        """
        # Convert corpus to format expected by BEIR's dense retriever
        beir_corpus = {}
        for doc_id, doc in corpus.items():
            beir_corpus[doc_id] = {
                "text": doc["text"],
                "title": doc.get("title", "")
            }
            
        # Convert queries to format expected by BEIR's dense retriever
        beir_queries = {}
        for query_id, query_text in queries.items():
            beir_queries[query_id] = query_text
        
        # Use BEIR's retriever to get results
        results = self.retriever.retrieve(beir_corpus, beir_queries)
        
        # Evaluate using standard BEIR metrics
        ndcg, _map, recall, precision = self.retriever.evaluate(qrels, results, [1, 10])
        
        metrics = {
            "NDCG@1": ndcg["NDCG@1"],
            "NDCG@10": ndcg["NDCG@10"],
            "Recall@1": recall["Recall@1"],
            "Recall@10": recall["Recall@10"],
            "MRR@1": ndcg["NDCG@1"],  # Using NDCG@1 as approximation for MRR@1
            "MRR@10": np.mean([1.0/rank if rank <= 10 else 0.0 
                              for qid, doc_scores in results.items()
                              for rank, (doc_id, _) in enumerate(sorted(doc_scores.items(), 
                                                                      key=lambda x: x[1], 
                                                                      reverse=True), 1)
                              if doc_id in qrels.get(qid, {})])
        }
        
        return metrics, results

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Configure GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Get evaluation structure and filter based on arguments
    eval_structure = get_evaluation_structure()
    filtered_structure = filter_evaluation_structure(
        eval_structure,
        selected_tasks=args.tasks,
        selected_datasets=args.datasets,
        selected_pop_sizes=args.pop_sizes
    )
    
    # Initialize evaluator
    evaluator = SpladeEvaluator(args.model)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Process each dataset
    for task, datasets in filtered_structure.items():
        for dataset_name, paths in datasets.items():
            # Handle different query structures
            if isinstance(paths["queries"], dict):
                queries_tasks = paths["queries"]
            else:
                queries_tasks = {"default": paths["queries"]}

            for subtask_name, query_path in queries_tasks.items():
                print(f"\nProcessing {task}/{dataset_name}/{subtask_name}")
                
                # Load and transform data to BEIR format
                queries, corpus, qrels = evaluator.load_and_transform_data(
                    queries_path=query_path,
                    corpus_path=paths["corpus"],
                    qrels_path=paths["qrels"]
                )
                
                # Evaluate
                metrics, results = evaluator.evaluate(queries, corpus, qrels)
                
                # Print results
                print("=" * 50)
                print(f"Task: {task}")
                print(f"Dataset: {dataset_name}")
                print(f"Subtask: {subtask_name}")
                print(f"NDCG@1: {metrics['NDCG@1']:.4f} | NDCG@10: {metrics['NDCG@10']:.4f}")
                print(f"MRR@1: {metrics['MRR@1']:.4f} | MRR@10: {metrics['MRR@10']:.4f}")
                print(f"Recall@1: {metrics['Recall@1']:.4f} | Recall@10: {metrics['Recall@10']:.4f}")
                print("=" * 50)
                
                # Save results with file locking
                with open(args.output, mode='a', newline='') as file:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                    try:
                        writer = csv.writer(file)
                        
                        # Write header if file is empty
                        if file.tell() == 0:
                            writer.writerow([
                                "Task", "Dataset", "Subtask",
                                "NDCG@1", "NDCG@10",
                                "MRR@1", "MRR@10",
                                "Recall@1", "Recall@10"
                            ])
                        
                        writer.writerow([
                            task,
                            dataset_name,
                            subtask_name,
                            metrics['NDCG@1'],
                            metrics['NDCG@10'],
                            metrics['MRR@1'],
                            metrics['MRR@10'],
                            metrics['Recall@1'],
                            metrics['Recall@10']
                        ])
                        
                        file.flush()
                    finally:
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)

if __name__ == "__main__":
    main()