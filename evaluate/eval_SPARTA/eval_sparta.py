import json
import logging
import argparse
import os
from typing import Dict, List, Optional, Union
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseSearch
import fcntl,csv
class CustomDataLoader:
    """Custom data loader for the evaluation structure"""
    def __init__(self, corpus_path: str, queries_path: Union[str, Dict[str, str]], qrels_path: str):
        self.corpus_path = corpus_path
        self.queries_path = queries_path 
        self.qrels_path = qrels_path

    def load_queries(self, query_path: str) -> Dict:
        """Load queries from a file path"""
        with open(query_path, 'r') as f:
            queries = json.load(f)
        # Convert to dict if it's a list
        return {k: v for d in queries for k, v in d.items()} if isinstance(queries, list) else queries

    def load(self):
        # Load corpus
        with open(self.corpus_path, 'r') as f:
            corpus = json.load(f)
        # Convert to BEIR format if needed
        corpus = {
            doc_id: {
                "title": "",  # Add empty title
                "text": doc["text"] if isinstance(doc, dict) else doc
            }
            for doc_id, doc in corpus.items()
        }

        # Handle queries loading based on type
        if isinstance(self.queries_path, dict):
            # For PartOfPassageSearch, return a dict of query sets
            queries = {size: self.load_queries(path) 
                      for size, path in self.queries_path.items()}
        else:
            # For other tasks, load single query file
            queries = self.load_queries(self.queries_path)

        # Load qrels
        with open(self.qrels_path, 'r') as f:
            qrels = json.load(f)
        # Convert to dict if it's a list
        qrels = {k: v for d in qrels for k, v in d.items()} if isinstance(qrels, list) else qrels
        
        # Convert qrels to BEIR format
        beir_qrels = {}
        for qid, relevant_docs in qrels.items():
            beir_qrels[qid] = {doc_id: 1 for doc_id in relevant_docs.keys()}

        return corpus, queries, beir_qrels

def get_evaluation_structure():
    base_dir = "./LexSemBridge_eval"
    # Your existing evaluation structure code here
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

def filter_evaluation_structure(
    eval_structure: Dict,
    selected_tasks: Optional[List[str]] = None,
    selected_datasets: Optional[List[str]] = None,
) -> Dict:
    """Filter the evaluation structure based on selected tasks and datasets."""
    filtered_structure = {}
    
    for task, datasets in eval_structure.items():
        if selected_tasks and task not in selected_tasks:
            continue
            
        filtered_structure[task] = {}
        
        for dataset_name, paths in datasets.items():
            if selected_datasets and dataset_name not in selected_datasets:
                continue
            filtered_structure[task][dataset_name] = paths
            
    return filtered_structure

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluation configuration for SPARTA retrieval model')
    
    parser.add_argument('--tasks', nargs='+', choices=['SemanticSearch', 'KeywordsSearch', 'PartOfPassageSearch'],
                      help='Specify tasks to evaluate')
    parser.add_argument('--datasets', nargs='+',
                      help='Specify datasets to evaluate')
    parser.add_argument('--output', type=str, default='./results_sparta.csv',
                      help='Specify output file path')
    parser.add_argument('--model', type=str, default="BeIR/sparta-msmarco-distilbert-base-v1",
                      help='Specify SPARTA model path')
    
    return parser.parse_args()

def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
    
    # Parse arguments
    args = parse_arguments()
    
    # Get evaluation structure and filter based on arguments
    eval_structure = get_evaluation_structure()
    filtered_structure = filter_evaluation_structure(
        eval_structure,
        selected_tasks=args.tasks,
        selected_datasets=args.datasets
    )
    
    # Initialize SPARTA model
    sparse_model = SparseSearch(models.SPARTA(args.model), batch_size=128)
    retriever = EvaluateRetrieval(sparse_model)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Initialize CSV file with headers if it doesn't exist
    if not os.path.exists(args.output):
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Task", "Dataset", "Subtask",
                "NDCG@1", "NDCG@10",
                "MAP@1", "MAP@10",
                "Recall@1", "Recall@10",
                "P@1", "P@10"
            ])
    
    # Process each dataset
    for task, datasets in filtered_structure.items():
        for dataset_name, paths in datasets.items():
            logging.info(f"\nProcessing {task}/{dataset_name}")
            
            # Load data using custom data loader
            data_loader = CustomDataLoader(
                corpus_path=paths["corpus"],
                queries_path=paths["queries"],
                qrels_path=paths["qrels"]
            )
            corpus, queries, qrels = data_loader.load()
            
            # Handle different query structures
            if isinstance(queries, dict) and task == "PartOfPassageSearch":
                # Process each POP size separately
                for pop_size, pop_queries in queries.items():
                    logging.info(f"\nProcessing {pop_size}")
                    
                    # Retrieve results for this POP size
                    results = retriever.retrieve(corpus, pop_queries)
                    
                    # Evaluate using multiple metrics
                    logging.info(f"Evaluating retrieval for {pop_size}")
                    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
                    
                    # Print results
                    print("=" * 50)
                    print(f"Task: {task}")
                    print(f"Dataset: {dataset_name}")
                    print(f"POP size: {pop_size}")
                    print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
                    print(f"Recall@10: {recall['Recall@10']:.4f}")
                    print(f"Precision@10: {precision['P@10']:.4f}")
                    print("=" * 50)
                    
                    # Save results to CSV with file locking
                    with open(args.output, mode='a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                        try:
                            writer = csv.writer(file)
                            writer.writerow([
                                task,
                                dataset_name,
                                pop_size,
                                ndcg.get('NDCG@1', 0),
                                ndcg.get('NDCG@10', 0),
                                _map.get('MAP@1', 0),
                                _map.get('MAP@10', 0),
                                recall.get('Recall@1', 0),
                                recall.get('Recall@10', 0),
                                precision.get('P@1', 0),
                                precision.get('P@10', 0)
                            ])
                            file.flush()
                        finally:
                            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            else:
                # Regular processing for other tasks
                results = retriever.retrieve(corpus, queries)
                
                # Evaluate using multiple metrics
                logging.info("Evaluating retrieval for k in: {}".format(retriever.k_values))
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
                
                # Print results
                print("=" * 50)
                print(f"Task: {task}")
                print(f"Dataset: {dataset_name}")
                print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
                print(f"Recall@10: {recall['Recall@10']:.4f}")
                print(f"Precision@10: {precision['P@10']:.4f}")
                print("=" * 50)
                
                # Save results to CSV with file locking
                with open(args.output, mode='a', newline='') as file:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                    try:
                        writer = csv.writer(file)
                        writer.writerow([
                            task,
                            dataset_name,
                            "default",
                            ndcg.get('NDCG@1', 0),
                            ndcg.get('NDCG@10', 0),
                            _map.get('MAP@1', 0),
                            _map.get('MAP@10', 0),
                            recall.get('Recall@1', 0),
                            recall.get('Recall@10', 0),
                            precision.get('P@1', 0),
                            precision.get('P@10', 0)
                        ])
                        file.flush()
                    finally:
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)

if __name__ == "__main__":
    main()