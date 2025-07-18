from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
import csv
import sys
import os
from CustomJsonLoader import read_json, list2dict
import pathlib


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_list = [
    #Note: Add model name or Model Path Here
]

dataset_list = ["hotpotqa", "nq", "fever"]
task_list = ["query", "keyword", "pop16", "pop32", "pop64", "pop128", "pop256"]

corpus_path = {
    "hotpotqa": {
        "query": "./eval_data/hotpotqa5000/query/hotpotqa_text5000.json",
        "pop16": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_text5000.json",
        "pop32": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_text5000.json",
        "pop64": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_text5000.json",
        "pop128": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_text5000.json",
        "pop256": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_text5000.json",
        "keyword": "./eval_data/hotpotqa5000/keyword/hotpotqa_pop_text5000.json"
    },
    "nq": {
        "query": "./eval_data/nq5000/query/nq_text5000.json",
        "pop16": "./eval_data/nq5000/pop/nq_pop_text5000.json",
        "pop32": "./eval_data/nq5000/pop/nq_pop_text5000.json",
        "pop64": "./eval_data/nq5000/pop/nq_pop_text5000.json",
        "pop128": "./eval_data/nq5000/pop/nq_pop_text5000.json",
        "pop256": "./eval_data/nq5000/pop/nq_pop_text5000.json",
        "keyword": "./eval_data/nq5000/keyword/nq_keyword_text5000.json"
    },
    "fever": {
        "query": "./eval_data/fever1500/query/fever_text1500.json",
        "pop16": "./eval_data/fever1500/pop/fever_pop_text.json",
        "pop32": "./eval_data/fever1500/pop/fever_pop_text.json",
        "pop64": "./eval_data/fever1500/pop/fever_pop_text.json",
        "pop128": "./eval_data/fever1500/pop/fever_pop_text.json",
        "pop256": "./eval_data/fever1500/pop/fever_pop_text.json",
        "keyword": "./eval_data/fever1500/keyword/fever_pop_text.json"
    }
}

queries_path = {
    "hotpotqa": {
        "query": "./eval_data/hotpotqa5000/query/hotpotqa_queries5000.json",
        "pop16": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_16_queries.json",
        "pop32": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_32_queries.json",
        "pop64": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_64_queries.json",
        "pop128": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_128_queries.json",
        "pop256": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_256_queries.json",
        "keyword": "./eval_data/hotpotqa5000/keyword/hotpotqa_keyword_queries.json"
    },
    "nq": {
        "query": "./eval_data/nq5000/query/nq_queries5000.json",
        "pop16": "./eval_data/nq5000/pop/nq_pop_16_queries.json",
        "pop32": "./eval_data/nq5000/pop/nq_pop_32_queries.json",
        "pop64": "./eval_data/nq5000/pop/nq_pop_64_queries.json",
        "pop128": "./eval_data/nq5000/pop/nq_pop_128_queries.json",
        "pop256": "./eval_data/nq5000/pop/nq_pop_256_queries.json",
        "keyword": "./eval_data/nq5000/keyword/nq_keyword_queries.json"
    },
    "fever": {
        "query": "./eval_data/fever1500/query/fever_queries.json",
        "pop16": "./eval_data/fever1500/pop/fever_pop_16_queries.json",
        "pop32": "./eval_data/fever1500/pop/fever_pop_32_queries.json",
        "pop64": "./eval_data/fever1500/pop/fever_pop_64_queries.json",
        "pop128": "./eval_data/fever1500/pop/fever_pop_128_queries.json",
        "pop256": "./eval_data/fever1500/pop/fever_pop_256_queries.json",
        "keyword": "./eval_data/fever1500/keyword/fever_keyword_queries.json"
    }
}

qrels_path = {
    "hotpotqa": {
        "query": "./eval_data/hotpotqa5000/query/hotpotqa_qrels5000.json",
        "pop16": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_qrels.json",
        "pop32": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_qrels.json",
        "pop64": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_qrels.json",
        "pop128": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_qrels.json",
        "pop256": "./eval_data/hotpotqa5000/pop/hotpotqa_pop_qrels.json",
        "keyword": "./eval_data/hotpotqa5000/keyword/hotpotqa_keyword_qrels.json"
    },
    "nq": {
        "query": "./eval_data/nq5000/query/nq_qrels5000.json",
        "pop16": "./eval_data/nq5000/pop/nq_pop_qrels.json",
        "pop32": "./eval_data/nq5000/pop/nq_pop_qrels.json",
        "pop64": "./eval_data/nq5000/pop/nq_pop_qrels.json",
        "pop128": "./eval_data/nq5000/pop/nq_pop_qrels.json",
        "pop256": "./eval_data/nq5000/pop/nq_pop_qrels.json",
        "keyword": "./eval_data/nq5000/keyword/nq_keyword_qrels.json"
    },
    "fever": {
        "query": "./eval_data/fever1500/query/fever_qrels.json",
        "pop16": "./eval_data/fever1500/pop/fever_pop_qrels.json",
        "pop32": "./eval_data/fever1500/pop/fever_pop_qrels.json",
        "pop64": "./eval_data/fever1500/pop/fever_pop_qrels.json",
        "pop128": "./eval_data/fever1500/pop/fever_pop_qrels.json",
        "pop256": "./eval_data/fever1500/pop/fever_pop_qrels.json",
        "keyword": "./eval_data/fever1500/keyword/fever_keyword_qrels.json"
    }
}

for dataset_name in dataset_list:
    for task_name in task_list:
        for model_name in model_list:
            corpus = read_json(corpus_path[dataset_name][task_name])
            queries = list2dict(read_json(queries_path[dataset_name][task_name]))
            qrels = list2dict(read_json(qrels_path[dataset_name][task_name]))
            # logging.info("Current Model: {}".format(model_name))
            model = DRES(models.SentenceBERT(model_name), batch_size=16)

            retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "cos_sim" for cosine similarity
            results = retriever.retrieve(corpus, queries)

            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
            #---
            print("=============================")
            print("Current dataset：{}".format(dataset_name))
            print("Current Model：{}".format(model_name))
            print("Current Task：", task_name)
            print("NDCG@1: {:.4f} | NDCG@10: {:.4f}".format(ndcg['NDCG@1'], ndcg['NDCG@10']))
            print("MRR@1: {:.4f} | MRR@10: {:.4f}".format(mrr['MRR@1'], mrr['MRR@10']))
            print("Recall@1: {:.4f} | Recall@10: {:.4f}".format(recall['Recall@1'], recall['Recall@10']))
            print("=============================")
            #---
            with open('./results.csv', mode='a', newline='') as file:
                writer = csv.writer(file)

                # Check whether the file is empty. If it is empty, write it to the header.
                if file.tell() == 0: 
                    writer.writerow(["Datasets", "Model", "Task", "NDCG@1", "NDCG@10", "MRR@1", "MRR@10", "Recall@1", "Recall@10"])

                # 写入数据
                writer.writerow([
                    dataset_name,
                    model_name, 
                    task_name, 
                    ndcg['NDCG@1'], ndcg['NDCG@10'], 
                    mrr['MRR@1'], mrr['MRR@10'], 
                    recall['Recall@1'], recall['Recall@10']
                ])

