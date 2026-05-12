import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import mteb
import faiss
import torch

import csv
import json

import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

import sys

# Get model path from command-line arguments
model_path = sys.argv[1]
if ' ' in model_path:
    model_list = model_path.split(' ')
else:
    model_list = [model_path]

# 设置代理
#os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
#os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"

from datasets import Features, Value, get_dataset_config_names, load_dataset
from mteb.abstasks.AbsTaskRetrieval import HFDataLoader

_original_load_qrels = HFDataLoader._load_qrels


def _patched_load_qrels(self, split):
    try:
        return _original_load_qrels(self, split)
    except ValueError as e:
        if "configurations in the cache" not in str(e):
            raise

        qrels_ds = load_dataset(
            self.hf_repo_qrels,
            "default",
            keep_in_memory=self.keep_in_memory,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
        )[split]
        self.qrels = qrels_ds.cast(
            Features({
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            })
        )


HFDataLoader._load_qrels = _patched_load_qrels


DIM_LIST = [16, 32, 64, 128, 256, 512]


# Initialize model
def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    # print(f"Loading model and tokenizer from {model_path}...")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer



class MyModel:
    def __init__(self, modelpath, device='cuda:0'):
        self.model = SentenceTransformer(modelpath, trust_remote_code=True).to(device)
        self.device = device
        self.cache = {}

        self.model_card_data = {
            "model_name": "MyModel",
            "description": "Custom embedding model",
            "version": "1.0"
        }

    def get_dim(self):
        return self.model.encode("hello", convert_to_tensor=True).shape[-1]
    
    def set_config(self, batch_size=4, emb_len=64, seed=42, win_size=64, chunk_ids=[0], dtype='def'):
        self.batch_size = batch_size
        self.emb_len = emb_len
        self.seed = seed
        self.win_size = win_size
        self.chunk_ids = chunk_ids
        self.dtype = dtype

    def encode(self, input_texts, **kwargs):
        if "convert_to_tensor" not in kwargs:
            kwargs["convert_to_tensor"] = True
        if "device" not in kwargs:
            kwargs["device"] = self.device



        results = [None] * len(input_texts)  # 先占位（缓存中统一存 CPU tensor）
        texts_to_encode = []
        idx_to_encode = []

        for i, text in enumerate(input_texts):
            if text in self.cache:
                results[i] = self.cache[text]
            else:
                texts_to_encode.append(text)
                idx_to_encode.append(i)

        if texts_to_encode:
            #self.model.max_seq_length = 1024 #max_length应该这么写才对 问一问
            kwargs.pop('task_name', None) #也得移除不认识的参数
            kwargs.pop('prompt_type', None)
            new_embeddings = self.model.encode(texts_to_encode, **kwargs)
            for i, text, emb in zip(idx_to_encode, texts_to_encode, new_embeddings):
                emb_cpu = emb.detach().to(torch.float32).cpu()
                self.cache[text] = emb_cpu
                results[i] = emb_cpu
            del new_embeddings

        embeddings = torch.stack(results, dim=0).to(self.device, non_blocking=True)
        
        embeddings = get_texts_embeddings(
                                            embeddings,
                                            self.get_dim(),
                                            emb_len=self.emb_len,
                                            seed=self.seed,
                                            win_size=self.win_size,
                                            chunk_ids=self.chunk_ids,
                                            dtype=self.dtype,
                                            device=self.device
                                        )
        
        # print('【 embeddings 】', self.dtype, embeddings.shape)
        out = embeddings.detach().to(torch.float32).cpu().numpy()
        del embeddings, results, texts_to_encode, idx_to_encode
        return out   # 直接返回 torch.Tensor

    def clear_runtime_cache(self):
        self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def get_texts_embeddings(embeddings, model_dim, emb_len=64, seed=42, win_size=64, chunk_ids=[0], dtype='def', device="cuda:0"):
    if dtype=='def':
        return get_texts_embeddings_def(embeddings)
    elif dtype=='random':
        return get_texts_embeddings_random(embeddings, model_dim, emb_len=emb_len, seed=seed, device=device)
    elif dtype=='chunk':
        return get_texts_embeddings_chunk(embeddings, win_size=win_size, chunk_ids=chunk_ids)


def get_texts_embeddings_def(embeddings: torch.Tensor):
    return embeddings


def get_texts_embeddings_random(embeddings: torch.Tensor, model_dim, emb_len=64, seed=42, device="cuda:0"):
    g = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randperm(model_dim, generator=g, device=device)[:emb_len]
    return embeddings.index_select(1, indices)


def get_texts_embeddings_chunk(embeddings: torch.Tensor, win_size=64, chunk_ids=[0]):
    if not chunk_ids:
        print("Warning: chunk_ids list is empty. Returning empty tensor.")
        return torch.empty((embeddings.shape[0], 0), device=embeddings.device)

    original_dim = embeddings.shape[1]
    max_requested_index = max(chunk_ids)
    max_possible_end_index = (max_requested_index + 1) * win_size

    if max_possible_end_index > original_dim:
        raise IndexError(
            f"Requested chunk_id {max_requested_index} (ending at index {max_possible_end_index}) "
            f"is out of bounds for embedding dimension {original_dim}."
        )

    chunks = [embeddings[:, i*win_size:(i+1)*win_size] for i in chunk_ids]
    return torch.cat(chunks, dim=1)






def main(model,
         task_name,
         batch_size=4, 
         emb_len=64, 
         seed=42,
         win_size=64, 
         chunk_ids=[0], 
         dtype="chunk"):
    if batch_size < 1:
        raise "GPU OUT"

    model.set_config(batch_size, emb_len, seed, win_size, chunk_ids, dtype)

    # print(task_name, batch_size, emb_len, seed, win_size, chunk_ids, dtype)

    #也是直接在这里导入数据集了,没用上之前整理的数据集
    tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
    evaluation = mteb.MTEB(tasks=tasks)
    results = None
    try:
        results = evaluation.run(
                            model,
                            verbosity=0,
                            overwrite_results=True,
                            # save_predictions=False,
                            output_folder=f"./mteb/{model_name}",
                            encode_kwargs={'batch_size': batch_size},
                            save_corpus_embeddings=False,
                            # do_length_ablation=True
                            )
    except Exception as e:
        print(e)
        return main(model,
                    task_name,
                    batch_size=batch_size//2,
                    win_size=2,
                    chunk_ids=chunk_ids,
                    dtype=dtype)
    finally:
        del evaluation, tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # for res in results:
    #     print(f"Task: {res.task_name}")
    #     print('========================')
    #     print(f"scores: {res.scores['test'][0]['main_score']}")
    #     print('========================')
        # for metric, value in res.scores.items():
        #     print(f"  {metric}: {value}")

    main_score = results[0].scores['test'][0]['main_score']*100

    # a = input(main_score)
    
    
    del results
    return main_score


def write_result_csv(new_data, csv_filename='use_embedding_result.csv'):

    # 读取现有数据，如果文件存在
    existing_data = {}
    key_order = ['Model Name']  # 初始化键顺序，确保Model Name是第一个
    if os.path.exists(csv_filename):
        with open(csv_filename, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            key_order.extend(reader.fieldnames[1:])  # 保留现有数据的键顺序
            for row in reader:
                model_name = row['Model Name']
                existing_data[model_name] = row

    # 检查new_data中的键，将新的键添加到键顺序列表中
    for model_name, metrics in new_data.items():
        for key in metrics.keys():
            if key not in key_order:
                key_order.append(key)

    # 更新现有数据或添加新数据
    for model_name, metrics in new_data.items():
        if model_name not in existing_data:
            existing_data[model_name] = {'Model Name': model_name}
        for key in key_order[1:]:  # 从第二个键开始，因为第一个是Model Name
            if key in metrics:
                existing_data[model_name][key] = metrics[key]
            elif key not in existing_data[model_name]:  # 如果现有数据中也没有这个键，则补全为-1
                existing_data[model_name][key] = -1

    # 写入所有数据到CSV文件
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=key_order)
        writer.writeheader()
        for model_name, metrics in existing_data.items():
            # 确保所有参数都被填充，缺失的用-1代替
            row_to_write = {key: metrics.get(key, -1) for key in key_order}
            writer.writerow(row_to_write)

    # print(f'Data has been written to {csv_filename}')



task_categories = {
    "Retrieval": [
        'ArguAna', 
    ],
    "Classification": [
        'MTOPDomainClassification'
    ],
    "Clustering": [
        'BiorxivClusteringS2S'
    ],
    "PairClassification": [
        'TwitterSemEval2015'
    ],
    "Reranking": [
        'MindSmallReranking'
    ],
    "STS": [
        'BIOSSES', 'SICK-R', 'STSBenchmark'
    ],
    "Summarization": [
        'SummEval'
    ]
}




task_categories = {
    "Classification": [
        'AmazonCounterfactualClassification',
        'AmazonReviewsClassification', 'Banking77Classification',
        'EmotionClassification', 'ImdbClassification',
        'MTOPDomainClassification', 'MTOPIntentClassification',
        'MassiveIntentClassification', 'MassiveScenarioClassification',
        'ToxicConversationsClassification', 'TweetSentimentExtractionClassification'
    ],
    "Clustering": [
        'BiorxivClusteringS2S',
        'MedrxivClusteringS2S',
        'TwentyNewsgroupsClustering'
    ],
    "PairClassification": [
        'SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus'
    ],
    "Reranking": [
        'AskUbuntuDupQuestions', 'SciDocsRR', 'StackOverflowDupQuestions'
    ],
    "STS": [
        'BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15',
        'STS16', 'STS17', 'STSBenchmark'
    ],
    "Summarization": [
        'SummEval'
    ],
    "Retrieval": [
        'ArguAna', 'CQADupstackEnglishRetrieval',
        'NFCorpus', 'SCIDOCS', 'SciFact'
    ],
}


# Example usage
if __name__ == "__main__":

    # model_list = [
        # '/data/fengdm/models/bart-base',
        # '/data/fengdm/models/bge-m3',
        # '/data/fengdm/models/gte-base',
        # '/data/fengdm/models/jina-embeddings-v3',
        # '/data/fengdm/models/gtr-t5-large',
        # '/data/fengdm/models/instructor-large',
        # '/data/fengdm/models/roberta-large-InBedder',

        # '/data/fengdm/models/stella_en_400M_v5',
        # '/data/fengdm/models/stella_en_400M_v5-GEDI/epoch_3',
        # '/data/fengdm/models/gte-large-en-v1.5',
        # '/data/fengdm/models/Qwen3-Embedding-0.6B',
        # '/data/fengdm/models/roberta-large',
        # '/data/fengdm/models/mxbai-embed-large-v1',
    # ]


    
    #folder_path = './data/analyze/'
    folder_path = './data/analyze_new/'
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    all_models_data = {}

    for file in json_files:
        model_name = file.replace(".json", "")
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_models_data[model_name] = data

    
    # print(len(model_list))
    for model_path in model_list:
        
        model_name = model_path.replace('/data/fengdm/models/', '').replace('/', '-')

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = MyModel(model_path, device)

        analyze_path = "./data/task_similar_new"

        # 判断文件夹是否存在，不存在就创建
        if not os.path.exists(analyze_path):
            os.makedirs(analyze_path)
            print(f"文件夹 '{analyze_path}' 创建成功")
        else:
            print(f"文件夹 '{analyze_path}' 已存在")

        print(f"Model: {model_name}", analyze_path + "/{}.json".format(model_name))

        if os.path.exists(analyze_path + "/{}.json".format(model_name)):
            print(model_name, "已存在")
            with open(analyze_path + "/{}.json".format(model_name), "r", encoding="utf-8") as f:
                out_result_json = json.load(f)
            print("已经处理过的任务:\n", "\n".join(out_result_json.keys()), "\n==========================")
            
        else:
            out_result_json = {}


        #  外层循环选择不同参考任务,分析应该选用哪些chunks,并在目标任务上连续评估多个保留维度
        for category_s, task_names_s in task_categories.items():
            for dataset_name_s in task_names_s:
                print("当前参考任务：", dataset_name_s)
                out_result_json.setdefault(dataset_name_s, {})

                # 基于 win_size=2 下每个 chunk 的得分对 chunk 索引降序排序
                result = all_models_data[model_name]['task_name'][dataset_name_s]['split_win_size']['2']['chunk_result']
                sorted_index_list = [index for index, _ in sorted(enumerate(result), key=lambda x: x[1], reverse=True)]

                # 内两层：先目标任务 t，再维度 d；同一 t 连续跑完所有 d 以复用 MyModel.cache
                for category, task_names in task_categories.items():
                    for dataset_name in task_names:
                        print("当前任务：", dataset_name)
                        out_result_json[dataset_name_s].setdefault(dataset_name, {})

                        for d in DIM_LIST:
                            n_chunks = d // 2
                            if n_chunks > len(sorted_index_list):
                                print(f"跳过 dim={d}: 需要 {n_chunks} 个 chunk, 仅有 {len(sorted_index_list)} 个可用")
                                continue

                            d_key = str(d)
                            if d_key in out_result_json[dataset_name_s][dataset_name]:
                                print(
                                    f"[已完成-跳过] model={model_name} | ref_task={dataset_name_s} "
                                    f"| target_task={dataset_name} | dim={d} | win_size=2 | chunks={n_chunks}"
                                )
                                continue

                            chunk_ids = sorted_index_list[:n_chunks]
                            print(
                                f"[开始评测] model={model_name} | ref_task={dataset_name_s} "
                                f"| target_task={dataset_name} | dim={d} | win_size=2 | chunks={n_chunks}"
                            )
                            main_score = main(model,
                                              dataset_name,
                                              batch_size=8,
                                              win_size=2,
                                              chunk_ids=chunk_ids,
                                              dtype='chunk')

                            out_result_json[dataset_name_s][dataset_name][d_key] = main_score
                            print(
                                f"[评测完成] model={model_name} | ref_task={dataset_name_s} "
                                f"| target_task={dataset_name} | dim={d} | score={main_score:.6f}"
                            )

                        # 同一目标任务的多个维度跑完后再清理，避免 cache 在任务间无限增长
                        model.clear_runtime_cache()

                        with open(analyze_path + "/{}.json".format(model_name), 'w', encoding='utf-8') as file:
                            json.dump(out_result_json, file, ensure_ascii=False, indent=4)

        # 将方案 A 结果整合成方案 B（按维度汇总）视图
        by_dim = {}
        for s, t_map in out_result_json.items():
            by_dim[s] = {}
            for t, d_map in t_map.items():
                for d_key, score in d_map.items():
                    by_dim[s].setdefault(d_key, {})[t] = score
            by_dim[s] = {str(d): by_dim[s][str(d)] for d in DIM_LIST if str(d) in by_dim[s]}

        with open(analyze_path + "/{}_by_dim.json".format(model_name), 'w', encoding='utf-8') as file:
            json.dump(by_dim, file, ensure_ascii=False, indent=4)


        print("释放模型显存")
        import gc
        import time

        def free_gpu(delay: float = 2.0):
            """彻底清理 GPU 显存"""
            print("[清理] 删除对象并触发垃圾回收...")
            gc.collect()
            
            print("[清理] 清空 PyTorch CUDA 缓存...")
            torch.cuda.empty_cache()
            
            print(f"[清理] 等待 {delay} 秒，让 CUDA 驱动同步...")
            time.sleep(delay)
            
            print("[清理] 尝试释放未使用的显存...")
            torch.cuda.ipc_collect()  # 清理跨进程的内存引用
            
            print("[完成] GPU 内存清理完成。")
        
        # 使用示例
        del model  # 确保先删除模型和优化器
        free_gpu(1.0)  # 延时 3 秒


        
        
            # model_list = [
    #     '/data/fengdm/models/bart-base',
    #     '/data/fengdm/models/bge-m3',
    #     '/data/fengdm/models/gte-base',
    #     '/data/fengdm/models/jina-embeddings-v3',
    #     '/data/fengdm/models/gtr-t5-large',
    #     '/data/fengdm/models/instructor-large',
    #     '/data/fengdm/models/roberta-large-InBedder',

    #     # '/data/fengdm/models/stella_en_400M_v5',
    #     # '/data/fengdm/models/stella_en_400M_v5-GEDI/epoch_3',
    #     # '/data/fengdm/models/gte-large-en-v1.5',
    #     # '/data/fengdm/models/roberta-large',
    #     # '/data/fengdm/models/mxbai-embed-large-v1',
    #     # '/data/fengdm/models/Qwen3-Embedding-0.6B',
    #     # '/data/fengdm/models/gte-Qwen2-1.5B-instruct',
    
    # ]

# CUDA_VISIBLE_DEVICES=0 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/bart-base"

# CUDA_VISIBLE_DEVICES=1 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/bge-m3"

# CUDA_VISIBLE_DEVICES=2 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/gte-base"

# CUDA_VISIBLE_DEVICES=3 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/gtr-t5-large"

# CUDA_VISIBLE_DEVICES=4 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/instructor-large"

# CUDA_VISIBLE_DEVICES=5 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/roberta-large-InBedder"

# CUDA_VISIBLE_DEVICES=6 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/roberta-large"

# CUDA_VISIBLE_DEVICES=7 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/mxbai-embed-large-v1"

# CUDA_VISIBLE_DEVICES=6 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/Qwen3-Embedding-0.6B"

# CUDA_VISIBLE_DEVICES=2 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/gte-large-en-v1.5"

# CUDA_VISIBLE_DEVICES=1 /data/fengdm/anaconda3/envs/embedding/bin/python "./10 task similar mteb.py"  "/data/fengdm/models/stella_en_400M_v5"


"""
{
    "model_name": ,
    "model_dim": ,
    "task_name": {
        "fever": {
            "defult_score": {
                "main_score": ,
                "mtt": ,
                "recall": 
            },
            "split_win_size": {
                8: {
                    "chunk_result": [],
                    "chunk_win_size": {
                        64: {
                            "score": {
                                "main_score": ,
                                "mtt": ,
                                "recall": 
                            }
                        },
                        128: {
                            "score": {
                                "main_score": ,
                                "mtt": ,
                                "recall": 
                            }
                        }
                    }
                },
                16: {
                    "chunk_result": [],
                    "chunk_win_size": {
                        64: {
                            "score": {
                                "main_score": ,
                                "mtt": ,
                                "recall": 
                            }
                        },
                        128: {
                            "score": {
                                "main_score": ,
                                "mtt": ,
                                "recall": 
                            }
                        }
                    }
                },
            }
        }
    }
}
"""