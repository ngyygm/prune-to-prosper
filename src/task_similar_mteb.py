import os

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
os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"


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



        results = [None] * len(input_texts)  # 先占位
        texts_to_encode = []
        idx_to_encode = []

        for i, text in enumerate(input_texts):
            if text in self.cache:
                results[i] = self.cache[text]
            else:
                texts_to_encode.append(text)
                idx_to_encode.append(i)

        if texts_to_encode:
            new_embeddings = self.model.encode(texts_to_encode, max_len=1024, **kwargs)
            for i, text, emb in zip(idx_to_encode, texts_to_encode, new_embeddings):
                self.cache[text] = emb.detach()
                results[i] = emb.detach()

        embeddings = torch.stack(results, dim=0)
        
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
        return embeddings.detach().to(torch.float32).cpu().numpy()   # 直接返回 torch.Tensor


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


    tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
    evaluation = mteb.MTEB(tasks=tasks)
    try:
        results = evaluation.run(
                            model,
                            verbosity=0,
                            overwrite_results=True,
                            # save_predictions=False,
                            output_folder=f"/home/linkco/exa/Useful-Embedding/mteb/{model_name}",
                            encode_kwargs={'batch_size': batch_size},
                            save_corpus_embeddings=False,
                            # do_length_ablation=True
                            )
    except Exception as e:
        print(e)
        return main(model, 
                    dataset_name,
                    batch_size=batch_size/2,
                    win_size=2, 
                    chunk_ids=chunk_ids, 
                    dtype=dtype)

    # for res in results:
    #     print(f"Task: {res.task_name}")
    #     print('========================')
    #     print(f"scores: {res.scores['test'][0]['main_score']}")
    #     print('========================')
        # for metric, value in res.scores.items():
        #     print(f"  {metric}: {value}")

    main_score = results[0].scores['test'][0]['main_score']*100

    # a = input(main_score)
    
    
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
        # '/home/linkco/exa/models/bart-base',
        # '/home/linkco/exa/models/bge-m3',
        # '/home/linkco/exa/models/gte-base',
        # '/home/linkco/exa/models/jina-embeddings-v3',
        # '/home/linkco/exa/models/gtr-t5-large',
        # '/home/linkco/exa/models/instructor-large',
        # '/home/linkco/exa/models/roberta-large-InBedder',

        # '/home/linkco/exa/models/stella_en_400M_v5',
        # '/home/linkco/exa/models/stella_en_400M_v5-GEDI/epoch_3',
        # '/home/linkco/exa/models/gte-large-en-v1.5',
        # '/home/linkco/exa/models/Qwen3-Embedding-0.6B',
        # '/home/linkco/exa/models/roberta-large',
        # '/home/linkco/exa/models/mxbai-embed-large-v1',
    # ]


    
    folder_path = '/home/linkco/exa/Useful-Embedding/data/analyze/'
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
        
        model_name = model_path.replace('/home/linkco/exa/models/', '').replace('/', '-')

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = MyModel(model_path, device)

        analyze_path = "/home/linkco/exa/Useful-Embedding/data/task_similar"

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


        # 第1步 处理应该选用哪些chunks

        for category_s, task_names_s in task_categories.items():
            for dataset_name_s in task_names_s:
                print("当前参考任务：", dataset_name_s)
                # 先判断这个任务处理过没
                if dataset_name_s not in out_result_json:
                    out_result_json[dataset_name_s] = {}

                # 设定为一个任务是否可以推广到其他任务，因此选用特定的某个任务，用Classification\MTOPDomainClassification作为实验对象
                result = all_models_data[model_name]['task_name'][dataset_name_s]['split_win_size']['2']['chunk_result']

                sorted_index_list = [index for index, value in sorted(enumerate(result), key=lambda x: x[1], reverse=True)]

                select_index = 128

                chunk_ids = sorted_index_list[:128]
                print('【chunk_ids】', chunk_ids)

                
                # 开始跑结果
                for category, task_names in task_categories.items():
                    # if category == category_s:
                    for dataset_name in task_names:
                        print("当前任务：", dataset_name)
                        if dataset_name in out_result_json[dataset_name_s]:
                            continue


                        main_score = main(model, 
                                            dataset_name,
                                            batch_size=8,
                                            win_size=2, 
                                            chunk_ids=chunk_ids, 
                                            dtype='chunk')
                        
                        out_result_json[dataset_name_s][dataset_name] = main_score

                        with open(analyze_path + "/{}.json".format(model_name), 'w', encoding='utf-8') as file:
                            # 确保json格式是美化打印的，并确保中文能被正确写入
                            json.dump(out_result_json, file, ensure_ascii=False, indent=4)
                        
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
    #     '/home/linkco/exa/models/bart-base',
    #     '/home/linkco/exa/models/bge-m3',
    #     '/home/linkco/exa/models/gte-base',
    #     '/home/linkco/exa/models/jina-embeddings-v3',
    #     '/home/linkco/exa/models/gtr-t5-large',
    #     '/home/linkco/exa/models/instructor-large',
    #     '/home/linkco/exa/models/roberta-large-InBedder',

    #     # '/home/linkco/exa/models/stella_en_400M_v5',
    #     # '/home/linkco/exa/models/stella_en_400M_v5-GEDI/epoch_3',
    #     # '/home/linkco/exa/models/gte-large-en-v1.5',
    #     # '/home/linkco/exa/models/roberta-large',
    #     # '/home/linkco/exa/models/mxbai-embed-large-v1',
    #     # '/home/linkco/exa/models/Qwen3-Embedding-0.6B',
    #     # '/home/linkco/exa/models/gte-Qwen2-1.5B-instruct',
    
    # ]


# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/llama/bin/python "/home/linkco/exa/Useful-Embedding/10 task similar mteb.py"  "/home/linkco/exa/models/stella_en_400M_v5"

# CUDA_VISIBLE_DEVICES=1 /home/linkco/anaconda3/envs/llama/bin/python "/home/linkco/exa/Useful-Embedding/10 task similar mteb.py"  "/home/linkco/exa/models/gtr-t5-large /home/linkco/exa/models/instructor-large /home/linkco/exa/models/roberta-large-InBedder /home/linkco/exa/models/roberta-large"

# CUDA_VISIBLE_DEVICES=1 /home/linkco/anaconda3/envs/llama/bin/python "/home/linkco/exa/Useful-Embedding/10 task similar mteb.py"  "/home/linkco/exa/models/bart-base /home/linkco/exa/models/bge-m3 /home/linkco/exa/models/gte-base"

# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/llama/bin/python "/home/linkco/exa/Useful-Embedding/10 task similar mteb.py"  "/home/linkco/exa/models/stella_en_400M_v5-GEDI/epoch_3 /home/linkco/exa/models/gte-large-en-v1.5 /home/linkco/exa/models/stella_en_400M_v5"

# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/llama/bin/python "/home/linkco/exa/Useful-Embedding/10 task similar mteb.py"  "/home/linkco/exa/models/mxbai-embed-large-v1 /home/linkco/exa/models/Qwen3-Embedding-0.6B /home/linkco/exa/models/gte-Qwen2-1.5B-instruct"


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