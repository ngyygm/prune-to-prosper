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




# class MyModel():
#     def __init__(self, modelpath, device='cuda:0'):
#         self.model = SentenceTransformer(modelpath, trust_remote_code=True).to(device)
#         self.device = device
#         self.cache = {}  # 添加缓存字典
        
#         self.model_card_data = {
#             "model_name": "MyModel",
#             "description": "Custom embedding model",
#             "version": "1.0"
#         }

#     def get_dim(self):
#         return len(self.model.encode("hello"))
    
#     def set_config(self, batch_size=32, emb_len=64, seed=42, win_size=64, chunk_ids=[0], dtype='def'):
#         self.batch_size = batch_size
#         self.emb_len = emb_len
#         self.seed = seed
#         self.win_size = win_size
#         self.chunk_ids = chunk_ids
#         self.dtype = dtype
#         # print("model set_config", self.batch_size, self.emb_len, self.seed, self.win_size, self.chunk_ids, self.dtype)



#     def encode(self, inputtexts, max_length=2048, **kwargs):
        
#         # print("model encode", self.batch_size, self.emb_len, self.seed, self.win_size, self.chunk_ids, self.dtype)
#         # 初始化嵌入列表和待编码文本列表
#         embeddings = []
#         texts_to_encode = []

#         # 遍历输入文本，检查缓存
#         for text in inputtexts:
#             if text in self.cache:
#                 # 如果在缓存中，直接添加到嵌入列表
#                 embeddings.append(self.cache[text])
#             else:
#                 # 如果不在缓存中，添加到待编码文本列表
#                 texts_to_encode.append(text)

#         # 如果有待编码的文本，进行批量编码
#         if texts_to_encode:
#             new_embeddings = self.model.encode(texts_to_encode, **kwargs)
#             # 将新编码的嵌入存入缓存，并添加到嵌入列表
#             for text, embedding in zip(texts_to_encode, new_embeddings):
#                 if isinstance(embedding, torch.Tensor):
#                     embedding = embedding.detach().cpu().to(torch.float32).numpy()
#                 if hasattr(embedding, "cpu"):  # torch.Tensor
#                     self.cache[text] = embedding.detach().cpu().numpy()
#                     embeddings.append(embedding.detach().cpu().numpy())
#                 else:  # numpy.ndarray
#                     embeddings.append(embedding)
#                     self.cache[text] = embedding
        
#         embeddings = get_texts_embeddings(np.array(embeddings),
#                                             self.get_dim(),
#                                             emb_len=self.emb_len, 
#                                             seed=self.seed, 
#                                             win_size=self.win_size, 
#                                             chunk_ids=self.chunk_ids, 
#                                             dtype=self.dtype)
        
#         print('【 embeddings 】', self.dtype, embeddings.shape)

#         return np.array(embeddings)  # 将列表转换为NumPy数组



# def get_texts_embeddings(embeddings, model_dim, emb_len=64, seed=42, win_size=64, chunk_ids=[0], dtype='def'):
#     if dtype=='def':
#         # print('def')
#         return get_texts_embeddings_def(embeddings)
#     elif dtype=='random':
#         # print('random')
#         return get_texts_embeddings_random(embeddings, model_dim, emb_len=emb_len, seed=seed)
#     elif dtype=='chunk':
#         # print('chunk')
#         return get_texts_embeddings_chunk(embeddings, win_size=win_size, chunk_ids=chunk_ids)


# def get_texts_embeddings_def(embeddings):
#     return embeddings[:, :512]



# # --- 第一个函数：随机抽取参数保留 ---

# def get_texts_embeddings_random(embeddings, model_dim, emb_len=64, seed=42):
#     """
#     对文本进行批处理编码，并对每个生成的embedding向量进行随机降维。

#     Args:
#         texts (list): 字符串列表，包含所有待编码的文本。
#         model: 任何具有 .encode() 方法的编码模型（如Sentence-Transformer）。
#         batch_size (int): 每批处理的文本数量。
#         emb_len (int): 随机抽取后要保留的embedding维度。

#     Returns:
#         np.ndarray: 一个NumPy数组，形状为 (len(texts), emb_len)，包含了所有降维后的embedding。
#     """
#     # 设置随机种子
#     np.random.seed(seed)
#     # 生成固定的抽取索引
#     indices = np.random.choice(model_dim, min(emb_len, model_dim), replace=False)


#     all_embeddings = embeddings[:, indices]

#     return all_embeddings


# # --- 第二个函数：按窗口切片，保留特定片段 ---

# import numpy as np

# def get_texts_embeddings_chunk(embeddings, win_size=64, chunk_ids=[0]):
#     """
#     对文本进行批处理编码，并对每个生成的embedding向量进行多块切片，然后将这些切片拼接在一起。

#     Args:
#         embeddings (np.ndarray): 形状为 (batch_size, embedding_dim) 的数组，
#                                  包含所有待变化的编码。
#         win_size (int): 每个切片的窗口大小。
#         chunk_ids (list): 一个整数列表，指定要提取哪些片段并进行拼接。
#                           例如，win_size=64, chunk_ids=[0, 2] 会提取第0块 [0:64] 和
#                           第2块 [128:192]，并将它们拼接成形状为 (..., 128) 的向量。

#     Returns:
#         np.ndarray: 一个NumPy数组，形状为 (batch_size, len(chunk_ids) * win_size)，
#                     包含了所有拼接后的embedding。
#     """
#     # print("start  get_texts_embeddings_chunk", embeddings.shape, win_size, chunk_ids)
#     # --- 输入校验 ---
#     if not chunk_ids:
#         print("Warning: chunk_ids list is empty. Returning empty array.")
#         return np.array([])

#     # 1. 获取原始embedding的维度
#     original_dim = embeddings.shape[1]

#     # 2. 检查所有请求的chunk是否都在有效范围内
#     max_requested_index = max(chunk_ids)
#     max_possible_end_index = (max_requested_index + 1) * win_size

#     if max_possible_end_index > original_dim:
#         raise IndexError(
#             f"Requested chunk_id {max_requested_index} (ending at index {max_possible_end_index}) "
#             f"is out of bounds for embedding dimension {original_dim}."
#         )

#     # 3. 切片并拼接
#     chunks = [embeddings[:, i * win_size : (i + 1) * win_size] for i in chunk_ids]
#     result = np.hstack(chunks)

#     return result






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
                    emb_len=max_len,
                    seed=seed,
                    win_size=win_size, 
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
    
    
    # print(len(model_list))
    for model_path in model_list:
        
        model_name = model_path.replace('/home/linkco/exa/models/', '').replace('/', '-')

        analyze_path = "/home/linkco/exa/Useful-Embedding/data/analyze"
        print(f"Model: {model_name}", analyze_path + "/{}.json".format(model_name))

        if os.path.exists(analyze_path + "/{}.json".format(model_name)):
            print(model_name, "已存在")
            with open(analyze_path + "/{}.json".format(model_name), "r", encoding="utf-8") as f:
                out_result_json = json.load(f)

            # for category, task_names in task_categories.items():
            #     for dataset_name in task_names:
            #         if dataset_name in out_result_json["task_name"].keys():
            #             for win_size in out_result_json["task_name"][dataset_name]["split_win_size"].keys():
            #                 if "chunk_win_size" in out_result_json["task_name"][dataset_name]["split_win_size"][win_size].keys():
            #                     for max_len in out_result_json["task_name"][dataset_name]["split_win_size"][win_size]["chunk_win_size"].keys():
            #                         out_result_json["task_name"][dataset_name]["split_win_size"][win_size]["chunk_win_size"][max_len]["sort_score"] = {
            #                                                 "main_score": out_result_json["task_name"][dataset_name]["sort_score"][max_len]
            #                                             }
            # with open(analyze_path + "/{}.json".format(model_name), 'w', encoding='utf-8') as file:
            #     # 确保json格式是美化打印的，并确保中文能被正确写入
            #     json.dump(out_result_json, file, ensure_ascii=False, indent=4)
        else:
            out_result_json = {}
            out_result_json["model_name"] = model_name
            out_result_json["task_name"] = {}

        print("已经处理过的任务:\n", "\n".join(out_result_json["task_name"].keys()), "\n==========================")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 'Instruction', 'raceQA', 'raceFind', 'fever'
        for category, task_names in task_categories.items():
            for dataset_name in task_names:
                
                print("当前任务：", dataset_name)

                model = MyModel(model_path, device)

                size_list = []
                for max_len in [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 384, 512, 768, 1024]:
                    if max_len < model.get_dim():
                        size_list.append(max_len)
                
                out_result_json["model_dim"] = model.get_dim()

                      
        
                batch_size = 8
                # 使用随机的参数
                emb_len=64
                seed=66

                

                # 使用chunk的参数
                win_size=64
                # chunk_id=0
                # 类型
                dtype='chunk'


                if dataset_name not in out_result_json["task_name"].keys():
                    print(dataset_name, "任务不存在")
                    out_result_json["task_name"][dataset_name] = {
                            "defult_score": 0,
                            "random_score": {},
                            "sort_score": {},
                            "split_win_size": {}
                        }
              


                    # 原始结果
                    main_score = main(model, 
                                      dataset_name,
                                      batch_size=batch_size, 
                                      dtype="def")
                    
                    
                    
                    out_result_json["task_name"][dataset_name]["defult_score"] = main_score
                    
                    print('---------------原始结果---{}-----------'.format(main_score))


                #  先处理随机选取的结果
                for max_len in size_list:
                    print("当前处理随机和顺序", max_len)
                    # a = input()
                    if str(max_len) not in [str(it) for it in out_result_json["task_name"][dataset_name]["random_score"].keys()]:
                        
                        print("处理随机", max_len)
                        temp_random_score = []
                        for seed in range(10):
                            main_score = main(model, 
                                                dataset_name,
                                                batch_size=batch_size, 
                                                emb_len=max_len,
                                                seed=seed,
                                                win_size=win_size, 
                                                chunk_ids=[0], 
                                                dtype="random")
                            temp_random_score.append(main_score)
                            print('---------------随机截取--{}---{}---{}----------'.format(max_len, seed, main_score))
                        
                        out_result_json["task_name"][dataset_name]["random_score"][str(max_len)] = temp_random_score
                    
                        print('---------------随机截取平均--{}---{}-------------'.format(max_len, np.mean(temp_random_score)))

                    
                    if str(max_len) not in [str(it) for it in out_result_json["task_name"][dataset_name]["sort_score"].keys()]:
                        print("处理顺序", max_len)
                        main_score = main(model, 
                                            dataset_name,
                                            batch_size=batch_size, 
                                            emb_len=max_len,
                                            seed=seed,
                                            win_size=1, 
                                            chunk_ids=range(max_len), 
                                            dtype='chunk')
                        out_result_json["task_name"][dataset_name]["sort_score"][str(max_len)] = main_score
                        
                        print('---------------顺序截取--{}---{}-------------'.format(max_len, main_score))
    
                    # out_result_json["task_name"][dataset_name]["random_score"] = random_main_result
                    # out_result_json["task_name"][dataset_name]["sort_score"] = sort_main_result


                print("out_result_json[task_name][dataset_name][sort_score]", out_result_json["task_name"][dataset_name]["sort_score"].keys())

                print("已经处理过的win_size:\n", "\n".join(out_result_json["task_name"][dataset_name]["split_win_size"].keys()), "\n==========================")
                for win_size in size_list:

                    print("当前win_size：", win_size)
                    if str(win_size) in [str(it) for it in out_result_json["task_name"][dataset_name]["split_win_size"].keys()]:
                        flag = False
                        
                        if "chunk_win_size" in out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)].keys():
                            print(out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)]["chunk_win_size"].keys())
                            for it in [itt for itt in size_list if int(itt) >= int(win_size)]:
                                if str(it) not in [str(itt) for itt in out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)]["chunk_win_size"].keys()]:
                                    flag = True
                            if flag:
                                print("还有数据没跑完。")
                            else:
                                print(win_size, "已存在")
                                continue
                    
                    out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)] = {}

                    model_dim = out_result_json["model_dim"]
                    if model_dim % win_size != 0:
                        print(model_dim % win_size)
                        break
                    chunks_result_score_list = []
                    for chunk_id in range(int(model_dim / win_size)):
                        
                        # print('start chunk_id ......')

                        main_score = main(model, 
                                            dataset_name,
                                            batch_size=batch_size, 
                                            emb_len=emb_len, 
                                            seed=seed,
                                            win_size=win_size, 
                                            chunk_ids=[chunk_id], 
                                            dtype=dtype)
                        
                        # 保存一下当前chunk_id的结果
                        chunks_result_score_list.append(main_score)

                        print('----------------{}--{}--{}-----------------'.format(dataset_name, chunk_id, main_score))

                    print('===========================================')

                    out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)]["chunk_result"] = chunks_result_score_list


                    sorted_index_list = [index for index, value in sorted(enumerate(chunks_result_score_list), key=lambda x: x[1], reverse=True)]
                    # print(sorted_index_list)




                    # 比较精挑细选和直接截取的差别
                    out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)]["chunk_win_size"] = {}
                    for max_len in size_list:
                        if max_len >= win_size:
                        

                            list_size = int(max_len/win_size)

                            # list_size = 16
                            if list_size*win_size < model_dim:
                                out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)]["chunk_win_size"][str(max_len)] = {}
                            
                                


                                # 顺序截取结果
                                out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)]["chunk_win_size"][str(max_len)]["sort_score"] = {
                                                    "main_score": out_result_json["task_name"][dataset_name]["sort_score"][str(max_len)]
                                                }
                                


                               # 加上随机选取的结果
                                
                                out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)]["chunk_win_size"][str(max_len)]["random_score"] = {
                                                    "main_score": np.mean(out_result_json["task_name"][dataset_name]["random_score"][str(max_len)])
                                                }



                                main_score = main(model, 
                                                    dataset_name,
                                                    batch_size=batch_size, 
                                                    emb_len=emb_len, 
                                                    seed=seed,
                                                    win_size=win_size, 
                                                    chunk_ids=sorted_index_list[:list_size], 
                                                    dtype=dtype)
                                
                                print('---------------精挑细选--{}---{}-------------'.format(list_size*win_size, main_score))
                                
                                
                                out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)]["chunk_win_size"][str(max_len)]["head_score"] = {
                                                    "main_score": main_score
                                                }


                                main_score = main(model, 
                                                    dataset_name,
                                                    batch_size=batch_size, 
                                                    emb_len=emb_len, 
                                                    seed=seed,
                                                    win_size=win_size, 
                                                    chunk_ids=sorted_index_list[-list_size:], 
                                                    dtype=dtype)
                                
                                print('---------------最差结果--{}---{}-------------'.format(list_size*win_size, main_score))

                                out_result_json["task_name"][dataset_name]["split_win_size"][str(win_size)]["chunk_win_size"][str(max_len)]["end_score"] = {
                                                    "main_score": main_score
                                                }

                                
                                print('==================================================')

                                with open(analyze_path + "/{}.json".format(model_name), 'w', encoding='utf-8') as file:
                                    # 确保json格式是美化打印的，并确保中文能被正确写入
                                    json.dump(out_result_json, file, ensure_ascii=False, indent=4)
                    

                    print('====================================================================')
            
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

            # del model
            # import gc
            # gc.collect()
            # # 清理未使用的缓存内存
            # torch.cuda.empty_cache()

        
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


# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/llama/bin/python "/home/linkco/exa/Useful-Embedding/10 rank chunk mteb.py"  "/home/linkco/exa/models/roberta-large"

# CUDA_VISIBLE_DEVICES=1 /home/linkco/anaconda3/envs/llama/bin/python "/home/linkco/exa/Useful-Embedding/10 rank chunk mteb.py"  "/home/linkco/exa/models/gtr-t5-large /home/linkco/exa/models/instructor-large /home/linkco/exa/models/roberta-large-InBedder /home/linkco/exa/models/roberta-large"

# CUDA_VISIBLE_DEVICES=1 /home/linkco/anaconda3/envs/llama/bin/python "/home/linkco/exa/Useful-Embedding/10 rank chunk mteb.py"  "/home/linkco/exa/models/bart-base /home/linkco/exa/models/bge-m3 /home/linkco/exa/models/gte-base"

# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/llama/bin/python "/home/linkco/exa/Useful-Embedding/10 rank chunk mteb.py"  "/home/linkco/exa/models/stella_en_400M_v5 /home/linkco/exa/models/stella_en_400M_v5-GEDI/epoch_3 /home/linkco/exa/models/gte-large-en-v1.5"

# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/llama/bin/python "/home/linkco/exa/Useful-Embedding/10 rank chunk mteb.py"  "/home/linkco/exa/models/mxbai-embed-large-v1 /home/linkco/exa/models/Qwen3-Embedding-0.6B /home/linkco/exa/models/gte-Qwen2-1.5B-instruct"


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