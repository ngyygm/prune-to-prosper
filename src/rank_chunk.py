import os

import argparse

import faiss
import torch

import csv
import json

import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


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

class MyModel():
    def __init__(self, modelpath, device='cuda:0'):
        self.model = SentenceTransformer(modelpath, trust_remote_code=True).to(device)
        self.device = device
        self.cache = {}  # 添加缓存字典

    def get_dim(self):
        return len(self.model.encode("hello"))

    def encode(self, inputtexts, maxlength=1024):
        # 初始化嵌入列表和待编码文本列表
        embeddings = []
        texts_to_encode = []

        # 遍历输入文本，检查缓存
        for text in inputtexts:
            if text in self.cache:
                # 如果在缓存中，直接添加到嵌入列表
                embeddings.append(self.cache[text])
            else:
                # 如果不在缓存中，添加到待编码文本列表
                texts_to_encode.append(text)

        # 如果有待编码的文本，进行批量编码
        if texts_to_encode:
            new_embeddings = self.model.encode(texts_to_encode, max_length=maxlength)
            # 将新编码的嵌入存入缓存，并添加到嵌入列表
            for text, embedding in zip(texts_to_encode, new_embeddings):
                self.cache[text] = embedding
                embeddings.append(embedding)

        return np.array(embeddings)  # 将列表转换为NumPy数组



def get_texts_embeddings(texts, model, batch_size=32, emb_len=64, seed=42, win_size=64, chunk_ids=[0], type='def'):
    if type=='def':
        # print('def')
        return get_texts_embeddings_def(texts, model, batch_size=batch_size)
    elif type=='random':
        # print('random')
        return get_texts_embeddings_random(texts, model, batch_size=batch_size, emb_len=emb_len, seed=seed)
    elif type=='chunk':
        # print('chunk')
        return get_texts_embeddings_chunk(texts, model, batch_size=batch_size, win_size=win_size, chunk_ids=chunk_ids)


def get_texts_embeddings_def(texts, model, batch_size=32):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        embedings = model.encode(batch_texts)
        # Append to the result list
        all_embeddings.append(embedings)

    # Concatenate all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings



# --- 第一个函数：随机抽取参数保留 ---

def get_texts_embeddings_random(texts, model, batch_size=32, emb_len=64, seed=42):
    """
    对文本进行批处理编码，并对每个生成的embedding向量进行随机降维。

    Args:
        texts (list): 字符串列表，包含所有待编码的文本。
        model: 任何具有 .encode() 方法的编码模型（如Sentence-Transformer）。
        batch_size (int): 每批处理的文本数量。
        emb_len (int): 随机抽取后要保留的embedding维度。

    Returns:
        np.ndarray: 一个NumPy数组，形状为 (len(texts), emb_len)，包含了所有降维后的embedding。
    """
    # 设置随机种子
    np.random.seed(seed)
    # 生成固定的抽取索引
    indices = np.random.choice(model.get_dim(), emb_len, replace=False)


    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Random Embeddings"):
        batch_texts = texts[i:i+batch_size]
        # model.encode 返回一个形状为 (batch_size, original_dim) 的数组
        embeddings = model.encode(batch_texts)
        embeddings = embeddings[:, indices]
        # --- 实现结束 ---

        all_embeddings.append(embeddings)

    # Concatenate all batches
    if all_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        all_embeddings = np.array([])
        
    return all_embeddings


# --- 第二个函数：按窗口切片，保留特定片段 ---

def get_texts_embeddings_chunk(texts, model, batch_size=32, win_size=64, chunk_ids=[0]):
    """
    对文本进行批处理编码，并对每个生成的embedding向量进行多块切片，然后将这些切片拼接在一起。

    Args:
        texts (list): 字符串列表，包含所有待编码的文本。
        model: 任何具有 .encode() 方法的编码模型。
        batch_size (int): 每批处理的文本数量。
        win_size (int): 每个切片的窗口大小。
        chunk_ids (list): 一个整数列表，指定要提取哪些片段并进行拼接。
                          例如，win_size=64, chunk_ids=[0, 2] 会提取第0块 [0:64] 和
                          第2块 [128:192]，并将它们拼接成形状为 (..., 128) 的向量。

    Returns:
        np.ndarray: 一个NumPy数组，形状为 (len(texts), len(chunk_ids) * win_size)，
                    包含了所有拼接后的embedding。
    """
    # --- 输入校验 ---
    if not chunk_ids:
        print("Warning: chunk_ids list is empty. Returning empty array.")
        return np.array([])
    
    # 计算最终输出每个向量的维度
    final_emblen = len(chunk_ids) * win_size

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # model.encode 返回一个形状为 (batch_size, original_dim) 的数组
        embeddings = model.encode(batch_texts)

        # --- 实现开始 ---
        # 1. 获取原始embedding的维度
        original_dim = embeddings.shape[1]
        
        # 2. 检查所有请求的chunk是否都在有效范围内
        max_requested_index = max(chunk_ids)
        max_possible_end_index = (max_requested_index + 1) * win_size
        
        if max_possible_end_index > original_dim:
            raise IndexError(
                f"Requested chunk_id {max_requested_index} (ending at index {max_possible_end_index}) "
                f"is out of bounds for embedding dimension {original_dim}."
            )

        # 3. 使用列表推导式和np.hstack进行高效的切片和拼接
        # 对于每个chunk_id，我们切出对应的片段 embeddings[:, id*win_size : (id+1)*win_size]
        # 这会生成一个列表，其中每个元素都是一个形状为 (batch_size, win_size) 的数组
        # 例如: [embeddings[:, 0:64], embeddings[:, 128:192]]
        
        chunk_slices = [
            embeddings[:, c_id * win_size : (c_id + 1) * win_size]
            for c_id in chunk_ids
        ]
        
        # 4. 使用 np.hstack 沿着水平方向（列方向）将所有切片拼接起来
        # hstack 会将 (batch_size, 64) 和 (batch_size, 64) 拼接成 (batch_size, 128)
        processed_embeddings = np.hstack(chunk_slices)
        
        # --- 实现结束 ---

        all_embeddings.append(processed_embeddings)

    # Concatenate all batches
    if all_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        # 如果输入texts为空，则返回一个形状为 (0, final_emblen) 的空数组
        all_embeddings = np.empty((0, final_emblen))
        
    return all_embeddings




# Functions for CRUD operations
def add_item(embedding, faiss_index):
    # Add to FAISS
    faiss_index.add(np.array([embedding], dtype=np.float32))
    # print(f"Item added: {text} (using {'CLS' if use_cls else 'SEP'} vector)")

def search(query_embedding, faiss_index, top_k=5):
    # Perform search in FAISS
    distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), top_k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        results.append({"index": idx, "distance": dist})
    return results



def _read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
def _read_json_by_lines(file_path):
    filtered_qrels = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a separate JSON object
            try:
                json_object = json.loads(line)
                filtered_qrels.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
    return filtered_qrels

def read_json(file_path):
    try:
        return _read_json(file_path)
    except:
        return _read_json_by_lines(file_path)
    
def list2dict(data):
    dict_data = {}
    for item in data:
        for k,v in item.items():
            dict_data[k] = v
    return dict_data


def main(model,
        dataset_name, 
        recall_len, 
        device, 
        batch_size=32, 
        emb_len=64, 
        seed=42,
        win_size=64, 
        chunk_ids=[0], 
        type="chunk"):



    embedding_dim = len(get_texts_embeddings(["hello"],
                                              model, 
                                              batch_size=batch_size, 
                                              emb_len=emb_len, 
                                              seed=seed,
                                              win_size=win_size, 
                                              chunk_ids=chunk_ids, 
                                              type=type)[0])
    # print(embedding_dim)
    # FAISS index initialization
    faiss_index = faiss.IndexFlatL2(embedding_dim)

    base_path = "/home/linkco/exa/datas"

    corpus_path = base_path + f"/{dataset_name}/{dataset_name}_text.json"
    queries_path = base_path + f"/{dataset_name}/{dataset_name}_queries.json"
    qrels_path = base_path + f"/{dataset_name}/{dataset_name}_qrels.json"

    corpus = read_json(corpus_path)
    queries = list2dict(read_json(queries_path))
    qrels = list2dict(read_json(qrels_path))

    corpus_id_list = []
    corpus_id_text_list = []
    for corpus_id in corpus.keys():
        corpus_id_text_list.append(corpus[corpus_id]['text'])
        # add_item(corpus[corpus_id]['text'], model, tokenizer, faiss_index, device)
        corpus_id_list.append(corpus_id)

    corpus_embeddings = get_texts_embeddings(corpus_id_text_list, model, batch_size=batch_size, emb_len=emb_len, seed=seed, win_size=win_size, chunk_ids=chunk_ids, type=type)
    for emb in corpus_embeddings:
        add_item(emb, faiss_index)
    
    # print(search("test document", tokenizer, model, device, top_k=2))

    recall_count = {}

    for i in range(recall_len):
        recall_count[i + 1] = 0

    query_id_list = []
    query_id_text_list = []
    for query_id in queries.keys():
        query_id_list.append(query_id)
        query_id_text_list.append(queries[query_id])
    
    querys_embeddings = get_texts_embeddings(query_id_text_list, model, batch_size=batch_size, emb_len=emb_len, seed=seed, win_size=win_size, chunk_ids=chunk_ids, type=type)

    for idx in range(len(querys_embeddings)):
        query_id = query_id_list[idx]
        que_emb = querys_embeddings[idx]
        result = search(que_emb, faiss_index, top_k=recall_len)

        temp_corpus_id_list = [corpus_id_list[it['index']] for it in result]

        for recall_rate in recall_count.keys():
            flag = False
            for tar_corpus_id in qrels["{}".format(query_id)].keys():
                if tar_corpus_id in temp_corpus_id_list[:recall_rate]:
                    flag = True
            
            if flag:
                recall_count[recall_rate] += 1


    for recall_rate in recall_count.keys():
        recall_count[recall_rate] = recall_count[recall_rate] / len(queries.keys())
    
    return recall_count


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



# Example usage
if __name__ == "__main__":

    model_list = [
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
        '/home/linkco/exa/models/Qwen3-Embedding-0.6B',
        '/home/linkco/exa/models/roberta-large',
        '/home/linkco/exa/models/mxbai-embed-large-v1',
    ]
    
    
    # print(len(model_list))
    for model_path in model_list:
        
        model_name = model_path.replace('/home/linkco/exa/models/', '').replace('/', '-')
        print(f"Model: {model_name}", "/home/linkco/exa/Useful-Embedding/data/analyze/{}.json".format(model_name))

        if os.path.exists("/home/linkco/exa/Useful-Embedding/data/analyze/{}.json".format(model_name)):
            print(model_name, "已存在")
            with open("/home/linkco/exa/Useful-Embedding/data/analyze/{}.json".format(model_name), "r", encoding="utf-8") as f:
                out_result_json = json.load(f)
        else:
            out_result_json = {}
            out_result_json["model_name"] = model_name
            out_result_json["task_name"] = {}

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 'Instruction', 'raceQA', 'raceFind', 'fever'
        for dataset_name in ['Instruction', 'raceQA', 'raceFind', 'FEVER', 'FiQA', 'NQ', 
                          'Quora', 'SCIDOCS', 'SCIFACT']:
            
            if dataset_name in out_result_json["task_name"].keys():
                print(dataset_name, "已存在")
                continue

            # if not os.path.exists("/home/linkco/exa/Useful-Embedding/data/output/{}".format(dataset_name)):
            #     continue
            
            model = MyModel(model_path, device)
            
            out_result_json["model_dim"] = model.get_dim()
        

            out_result_json["task_name"][dataset_name] = {
                    "defult_score": {
                        "main_score": 0
                    },
                    "split_win_size": {

                    }
                }
                
    
            batch_size = 8
            # 使用随机的参数
            emb_len=64
            seed=66
            # 使用chunk的参数
            win_size=64
            # chunk_id=0
            # 类型
            type='chunk'

            for win_size in [8, 16, 32, 64, 128, 256, 512]:
                
                out_result_json["task_name"][dataset_name]["split_win_size"][win_size] = {}

                model_dim = model.get_dim()
                if model_dim % win_size != 0:
                    print(model_dim % win_size)
                    break
                chunks_result_score_list = []
                for chunk_id in range(int(model_dim / win_size)):


                    recall_count = main(model, 
                                        dataset_name, 
                                        100, 
                                        device, 
                                        batch_size=batch_size, 
                                        emb_len=emb_len, 
                                        seed=seed,
                                        win_size=win_size, 
                                        chunk_ids=[chunk_id], 
                                        type=type)
                    
                    # 保存一下当前chunk_id的结果
                    chunks_result_score_list.append(recall_count[1]*100)

                    print('------------------{}--{}-----------------'.format(dataset_name, chunk_id))

                    result_str_list = []
                    for recall_id in [1, 3, 10]:
                        result_str_list.append("{}: {:.2f}".format(recall_id, recall_count[recall_id]*100))
                    
                    print("\t".join(result_str_list))
                print('===========================================')

                out_result_json["task_name"][dataset_name]["split_win_size"][win_size]["chunk_result"] = chunks_result_score_list


                sorted_index_list = [index for index, value in sorted(enumerate(chunks_result_score_list), key=lambda x: x[1], reverse=True)]
                # print(sorted_index_list)


                # 原始结果
                print('---------------原始结果--------------', range(int(model_dim / win_size)))
                recall_count = main(model, 
                                        dataset_name, 
                                        100, 
                                        device, 
                                        batch_size=batch_size, 
                                        emb_len=emb_len, 
                                        seed=seed,
                                        win_size=win_size, 
                                        chunk_ids=range(int(model_dim / win_size)), 
                                        type=type)
                
                result_str_list = []
                for recall_id in [1, 3, 10]:
                    result_str_list.append("{}: {:.2f}".format(recall_id, recall_count[recall_id]*100))
                
                out_result_json["task_name"][dataset_name]["defult_score"]["main_score"] = recall_count[1]*100

                print("\t".join(result_str_list))


                # 比较精挑细选和直接截取的差别
                out_result_json["task_name"][dataset_name]["split_win_size"][win_size]["chunk_win_size"] = {}
                for max_len in [8, 16, 32, 64, 128, 256, 512, 1024]:
                    if max_len >= win_size:
                    

                        list_size = int(max_len/win_size)

                        # list_size = 16
                        if list_size*win_size <= model_dim:
                            out_result_json["task_name"][dataset_name]["split_win_size"][win_size]["chunk_win_size"][max_len] = {}
                        
                            print('---------------直接截取--{}----------------'.format(list_size*win_size))
                            recall_count = main(model, 
                                                    dataset_name, 
                                                    100, 
                                                    device, 
                                                    batch_size=batch_size, 
                                                    emb_len=emb_len, 
                                                    seed=seed,
                                                    win_size=win_size, 
                                                    chunk_ids=range(list_size), 
                                                    type=type)


                            result_str_list = []
                            for recall_id in [1, 3, 10]:
                                result_str_list.append("{}: {:.2f}".format(recall_id, recall_count[recall_id]*100))
                            
                            out_result_json["task_name"][dataset_name]["split_win_size"][win_size]["chunk_win_size"][max_len]["def_score"] = {
                                                "main_score": recall_count[1]*100
                                            }

                            print("\t".join(result_str_list))
                            


                            print('---------------精挑细选--{}----------------'.format(list_size*win_size))
                            recall_count = main(model, 
                                                    dataset_name, 
                                                    100, 
                                                    device, 
                                                    batch_size=batch_size, 
                                                    emb_len=emb_len, 
                                                    seed=seed,
                                                    win_size=win_size, 
                                                    chunk_ids=sorted_index_list[:list_size], 
                                                    type=type)
                            
                            result_str_list = []
                            for recall_id in [1, 3, 10]:
                                result_str_list.append("{}: {:.2f}".format(recall_id, recall_count[recall_id]*100))
                            
                            out_result_json["task_name"][dataset_name]["split_win_size"][win_size]["chunk_win_size"][max_len]["head_score"] = {
                                                "main_score": recall_count[1]*100
                                            }

                            print("\t".join(result_str_list))


                            print('---------------最差结果--{}----------------'.format(list_size*win_size))
                            recall_count = main(model, 
                                                    dataset_name, 
                                                    100, 
                                                    device, 
                                                    batch_size=batch_size, 
                                                    emb_len=emb_len, 
                                                    seed=seed,
                                                    win_size=win_size, 
                                                    chunk_ids=sorted_index_list[-list_size:], 
                                                    type=type)
                            
                            result_str_list = []
                            for recall_id in [1, 3, 10]:
                                result_str_list.append("{}: {:.2f}".format(recall_id, recall_count[recall_id]*100))

                            print("\t".join(result_str_list))

                            out_result_json["task_name"][dataset_name]["split_win_size"][win_size]["chunk_win_size"][max_len]["end_score"] = {
                                                "main_score": recall_count[1]*100
                                            }

                            
                            print('==================================================')

                            with open("/home/linkco/exa/Useful-Embedding/data/analyze/{}.json".format(model_name), 'w', encoding='utf-8') as file:
                                # 确保json格式是美化打印的，并确保中文能被正确写入
                                json.dump(out_result_json, file, ensure_ascii=False, indent=4)



                print('====================================================================')
        
            del model
            # 清理未使用的缓存内存
            torch.cuda.empty_cache()

        
        


# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/omo10/bin/python "/home/linkco/exa/LongEmbed/42 baseline self.py"


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