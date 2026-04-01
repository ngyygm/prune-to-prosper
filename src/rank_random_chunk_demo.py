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
    
    def __init__(self, model_path, device='cuda:0'):
        
        self.model = SentenceTransformer(model_path, trust_remote_code=True).to(device)
        
        self.device = device
    
    def get_dim(self):

        return len(self.model.encode("hello"))
        
    def encode(self, input_texts, max_length=1024):
        embedings = self.model.encode(input_texts, max_length=max_length)


        return embedings


def get_texts_embeddings(texts, model, batch_size=32, emb_len=64, seed=42, win_size=64, chunk_id=0, type='def'):
    if type=='def':
        print('def')
        return get_texts_embeddings_def(texts, model, batch_size=batch_size)
    elif type=='random':
        print('random')
        return get_texts_embeddings_random(texts, model, batch_size=batch_size, emb_len=emb_len, seed=seed)
    elif type=='chunk':
        print('chunk')
        return get_texts_embeddings_chunk(texts, model, batch_size=batch_size, win_size=win_size, chunk_id=chunk_id)


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

def get_texts_embeddings_chunk(texts, model, batch_size=32, win_size=64, chunk_id=0):
    """
    对文本进行批处理编码，并对每个生成的embedding向量进行切片，保留一个连续的片段。

    Args:
        texts (list): 字符串列表，包含所有待编码的文本。
        model: 任何具有 .encode() 方法的编码模型。
        batch_size (int): 每批处理的文本数量。
        win_size (int): 切片的窗口大小，即要保留的连续片段的长度。
        chunk_id (int): 要保留的片段的索引。例如，如果win_size=64，原始维度=768，
                        则chunk_id=0保留 [0:64], chunk_id=1保留 [64:128], 以此类推。

    Returns:
        np.ndarray: 一个NumPy数组，形状为 (len(texts), win_size)，包含了所有切片后的embedding。
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Chunk Embeddings"):
        batch_texts = texts[i:i+batch_size]
        # model.encode 返回一个形状为 (batch_size, original_dim) 的数组
        embeddings = model.encode(batch_texts)

        # --- 实现开始 ---
        # 1. 获取原始embedding的维度
        original_dim = embeddings.shape[1]
        
        # 2. 计算切片的起始和结束位置
        start_index = chunk_id * win_size
        end_index = start_index + win_size
        
        # 3. 检查切片范围是否有效
        if end_index > original_dim:
            print(f"Warning: Chunk range [{start_index}:{end_index}] is out of bounds for original dimension ({original_dim}).")
            # 如果越界，可以采取几种策略：
            # a) 报错 (推荐，让调用者意识到问题)
            raise IndexError(f"Calculated end index {end_index} is out of bounds for embedding dimension {original_dim}.")
            # b) 返回空数组或截断到末尾
            # end_index = original_dim
            # if start_index >= original_dim:
            #     processed_embeddings = np.empty((embeddings.shape[0], 0)) # 返回空特征
            # else:
            #     processed_embeddings = embeddings[:, start_index:end_index]
        else:
            # 4. 使用NumPy切片操作获取连续片段
            # embeddings[:, start_index:end_index] 的意思是：
            # 对所有行 (:)，选取从 start_index 到 end_index (不包含) 的列
            # 结果形状: (batch_size, win_size)
            processed_embeddings = embeddings[:, start_index:end_index]
        # --- 实现结束 ---

        all_embeddings.append(processed_embeddings)

    # Concatenate all batches
    if all_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        all_embeddings = np.array([])
        
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


def main(model, dataset_name, recall_len, device):

    
    batch_size = 32
    # 使用随机的参数
    emb_len=64
    seed=66
    # 使用chunk的参数
    win_size=64
    chunk_id=0
    # 类型
    type='chunk'

    embedding_dim = len(get_texts_embeddings(["hello"],
                                              model, 
                                              batch_size=batch_size, 
                                              emb_len=emb_len, 
                                              seed=seed,
                                              win_size=win_size, 
                                              chunk_id=chunk_id, 
                                              type=type)[0])
    print(embedding_dim)
    # FAISS index initialization
    faiss_index = faiss.IndexFlatL2(embedding_dim)

    corpus_path = f"/home/linkco/exa/datas/{dataset_name}/{dataset_name}_text.json"
    queries_path = f"/home/linkco/exa/datas/{dataset_name}/{dataset_name}_queries.json"
    qrels_path = f"/home/linkco/exa/datas/{dataset_name}/{dataset_name}_qrels.json"

    corpus = read_json(corpus_path)
    queries = list2dict(read_json(queries_path))
    qrels = list2dict(read_json(qrels_path))

    corpus_id_list = []
    corpus_id_text_list = []
    for corpus_id in corpus.keys():
        corpus_id_text_list.append(corpus[corpus_id]['text'])
        # add_item(corpus[corpus_id]['text'], model, tokenizer, faiss_index, device)
        corpus_id_list.append(corpus_id)

    corpus_embeddings = get_texts_embeddings(corpus_id_text_list, model, batch_size=batch_size, emb_len=emb_len, seed=seed, win_size=win_size, chunk_id=chunk_id, type=type)
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
    
    querys_embeddings = get_texts_embeddings(query_id_text_list, model, batch_size=batch_size, emb_len=emb_len, seed=seed, win_size=win_size, chunk_id=chunk_id, type=type)

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

    print(f'Data has been written to {csv_filename}')



# Example usage
if __name__ == "__main__":

    model_list = [
        # '/home/linkco/exa/models/bart-base',
        # '/home/linkco/exa/models/bge-m3',
        # '/home/linkco/exa/models/gte-base',
        '/home/linkco/exa/models/jina-embeddings-v3',
        # '/home/linkco/exa/models/gtr-t5-large',
        # '/home/linkco/exa/models/instructor-large',
        # '/home/linkco/exa/models/roberta-large-InBedder',

        # '/home/linkco/exa/models/stella_en_400M_v5',
        # '/home/linkco/exa/models/stella_en_400M_v5-inbedder/epoch_3',
        # '/home/linkco/exa/models/stella_en_400M_v5-medi/epoch_3',
        # '/home/linkco/exa/models/stella_en_400M_v5-GEDI/epoch_3',
        # '/home/linkco/exa/models/gte-large-en-v1.5',
        # '/home/linkco/exa/models/gte-large-en-v1.5-inbedder/epoch_3',
        # '/home/linkco/exa/models/gte-large-en-v1.5-medi/epoch_3',
        # '/home/linkco/exa/models/gte-large-en-v1.5-GEDI/epoch_3',
        # '/home/linkco/exa/models/Qwen3-Embedding-0.6B',
        # '/home/linkco/exa/models/Qwen3-Embedding-0.6B-inbedder/epoch_3',
        # '/home/linkco/exa/models/Qwen3-Embedding-0.6B-medi/epoch_3',
        # '/home/linkco/exa/models/Qwen3-Embedding-0.6B-GEDI/epoch_3',
        # '/home/linkco/exa/models/roberta-large',
        # '/home/linkco/exa/models/roberta-large-inbedder/epoch_3',
        # '/home/linkco/exa/models/roberta-large-medi/epoch_3',
        # '/home/linkco/exa/models/roberta-large-GEDI/epoch_3',
        # '/home/linkco/exa/models/mxbai-embed-large-v1',
        # '/home/linkco/exa/models/mxbai-embed-large-v1-inbedder/epoch_3',
        # '/home/linkco/exa/models/mxbai-embed-large-v1-medi/epoch_3',
        # '/home/linkco/exa/models/mxbai-embed-large-v1-GEDI/epoch_3',

        
        # '/home/linkco/exa/models/stella_en_400M_v5-GEDI/epoch_1',
        # '/home/linkco/exa/models/stella_en_400M_v5-GEDI/epoch_2',
    ]
    
    

    # print(len(model_list))
    for model_path in model_list:
        model_name = model_path.replace('/home/linkco/exa/models/', '').replace('/', '-')
        print(f"Model: {model_name}")
        eval_result = {}
        eval_result[model_name] = {}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        model = MyModel(model_path, device)
        # 'Instruction', 'raceQA', 'raceFind', 'fever'
        for dataset_name in ['raceFind']:


            recall_count = main(model, dataset_name, 100, device)

            print('------------------{}-------------------'.format(dataset_name))
            result_str_list = []
            for recall_id in [1, 3, 10]:
                eval_result[model_name]["{} @{}".format(dataset_name, recall_id)] = recall_count[recall_id]*100
                result_str_list.append("{}: {:.2f}".format(recall_id, recall_count[recall_id]*100))

            print("\t".join(result_str_list))
            write_result_csv(eval_result, csv_filename='/home/linkco/exa/LongEmbed/output/IRS_result.csv')
        print('===========================================')
        
        del model
        # 清理未使用的缓存内存
        torch.cuda.empty_cache()


# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/omo10/bin/python "/home/linkco/exa/LongEmbed/42 baseline self.py"