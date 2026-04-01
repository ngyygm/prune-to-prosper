import json
import os
import argparse
from typing import Dict, List, Any
from datasets import load_dataset
from mteb import MTEB

# ---------------------------
# BEIR 风格 Retrieval 数据集处理
# ---------------------------
def process_beir_style_dataset(dataset_name: str, output_dir: str):
    queries = []
    passages = {}
    qrels = []
    
    try:
        print(f"加载 {dataset_name} 数据集的三个子集...")
        corpus_dataset = load_dataset(f'mteb/{dataset_name.lower()}', 'corpus')
        queries_dataset = load_dataset(f'mteb/{dataset_name.lower()}', 'queries')
        qrels_dataset = load_dataset(f'mteb/{dataset_name.lower()}', 'default')
        
        # 文档库
        corpus_split = 'corpus' if 'corpus' in corpus_dataset else list(corpus_dataset.keys())[0]
        for item in corpus_dataset[corpus_split]:
            doc_id = item.get('_id', '')
            title = item.get('title', '')
            text = item.get('text', '')
            if doc_id and text:
                full_text = f"{title} {text}" if title else text
                passages[doc_id] = {"text": full_text}
        
        # 查询
        queries_split = 'queries' if 'queries' in queries_dataset else list(queries_dataset.keys())[0]
        for item in queries_dataset[queries_split]:
            query_id = item.get('_id', '')
            query_text = item.get('text', '')
            if query_id and query_text:
                queries.append({query_id: query_text})
        
        # qrels
        qrels_split = None
        for split in ['test', 'validation', 'dev', 'train']:
            if split in qrels_dataset:
                qrels_split = split
                break
        if qrels_split is None:
            qrels_split = list(qrels_dataset.keys())[0]
            
        qrels_dict = {}
        for item in qrels_dataset[qrels_split]:
            query_id = item.get('query-id', '')
            corpus_id = item.get('corpus-id', '')
            score = item.get('score', 0)
            if query_id and corpus_id and score > 0:
                if query_id not in qrels_dict:
                    qrels_dict[query_id] = {}
                qrels_dict[query_id][corpus_id] = 1
        for query_id, rel_docs in qrels_dict.items():
            qrels.append({query_id: rel_docs})
        
        # 保存
        with open(os.path.join(output_dir, 'queries.json'), 'w') as f:
            json.dump(queries, f, indent=2)
        with open(os.path.join(output_dir, 'passages.json'), 'w') as f:
            json.dump(passages, f, indent=2)
        with open(os.path.join(output_dir, 'qrels.json'), 'w') as f:
            json.dump(qrels, f, indent=2)

        print(f"{dataset_name} 转换完成! queries={len(queries)}, passages={len(passages)}, qrels={len(qrels)}")
        
    except Exception as e:
        print(f"处理 {dataset_name} 数据集时出错: {e}")
        raise


# ---------------------------
# 通用任务处理（非 BEIR Retrieval）
# ---------------------------
def convert_mteb_dataset(dataset_name: str, task_category: str, output_dir: str):
    task = MTEB(tasks=[dataset_name], task_langs=['eng']).tasks[0]
    dataset = task.load_data()

    task = MTEB(tasks=[dataset_name], task_langs=['eng']).tasks[0]
    task.load_data()   # 注意，这里不返回数据
    dataset = task.dataset
    if dataset is None:
        raise ValueError("MTEB load_data 返回 None")
    
    
    split = 'test' if 'test' in dataset else list(dataset.keys())[0]
    data = dataset[split]

    queries, passages, qrels = [], {}, []

    def get_text(item, fields=["text", "sentence", "content", "document"]):
        for f in fields:
            if f in item:
                return item[f]
        # 如果啥都没找到，就直接返回 None
        return None

    if task_category == "Classification":
        for i, item in enumerate(data):
            qid = f"q{i}"
            text = get_text(item)
            if text is None:
                continue
            label = str(item['label'])
            queries.append({qid: text})
            passages[label] = {"text": label}
            qrels.append({qid: {label: 1}})

    elif task_category == "Clustering":
        for i, item in enumerate(data):
            qid = f"q{i}"
            text = item['text']
            label = str(item['label'])
            queries.append({qid: text})
            passages[label] = {"text": label}
            qrels.append({qid: {label: 1}})

    elif task_category == "PairClassification":
        for i, item in enumerate(data):
            qid = f"q{i}"
            text1, text2, label = item['text1'], item['text2'], item['label']
            queries.append({qid: text1})
            docid = f"doc{i}"
            passages[docid] = {"text": text2}
            qrels.append({qid: {docid: int(label)}})

    elif task_category == "Reranking":
        for i, item in enumerate(data):
            qid = f"q{i}"
            query, candidates, labels = item['query'], item['candidates'], item['labels']
            queries.append({qid: query})
            rel_docs = {}
            for j, cand in enumerate(candidates):
                docid = f"doc{i}_{j}"
                passages[docid] = {"text": cand}
                rel_docs[docid] = int(labels[j])
            qrels.append({qid: rel_docs})

    elif task_category == "STS":
        for i, item in enumerate(data):
            qid = f"q{i}"
            text1, text2, score = item['sentence1'], item['sentence2'], float(item['score'])
            queries.append({qid: text1})
            docid = f"doc{i}"
            passages[docid] = {"text": text2}
            qrels.append({qid: {docid: score}})

    elif task_category == "Summarization":
        for i, item in enumerate(data):
            qid = f"q{i}"
            doc, summary = item['document'], item['summary']
            queries.append({qid: doc})
            docid = f"doc{i}"
            passages[docid] = {"text": summary}
            qrels.append({qid: {docid: 1}})

    else:
        print(f"❌ 暂时未支持 {task_category} 任务")
        return

    if len(queries) > 0 and len(passages) > 0 and len(qrels) > 0:
        os.makedirs(output_dir, exist_ok=True)
        # 保存
        with open(os.path.join(output_dir, '{}_queries.json'.format(dataset_name)), 'w') as f:
            json.dump(queries, f, indent=2)
        with open(os.path.join(output_dir, '{}_text.json'.format(dataset_name)), 'w') as f:
            json.dump(passages, f, indent=2)
        with open(os.path.join(output_dir, '{}_qrels.json'.format(dataset_name)), 'w') as f:
            json.dump(qrels, f, indent=2)

    print(f"{dataset_name} 转换完成! queries={len(queries)}, text={len(passages)}, qrels={len(qrels)}")


# ---------------------------
# 主程序入口
# ---------------------------
# task_categories = {
#     "Classification": [
#         'AmazonCounterfactualClassification',
#         'AmazonReviewsClassification', 'Banking77Classification',
#         'EmotionClassification', 'ImdbClassification',
#         'MTOPDomainClassification', 'MTOPIntentClassification',
#         'MassiveIntentClassification', 'MassiveScenarioClassification',
#         'ToxicConversationsClassification', 'TweetSentimentExtractionClassification'
#     ],
#     "Clustering": [
#         'BiorxivClusteringS2S',
#         'MedrxivClusteringS2S',
#         'TwentyNewsgroupsClustering'
#     ],
#     "PairClassification": [
#         'SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus'
#     ],
#     "Reranking": [
#         'AskUbuntuDupQuestions', 'MindSmallReranking', 'SciDocsRR', 'StackOverflowDupQuestions'
#     ],
#     "Retrieval": [
#         'ArguAna', 'CQADupstackEnglishRetrieval',
#         'NFCorpus', 'SCIDOCS', 'SciFact'
#     ],
#     "STS": [
#         'BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15',
#         'STS16', 'STS17', 'STS22', 'STSBenchmark'
#     ],
#     "Summarization": [
#         'SummEval'
#     ],
# }

task_categories = {
    "Retrieval": [
        'CQADupstackEnglishRetrieval',
        'SciFact', 'NQ', 'FEVER', 'HotpotQA', 'FiQA', 'ArguAna', 'Touche2020',
    'Quora', 'DBPedia', 'SCIDOCS', 'NFCorpus'
    ],
}

beir_style_datasets = [
    'NQ', 'FEVER', 'HotpotQA', 'FiQA', 'ArguAna', 'Touche2020',
    'Quora', 'DBPedia', 'SCIDOCS', 'NFCorpus', 'SciFact'
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTEB Converter")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录路径")
    args = parser.parse_args()

    for category, datasets in task_categories.items():
        for dataset in datasets:
            output_path = os.path.join(args.output_dir, dataset)
            if category == "Retrieval" and dataset in beir_style_datasets:
                process_beir_style_dataset(dataset, output_path)
            else:
                convert_mteb_dataset(dataset, category, output_path)


# /home/linkco/anaconda3/envs/llama/bin/python /home/linkco/exa/Useful-Embedding/MTEBconverter.py --output_dir /home/linkco/exa/Useful-Embedding/data/output