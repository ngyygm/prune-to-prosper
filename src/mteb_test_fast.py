import mteb
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict, Optional
import sys
import numpy as np
import os


# Get model path from command-line arguments
model_path = sys.argv[1]
if ' ' in model_path:
    model_list = model_path.split(' ')
else:
    model_list = [model_path]


# Get benchmark tasks grouped by category (original MTEB tasks only)
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
        'AskUbuntuDupQuestions', 'MindSmallReranking', 'SciDocsRR', 'StackOverflowDupQuestions'
    ],
    "Retrieval": [
        'ArguAna', 'CQADupstackEnglishRetrieval',
        'NFCorpus', 'SCIDOCS', 'SciFact'
    ],
    "STS": [
        'BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15',
        'STS16', 'STS17', 'STS22', 'STSBenchmark'
    ],
    "Summarization": [
        'SummEval'
    ],
}




task_categories = {
    "Retrieval": [
        'ArguAna', 
    ],
    "Classification": [
        'AmazonCounterfactualClassification'
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



mteb_name = 'mteb-base-ss'

for model_path in model_list:

    model_name = model_path.replace('/home/linkco/exa/models/', '').replace('/', '-')
    print(f"Evaluating model: {model_name}")

    # Run evaluation by task categories
    for category, task_names in task_categories.items():
        print(f"\n=== Running {category} tasks ===")
        
        for task_name in task_names:
            if not os.path.exists(f"/home/linkco/exa/LongEmbed/{mteb_name}/{model_name}/no_model_name_available/no_revision_available/{task_name}.json"):
                print(f"\n=== Running {task_name} tasks {model_name}======{mteb_name}=====")
                # Get tasks for this category
                
                model = SentenceTransformer(model_path, trust_remote_code=True)


                tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
                
                if not tasks:
                    print(f"No tasks found for category: {category}")
                    continue
                
                # Initialize evaluator for these tasks
                evaluation = mteb.MTEB(tasks=tasks)
                
                # # Run evaluation for this category
                # results = evaluation.run(
                #         model,
                #         verbosity=0,
                #         overwrite_results=False,
                #         save_predictions=False,
                #         output_folder=f"/home/linkco/exa/LongEmbed/{mteb_name}/{model_name}",
                #         encode_kwargs={'batch_size': 16, "max_length": 1024},
                #         save_corpus_embeddings=True,
                #         do_length_ablation=True
                #     )

                batch_size = 16
                flag = True
                while flag:
                    try:
                        # Run evaluation for this category
                        results = evaluation.run(
                        model,
                        verbosity=0,
                        overwrite_results=False,
                        save_predictions=False,
                        output_folder=f"/home/linkco/exa/LongEmbed/{mteb_name}/{model_name}",
                        encode_kwargs={'batch_size': batch_size},
                        save_corpus_embeddings=True,
                        do_length_ablation=True
                        )
                        flag = False
                    except:
                        batch_size = batch_size/2
                        if batch_size < 1:
                            batch_size = 1
                
                # Print summary for this category
                print(f"\n=== {category} tasks completed ===")
                print(f"Evaluated {len(tasks)} tasks in {category} category")

                import gc, torch
                # 释放显存
                del model
                gc.collect()  # 强制回收
                torch.cuda.empty_cache()  # 清理缓存
            else:
                print('====================================\n===============================')


# export HF_ENDPOINT=https://hf-mirror.com

# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/omo10/bin/python "/home/linkco/exa/LongEmbed/MTEB test fast.py" "/home/linkco/exa/models/stella_en_400M_v5-GEDI2/epoch_1 /home/linkco/exa/models/stella_en_400M_v5-GEDI/epoch_1"


# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/omo10/bin/python "/home/linkco/exa/LongEmbed/MTEB test fast.py" "/home/linkco/exa/models/stella_en_400M_v5 /home/linkco/exa/models/stella_en_400M_v5-GEDI/epoch_3 /home/linkco/exa/models/stella_en_400M_v5-medi/epoch_3 /home/linkco/exa/models/stella_en_400M_v5-inbedder/epoch_3 /home/linkco/exa/models/roberta-large /home/linkco/exa/models/roberta-large-GEDI/epoch_3 /home/linkco/exa/models/roberta-large-medi/epoch_3 /home/linkco/exa/models/roberta-large-inbedder/epoch_3 /home/linkco/exa/models/gte-large-en-v1.5 /home/linkco/exa/models/gte-large-en-v1.5-GEDI/epoch_3 /home/linkco/exa/models/gte-large-en-v1.5-medi/epoch_3 /home/linkco/exa/models/gte-large-en-v1.5-inbedder/epoch_3 /home/linkco/exa/models/mxbai-embed-large-v1 /home/linkco/exa/models/mxbai-embed-large-v1-GEDI/epoch_3 /home/linkco/exa/models/mxbai-embed-large-v1-medi/epoch_3 /home/linkco/exa/models/mxbai-embed-large-v1-inbedder/epoch_3 /home/linkco/exa/models/Qwen3-Embedding-0.6B-GEDI/epoch_3 /home/linkco/exa/models/Qwen3-Embedding-0.6B-medi/epoch_3 /home/linkco/exa/models/Qwen3-Embedding-0.6B /home/linkco/exa/models/Qwen3-Embedding-0.6B-inbedder/epoch_3"

# CUDA_VISIBLE_DEVICES=1 /home/linkco/anaconda3/envs/omo10/bin/python "/home/linkco/exa/LongEmbed/MTEB test fast.py" "/home/linkco/exa/models/stella_en_400M_v5-deepseek-aux2-def/epoch_3 /home/linkco/exa/models/roberta-large-deepseek-rank-def/epoch_3 /home/linkco/exa/models/gte-base-deepseek-rank-def/epoch_3 /home/linkco/exa/models/roberta-large /home/linkco/exa/models/roberta-large-InBedder /home/linkco/exa/models/bart-base /home/linkco/exa/models/gtr-t5-large /home/linkco/exa/models/m3e-base /home/linkco/exa/models/bge-m3 /home/linkco/exa/models/gte-base /home/linkco/exa/models/stella_en_400M_v5 /home/linkco/exa/models/gte-large-en-v1.5 /home/linkco/exa/models/mxbai-embed-large-v1 /home/linkco/exa/models/jina-embeddings-v3 /home/linkco/exa/models/bge-m3"



# CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/omo10/bin/python "/home/linkco/exa/LongEmbed/MTEB test fast.py" "/home/linkco/exa/models/roberta-large-medi/epoch_3 /home/linkco/exa/models/roberta-large-gedi/epoch_3 /home/linkco/exa/models/roberta-large-inbedder/epoch_3 /home/linkco/exa/models/stella_en_400M_v5-medi/epoch_3 /home/linkco/exa/models/stella_en_400M_v5-inbedder/epoch_3 /home/linkco/exa/models/gtr-t5-large-gedi/epoch_10 /home/linkco/exa/models/gtr-t5-large-medi/epoch_10 /home/linkco/exa/models/gtr-t5-large-inbedder/epoch_10"

# medi
# # CUDA_VISIBLE_DEVICES=1 /home/linkco/anaconda3/envs/omo10/bin/python "/home/linkco/exa/LongEmbed/MTEB test fast.py" "/home/linkco/exa/models/gtr-t5-large-gedi/epoch_1 /home/linkco/exa/models/gtr-t5-large-medi/epoch_1 /home/linkco/exa/models/gtr-t5-large-inbedder/epoch_1"


# gedi
# CUDA_VISIBLE_DEVICES=1 /home/linkco/anaconda3/envs/omo10/bin/python "/home/linkco/exa/LongEmbed/MTEB test fast.py" "/home/linkco/exa/models/roberta-large-gedi/epoch_3 /home/linkco/exa/models/gtr-t5-large-gedi/epoch_1"


# inbedder
# CUDA_VISIBLE_DEVICES=1 /home/linkco/anaconda3/envs/omo10/bin/python "/home/linkco/exa/LongEmbed/MTEB test fast.py" "/home/linkco/exa/models/roberta-large-inbedder/epoch_3 /home/linkco/exa/models/gtr-t5-large-inbedder/epoch_1"
