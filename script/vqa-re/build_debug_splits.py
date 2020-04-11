import numpy as np
from tqdm import tqdm
import json



process_question_paths = [
    "../../data/re-vqa/data/revqa_train_proc.json",
    "../../data/re-vqa/data/revqa_val_proc.json"
]

answer_paths = [
    "../../datasets/VQA/cache/revqa_train_target.pkl",
    "../../datasets/VQA/cache/revqa_val_target.pkl",
    "../../datasets/VQA/cache/val_target.pkl"
]



debug_process_question_paths = [
    "../../data/re-vqa/data/debug_revqa_train_proc.json",
    "../../data/re-vqa/data/revqa_val_proc.json"
]

debug_answer_paths = [
    "../../datasets/VQA/cache/debug_revqa_train_target.pkl",
    "../../datasets/VQA/cache/debug_revqa_val_target.pkl",
    "../../datasets/VQA/cache/debug_val_target.pkl"
]



