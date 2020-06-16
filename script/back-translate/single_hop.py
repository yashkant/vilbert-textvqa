import os

import numpy as np
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from transformers.hf_api import HfApi
import json
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seq_id",
    type=int,
    help="Bert pre-trained model selected in the list: bert-base-uncased, "
         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
)
args = parser.parse_args()
seq_id = args.seq_id

split_path_dict = {
    "train": "../../datasets/VQA/v2_OpenEnded_mscoco_train2014_questions.json",
    "val": "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json",
    "test": "../../datasets/VQA/v2_OpenEnded_mscoco_test2015_questions.json"
}

save_path = "../../datasets/VQA/back-translate/"
os.makedirs(save_path, exist_ok=True)
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
torch.backends.cudnn.benchmark = True
n_gpu = torch.cuda.device_count()
print(f"Using GPUs: {n_gpu}")
langs = np.load("lang_seqs.npy", allow_pickle=True)[seq_id]


for split, path in split_path_dict.items():
    questions = json.load(open(path))["questions"]
    question_ids = [que["question_id"] for que in questions]
    questions = [que["question"] for que in questions]

    print(f"Processing split: {split} w/ languages: {langs} w/ seq_id: {seq_id}")
    data = {
        "question_ids": question_ids,
        "en": questions
    }

    for lang in tqdm(langs, "Languages: "):
        model_name_for = f'Helsinki-NLP/opus-mt-en-{lang}'
        model_name_back = f'Helsinki-NLP/opus-mt-{lang}-en'

        try:
            tokenizer_for = MarianTokenizer.from_pretrained(model_name_for)
            tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)

            model_for = MarianMTModel.from_pretrained(model_name_for).to(device)
            model_back = MarianMTModel.from_pretrained(model_name_back).to(device)

        except:
            print(f"Error with {lang}")
            continue

        translated_questions = []
        question_batches = np.array_split(questions, np.ceil(len(questions)/batch_size))

        # Add try-except here, check output!
        for batch in tqdm(question_batches, "Batches: "):
            translated_text = model_for.generate(**tokenizer_for.prepare_translation_batch(batch).to(device), max_length=23)
            translated_text = [tokenizer_for.decode(t, skip_special_tokens=True) for t in translated_text]
            back_translated_text = model_back.generate(**tokenizer_back.prepare_translation_batch(translated_text).to(device), max_length=23)
            back_translated_text = [tokenizer_back.decode(t, skip_special_tokens=True) for t in back_translated_text]
            assert len(back_translated_text) == len(translated_text)
            translated_questions.extend(back_translated_text)
            del translated_text, back_translated_text, batch

        data[lang] = translated_questions
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_path, f"vqa-{split}-{lang}.csv"), encoding='utf-8', index=False)
        print(f'Dumped file: {f"vqa-{split}-{lang}.csv"}')
