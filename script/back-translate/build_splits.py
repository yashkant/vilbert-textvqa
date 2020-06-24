import glob
import os
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seq_id",
    type=int,
    required=True,
    help="Bert pre-trained model selected in the list: bert-base-uncased, "
         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
)
args = parser.parse_args()
seq_id = args.seq_id
max_seqs = 10
assert seq_id < max_seqs

def cos_sim(A, B):
    # A and B -> [N x dim]
    Bt = np.transpose(B)
    AB_mul = np.matmul(A, Bt)
    A_mag = np.linalg.norm(A, axis=1, keepdims=True)
    B_mag = np.linalg.norm(B, axis=1, keepdims=True)
    AB_mag = np.matmul(A_mag, np.transpose(B_mag))
    assert AB_mag.shape == AB_mul.shape
    return AB_mul/AB_mag

split_path_dict = {
    "train": "../../datasets/VQA/v2_OpenEnded_mscoco_train2014_questions.json",
    "val": "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json",
    "test": "../../datasets/VQA/v2_OpenEnded_mscoco_test2015_questions.json"
}
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
torch.backends.cudnn.benchmark = True
n_gpu = torch.cuda.device_count()
print(f"Using GPUs: {n_gpu}")

data_path = "../../datasets/VQA/back-translate/vqa-{}-{}.csv"
org_factor = int(1e10)
sim_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens').to(device)
sim_thresh = 0.7

for split, path in split_path_dict.items():
    questions = json.load(open(path))["questions"]
    question_ids = sorted([que["question_id"] for que in questions])

    print(f"Min: {min(question_ids)}, Max: {max(question_ids)}, Split: {split}")
    langs = np.load("lang_seqs.npy", allow_pickle=True)
    data_files = [data_path.format(split, lang_pair[-1]) for lang_pair in langs]
    dfs = []
    for idx, file in tqdm(enumerate(data_files), "Reading CSVs", total=len(data_files)):
        if not os.path.exists(file):
            print(f"Couldn't find: {file} with seq-id: {idx}")
            continue
        df = pd.read_csv(file)
        dfs.append(df)

    # assert equal lens
    dfs_len = [len(df) for df in dfs]
    assert len(set(dfs_len)) == 1

    # join along y-axis and remove duplicate columns
    data = pd.concat(dfs, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    data.sort_values("question_ids")
    data_dict = {}

    indices = list(range(0, len(data)))
    seqs = np.array_split(indices, 10)
    seq_indices = seqs[seq_id]
    seq_start_index, seq_last_index = seq_indices[0], seq_indices[-1]
    data = data[seq_start_index:seq_last_index+1]
    assert len(data) == len(seq_indices)
    print(f"Processing seq-id: {seq_id}/{len(seqs)-1} w/ start: {seq_start_index}, end: {seq_last_index}")

    # similarity and then thresholding
    for index, row in tqdm(data.iterrows(), total=len(data)):
        # if (index-seq_start_index) == 10:
        #     break

        qid = row["question_ids"]
        ref_question = row["en"]
        del row["question_ids"]
        unique_langs = defaultdict(list)

        # remove duplicates
        for lang, question in row.items():
            unique_langs[question].append(lang)

        batch = list(unique_langs.keys())
        assert ref_question == batch[0]

        batch_embeddings = sim_model.encode(batch)
        sim_scores = cos_sim([batch_embeddings[0]], batch_embeddings).round(2)[0]

        rephrasings = []
        for (question, langs), sim_score in zip(unique_langs.items(), sim_scores):
            if sim_score > sim_thresh:
                item = {
                    "rephrasing": question,
                    "languages": langs,
                    "sim_score": sim_score
                }
                rephrasings.append(item)

        rephrasings = sorted(rephrasings, key=lambda item: item["sim_score"], reverse=True)
        data_dict[qid] = {
            "question": ref_question,
            "question_id": qid,
            "rephrasings_list": rephrasings
        }

    np.save(f"../../datasets/VQA/back-translate/sim-result2/rep_{split}_{seq_id}.npy", data_dict)
    print(f"Dumped file: ../../datasets/VQA/back-translate/sim-result2/rep_{split}_{seq_id}.npy")

