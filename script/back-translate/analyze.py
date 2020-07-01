import pandas as pd
from collections import defaultdict
from copy import deepcopy

import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import _pickle as cPickle


que_split_path_dict = {
    "minval": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_minval2014_questions.pkl",
    "train": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_train2014_questions.pkl",
    "val":  "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
    "test": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_test2015_questions.pkl",
    "trainval": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_trainval2014_questions.pkl",
}

dump_path_dict = {
    "minval": "../../datasets/VQA/back-translate/analyze_org2_bt_v2_OpenEnded_mscoco_minval2014_questions.pkl",
    "train": "../../datasets/VQA/back-translate/analyze_org2_bt_v2_OpenEnded_mscoco_train2014_questions.pkl",
    "val": "../../datasets/VQA/back-translate/analyze_org2_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
    "test": "../../datasets/VQA/back-translate/analyze_org2_bt_v2_OpenEnded_mscoco_test2015_questions.pkl",
    "trainval": "../../datasets/VQA/back-translate/analyze_org2_bt_v2_OpenEnded_mscoco_trainval2014_questions.pkl",
}

# get all languages
lang_seqs = np.load("lang_seqs.npy", allow_pickle=True)
all_langs = list(np.concatenate(lang_seqs))
df = pd.DataFrame(columns=["org_question_id"] + all_langs)

for split in ["minval", "train", "val", "test", "trainval"]:
    print(f"Processing: {split}")
    data = cPickle.load(open(que_split_path_dict[split], "rb"))
    questions = data["questions"]

    for row in tqdm(questions):
        langs = []
        values = [row[0]["question_id"]]
        for idx, item in enumerate(row):
            langs.extend(item["languages"])
            value = [(item["sim_score"], idx+1)] * len(item["languages"])
            values.extend(value)
        _df_row = pd.DataFrame([values], columns=["org_question_id"] + langs)
        df = df.append(_df_row)

    last_col = []
    col_names = []
    for col_name in df.columns:
        if col_name == "org_question_id":
            continue
        col = df[col_name]

        # remove NAN and empty tuples
        col = col[col.notna()]
        col = [item for item in col if item != ()]

        if len(col) == 0:
            continue

        sim_scores, ranks = list(zip(*col))
        avg_sim_score, avg_rank = sum(sim_scores)/len(sim_scores), sum(ranks)/len(ranks)
        last_col.append((avg_sim_score, avg_rank))
        col_names.append(col_name)

    last_row = pd.DataFrame([last_col], columns=col_names)
    df = df.append(last_row)

    # sort based on descending similarity
    # filter NAN
    filtered_inds = df.iloc[-1][df.iloc[-1].notna()]
    # create tuple of language w/ sim-rank
    fil_langs = list(filtered_inds.keys())
    df = df[fil_langs]
    last_row = [x[0] for x in list(df.iloc[-1])]
    sorted_inds = np.argsort(last_row)
    sorted_langs = np.array(fil_langs)[sorted_inds]
    sorted_langs = np.flip(sorted_langs)
    df = df.reindex(sorted_langs, axis=1)

    dump_path = dump_path_dict[split]
    # df.to_pickle(dump_path)
    print(df.iloc[-1])
    print("-"*80)
    # print(f"Dumped: {dump_path}")
