from copy import deepcopy
import torch
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from tqdm import tqdm
import _pickle as cPickle

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", required=True, type=int, help="0-3")
    arguments = parser.parse_args()
    return arguments


def cos_sim(A, B):
    # A and B -> [N x dim]
    Bt = np.transpose(B)
    AB_mul = np.matmul(A, Bt)
    A_mag = np.linalg.norm(A, axis=1, keepdims=True)
    B_mag = np.linalg.norm(B, axis=1, keepdims=True)
    AB_mag = np.matmul(A_mag, np.transpose(B_mag))
    assert AB_mag.shape == AB_mul.shape
    return AB_mul/AB_mag

# import pdb
# pdb.set_trace()
args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
torch.backends.cudnn.benchmark = True
n_gpu = torch.cuda.device_count()
print(f"Using GPUs: {n_gpu}")


org_factor = int(1e10)
model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

# decontract, depunctuate
filter_type = "dcp"

# sampling: ["top", "random"]
sampling_type = "top"

# filter_thresh
sim_thresh = 0.8

# num-rephrasings
num_re = 4

cc_rephrasings_path = [
    "/nethome/ykant3/pythia/results/noise-v2-fifth/results/pythia_cycle_consistent_with_failure_prediction/34035/gen_questions_train_88.npy",
    "/nethome/ykant3/pythia/results/noise-v2-fifth/results/pythia_cycle_consistent_with_failure_prediction/34035/gen_questions_88_val_ds.npy",

]

# load all data
cc_qdata = {}
for path in cc_rephrasings_path:
    cc_qdata.update(
        np.load(path, allow_pickle=True).item()
    )

# Dumping Paths
que_split_path_dict = {
    "train": ("../../datasets/VQA/cc-re/cc_v2_OpenEnded_mscoco_train2014_questions_88_split_{}.pkl",
              "../../datasets/VQA/v2_OpenEnded_mscoco_train2014_questions.json",
              "../../datasets/VQA/cc-re/train_cc_embeddings_88_split_{}.npy"),

    "val": ("../../datasets/VQA/cc-re/cc_v2_OpenEnded_mscoco_val2014_questions_88_split_{}.pkl",
            "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json",
            "../../datasets/VQA/cc-re/val_cc_embeddings_88_split_{}.npy"),

}


ans_split_path_dict = {
    "train": ("../../datasets/VQA/cc-re/cc_train_target_88_split_{}.pkl",
              "../../datasets/VQA/cache/train_target.pkl"),
    "val": ("../../datasets/VQA/cc-re/cc_val_target_88_split_{}.pkl",
              "../../datasets/VQA/cache/val_target.pkl"),
}

num_splits = 4
data_split = args.data_split

for split in ["val"]:

    print(f"Processing {split}, data split: {data_split}/{num_splits}")
    dump_qdict = {}
    question_embeddings_dict = {}

    qlist, alist = [], []
    cc_qpath, org_qpath, embeddings_path = que_split_path_dict[split][0].format(data_split), que_split_path_dict[split][1], que_split_path_dict[split][2].format(data_split)
    cc_apath = ans_split_path_dict[split][0].format(data_split)

    org_ans = cPickle.load(open(ans_split_path_dict[split][1], "rb"))
    org_qdata = json.load(open(org_qpath))["questions"]
    org_zip_data = list(zip(org_qdata, org_ans))
    org_data_split = np.array_split(org_zip_data, num_splits)[data_split]

    for que, ans in tqdm(org_data_split, total=len(org_data_split)):

        qid = que["question_id"]
        cc_item = cc_qdata[qid]
        # Remove duplicates
        gen_questions = list(set([i["generated_questions"] for i in cc_item]))
        # Capitalize and add question-mark
        gen_questions = [x.capitalize() + "?" for x in gen_questions]
        questions = [que['question']] + gen_questions
        allowed = [True] + [x["allowed_indices"] for x in cc_item]

        questions_embeddings = model.encode(questions)
        sim_scores = cos_sim([questions_embeddings[0]], questions_embeddings).round(2)[0]
        que_sim_embeds = zip(questions, sim_scores, questions_embeddings, allowed)
        que_sim_embeds = sorted(que_sim_embeds, key=lambda x: x[1], reverse=True)[:num_re]

        _q, _a = [], []
        org_idx = 0
        for idx, item in enumerate(que_sim_embeds):
            q, s, e, a = item
            if s >= sim_thresh:
                qi = deepcopy(que)
                qi["question"] = q
                qi["sim_score"] = s
                qi["allowed"] = a
                if idx != 0:
                    qi["question_id"] = qi["question_id"] * org_factor + org_idx
                    org_idx += 1

                _q.append(qi)
                _a.append(deepcopy(ans))
                question_embeddings_dict[qi["question_id"]] = e

                # import pdb
                # pdb.set_trace()

        # append to main-list
        qlist.append(_q)
        alist.append(_a)

    import pdb
    pdb.set_trace()

    cPickle.dump(qlist, open(cc_qpath, "wb"))
    cPickle.dump(alist, open(cc_apath, "wb"))
    np.save(embeddings_path, question_embeddings_dict)
