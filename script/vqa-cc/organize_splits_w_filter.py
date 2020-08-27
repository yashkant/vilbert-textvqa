from copy import deepcopy

from sentence_transformers import SentenceTransformer
import json
import numpy as np
from tqdm import tqdm
import _pickle as cPickle

def cos_sim(A, B):
    # A and B -> [N x dim]
    Bt = np.transpose(B)
    AB_mul = np.matmul(A, Bt)
    A_mag = np.linalg.norm(A, axis=1, keepdims=True)
    B_mag = np.linalg.norm(B, axis=1, keepdims=True)
    AB_mag = np.matmul(A_mag, np.transpose(B_mag))
    assert AB_mag.shape == AB_mul.shape
    return AB_mul/AB_mag

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

cc_rephrasings_path = "/nethome/ykant3/pythia/results/noise/results/pythia_cycle_consistent_with_failure_prediction/58383/gen_questions.npy"
cc_data = np.load(cc_rephrasings_path, allow_pickle=True).item()
reference_file = "../../datasets/VQA/back-translate/bt_fil_dcp_sampling_top_v2_OpenEnded_mscoco_train2014_questions.pkl"
ref_data = cPickle.load(open(reference_file, "rb"))
#[ [sim_score, question_id, image_id, question] ]

# Dumping Paths
que_split_path_dict = {
    "train": ("../../datasets/VQA/cc-re/cc_v2_OpenEnded_mscoco_train2014_questions_88.pkl",
              "../../datasets/VQA/v2_OpenEnded_mscoco_train2014_questions.json"),
}


ans_split_path_dict = {
    "train": ("../../datasets/VQA/cc-re/cc_train_target_88.pkl",
              "../../datasets/VQA/cache/train_target.pkl"),
}

embeddings_path = "../../datasets/VQA/cc-re/train_cc_embeddings_88.npy"

for split in ["train"]:
    dump_qdict = {}
    question_embeddings_dict = {}

    qlist, alist = [], []
    cc_qpath, org_qpath = que_split_path_dict[split][0], que_split_path_dict[split][1]

    org_ans = cPickle.load(open(ans_split_path_dict[split][1], "rb"))
    cc_qdata = np.load(cc_rephrasings_path, allow_pickle=True).item()
    org_qdata = json.load(open(org_qpath))["questions"]

    for que, ans in tqdm(zip(org_qdata, org_ans), total=len(org_qdata)):
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

        # append to main-list
        qlist.append(_q)
        alist.append(_a)

    import pdb
    pdb.set_trace()

    cPickle.dump(qlist, open(cc_qpath, "wb"))
    cPickle.dump(alist, open(ans_split_path_dict[split][0], "wb"))
    np.save(embeddings_path, question_embeddings_dict)
