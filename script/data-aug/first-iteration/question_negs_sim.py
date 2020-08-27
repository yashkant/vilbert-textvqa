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


que_split_path_dict = {
    "train": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_train2014_questions.pkl",
    "val":  "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
}

embed_path_dict = {
    "train": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_train2014_questions_emb.npy",
    "val":  "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_val2014_questions_emb.npy",
}

save_path = "../../datasets/VQA/back-translate/train_val_question_negs.npy"

sim_thresh = 0.9
max_samples = 5
num_samples = 300
iter_limit = 1000

train_data = np.load(embed_path_dict["train"], allow_pickle=True).item()
val_data = np.load(embed_path_dict["val"], allow_pickle=True).item()
# dict of qid -> embedding vector
data = {**train_data, **val_data}

question_ids = np.array(list(data.keys()))
questions_embed = np.stack([data[key] for key in question_ids])
hard_negatives = {}
num_iters = int(np.ceil(len(question_ids)/iter_limit))

new_qids = []
new_sim_qids = []
new_sim_scores = []



def process (idx):
    que_vector = questions_embed[idx * iter_limit: (idx+1) * iter_limit]
    que_scores = cos_sim(que_vector, questions_embed)
    # sort last-300 scores
    sorted_inds = np.argpartition(-1 * que_scores, kth=num_samples, axis=-1)[:, :num_samples]

    qids = question_ids[idx * iter_limit: (idx+1) * iter_limit]
    sim_qids = question_ids[sorted_inds]
    sim_scores = np.take_along_axis(que_scores, sorted_inds, -1)
    return qids, sim_qids, sim_scores

import multiprocessing as mp

sp_pool = mp.Pool(8)
args_list = []
for idx in (range(num_iters)):
    _args = idx
    args_list.append(_args)

import pdb
pdb.set_trace()

batches_list = list(tqdm(sp_pool.imap(process, args_list), total=len(args_list)))
sp_pool.close()
sp_pool.join()

new_qids, new_sim_qids, new_sim_scores = list(zip(*batches_list))
new_qids = np.concatenate(new_qids)
new_sim_qids = np.concatenate(new_sim_qids)
new_sim_scores = np.concatenate(new_sim_scores)
print(f"Len: {len(new_qids)}")

import pdb
pdb.set_trace()

# np.save(save_path, {"qids": new_qids, "sim_scores": new_sim_qids, "sim_qids": new_sim_qids})
