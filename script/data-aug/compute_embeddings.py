from sentence_transformers import SentenceTransformer
import json
import numpy as np
from tqdm import tqdm
import _pickle as cPickle


# decontract, depunctuate
filter_type = "dcp"

# sampling: ["top", "random"]
# sampling_type = "random"
sampling_type = "top"

# filter_thresh
sim_thresh = 0.8


que_split_path_dict = {
    "train": "../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_train2014_questions.pkl"
        .format(filter_type, sampling_type),
    "val": "../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_val2014_questions.pkl"
        .format(filter_type, sampling_type),
    "test": "../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_test2015_questions.pkl"
        .format(filter_type, sampling_type),
}

embed_path_dict = {
    "train": "../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_train2014_questions_emb.pkl"
        .format(filter_type, sampling_type),
    "val": "../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_val2014_questions_emb.pkl"
        .format(filter_type, sampling_type),
    "test": "../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_test2015_questions_emb.pkl"
        .format(filter_type, sampling_type),
}


# que_split_path_dict = {
#     "train": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_train2014_questions.pkl",
#     "val":  "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
#     "test": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_test2015_questions.pkl",
#     # "trainval": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_trainval2014_questions.pkl",
#     # "minval": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_minval2014_questions.pkl",
# }

# embed_path_dict = {
#     "train": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_train2014_questions_emb.npy",
#     "val":  "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_val2014_questions_emb.npy",
#     "test": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_test2015_questions_emb.npy",
#     # "trainval": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_trainval2014_questions_emb.npy",
#     # "minval": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_minval2014_questions_emb.npy",
# }

model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

batch_size = 160

for split, path in que_split_path_dict.items():
    q_data = cPickle.load(open(path, "rb"))["questions"]

    qid_embed_dict = {}
    qid_que_dict = {}

    for _q in tqdm(q_data, desc="Filtering"):
        # assert "en" in _q[0]["languages"]
        assert _q[0]["sim_score"] == 1
        assert len(_q) < 5

        for que in _q:
            if que["sim_score"] >= sim_thresh:
                qid_que_dict[que["question_id"]] = que["question"]

    batches_keys = list(qid_que_dict.keys())

    # all are unique
    assert len(set(batches_keys)) == len(batches_keys)

    batches = np.array_split(batches_keys, int(len(batches_keys)/batch_size))

    for batch_ids in tqdm(batches, desc=f"Processing {split}"):
        _batch = [qid_que_dict[qid] for qid in batch_ids]
        _batch_embeddings = model.encode(_batch)
        for id, emb in zip(batch_ids, _batch_embeddings):
            qid_embed_dict[id] = emb

    embedding_file = embed_path_dict[split]
    np.save(embedding_file, qid_embed_dict)
    print(f"File Saved: {embedding_file}")



