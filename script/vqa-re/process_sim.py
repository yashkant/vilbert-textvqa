import numpy as np
from tqdm import tqdm
import json

blocksize = 1024  # tune this for performance/granularity


question_paths = [
    "../../data/re-vqa/data/revqa_train.json",
    # "../../data/re-vqa/data/revqa_val.json"
]


question_embeddings_sim = [
    "../../data/re-vqa/data/revqa_train_embeddings_sim.npy",
    # "../../data/re-vqa/data/revqa_val_embeddings_sim.npy"
]

process_question_paths = [
    "../../data/re-vqa/data/revqa_train_proc.json",
    # "../../data/re-vqa/data/revqa_val_proc.json"
]


top_k = 300
neg_value = -10

for sim_file, que_file, proc_que_file in zip(question_embeddings_sim, question_paths, process_question_paths):
    try:
        mmap = np.load(sim_file, mmap_mode='r')
        data = np.empty_like(mmap)
        data_top_k = np.zeros((data.shape[0], top_k), dtype=np.int32) - 1
        n_blocks = int(np.ceil(mmap.shape[0] / blocksize))
        for b in tqdm(range(n_blocks), total=n_blocks, desc="Reading"):
            data[b*blocksize : (b+1) * blocksize] = mmap[b*blocksize : (b+1) * blocksize]
    finally:
        del mmap  # make sure file is closed again

    col_indices = range(data.shape[0])
    for idx in tqdm(range(top_k), total=top_k, desc="Calculating Top-k"):
        argmax_inds = np.argmax(data, axis=1)
        data_top_k[col_indices, idx] = argmax_inds
        data[col_indices, argmax_inds] = neg_value

    json_data = json.load(open(que_file))
    questions = json_data["questions"]
    question_ids = np.array([x["question_id"] for x in questions])

    assert len(question_ids) == (data.shape[0])

    for idx, _q in enumerate(questions):
        _q["top_k_questions"] = question_ids[data_top_k[idx]].tolist()

    json.dump(json_data, open(proc_que_file, "w"))
    print(f"Dumping File: {proc_que_file}")

