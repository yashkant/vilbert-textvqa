from sentence_transformers import SentenceTransformer
import json
import numpy as np
from tqdm import tqdm


def cos_sim(A, B):
    # A and B -> [N x dim]
    Bt = np.transpose(B)
    AB_mul = np.matmul(A, Bt)
    A_mag = np.linalg.norm(A, axis=1, keepdims=True)
    B_mag = np.linalg.norm(B, axis=1, keepdims=True)
    AB_mag = np.matmul(A_mag, np.transpose(B_mag))
    assert AB_mag.shape == AB_mul.shape
    return AB_mul/AB_mag


model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

question_paths = [
    "../../data/re-vqa/data/revqa_train.json",
    "../../data/re-vqa/data/revqa_val.json"
]

question_embeddings = [
    "../../data/re-vqa/data/revqa_train_embeddings.npy",
    "../../data/re-vqa/data/revqa_val_embeddings.npy"
]

question_embeddings_sim = [
    "../../data/re-vqa/data/revqa_train_embeddings_sim.npy",
    "../../data/re-vqa/data/revqa_val_embeddings_sim.npy"
]

batch_size = 160

for question_file, embedding_file, sim_file in zip(question_paths, question_embeddings, question_embeddings_sim):
    q_data = json.load(open(question_file))
    qid_embed_dict = {}
    qid_que_dict = {}
    for _q in (q_data["questions"]):
        qid_que_dict[_q["question_id"]] = _q["question"]

    batches_keys = list(qid_que_dict.keys())

    # all are unique
    assert len(set(batches_keys)) == len(batches_keys)

    batches = np.array_split(batches_keys, int(len(batches_keys)/batch_size))
    embeddings = []
    cos_sim_matrix = np.zeros((len(batches_keys), len(batches_keys)))

    for batch_ids in tqdm(batches):
        _batch = [qid_que_dict[qid] for qid in batch_ids]
        _batch_embeddings = model.encode(_batch)
        embeddings.extend(_batch_embeddings)
        for id, emb in zip(batch_ids, _batch_embeddings):
            qid_embed_dict[id] = emb

    np.save(embedding_file, qid_embed_dict)
    print(f"File Saved: {embedding_file}")

    similarity_matrix = cos_sim(embeddings, embeddings)
    np.save(sim_file, similarity_matrix)
    print(f"File Saved: {sim_file}")





