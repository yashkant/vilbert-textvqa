import _pickle as cPickle
from collections import Counter
from tqdm import tqdm

spatial_cache = {
    "test": "../../datasets/textvqa/cache/TextVQA_test_20_vocab_type5k_dynamic_True_spatial.pkl",
    "val": "../../datasets/textvqa/cache/TextVQA_val_20_vocab_type5k_dynamic_True_spatial.pkl",
    "train": "../../datasets/textvqa/cache/TextVQA_train_20_vocab_type5k_dynamic_True_spatial.pkl"
    }

relations_counters = {
    "val": [],
    "test": [],
    "train": []
}

for split in ["val", "test", "train"]:
    entries = cPickle.load(open(spatial_cache[split], "rb"))
    counter = relations_counters[split]

    for entry in tqdm(entries):
        spat_matrix = ((entry["spatial_adj_matrix_shared"]["1"] != 0)*1).sum(axis=-1)
        average_sparsity = spat_matrix.sum()/(len(spat_matrix.nonzero()[0]))
        counter.append(average_sparsity)

    print(f"Average Sparsity for {split}: {sum(counter)/len(counter)}")
