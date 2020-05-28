# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import _pickle as cPickle
import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
import tqdm

MAP_SIZE = 1099511627776


CACHES = [
    "../../datasets/stvqa/cache/STVQA_train_20_vocab_type5k_stvqa_dynamic_True_spatial.pkl",
    "../../datasets/stvqa/cache/STVQA_val_20_vocab_type5k_stvqa_dynamic_True_spatial.pkl",
    "../../datasets/stvqa/cache/STVQA_test_20_vocab_type5k_stvqa_dynamic_True_spatial.pkl",
]

CACHE_LMDBS = [
    "../../datasets/stvqa/cache/STVQA_train_20_vocab_type5k_stvqa_dynamic_True_spatial.lmdb",
    "../../datasets/stvqa/cache/STVQA_val_20_vocab_type5k_stvqa_dynamic_True_spatial.lmdb",
    "../../datasets/stvqa/cache/STVQA_test_20_vocab_type5k_stvqa_dynamic_True_spatial.lmdb",
]

if __name__ == "__main__":
    for cache_pkl, cache_lmdb in zip(CACHES, CACHE_LMDBS):
        print(f"Reading from: {cache_pkl}")
        print(f"Storing to: {cache_lmdb}")

        cache_entries = cPickle.load(open(cache_pkl, "rb"))
        id_list = []
        env = lmdb.open(cache_lmdb, map_size=MAP_SIZE)
        with env.begin(write=True) as txn:
            for entry in tqdm.tqdm(cache_entries):
                try:
                    item = {}
                    id = str(entry["image_id"]).encode()
                    id_list.append(id)
                    item["spatial_adj_matrix_shared"] = entry["spatial_adj_matrix_shared"]
                    txn.put(id, pickle.dumps(item))
                except:
                    print(f"Error Occurred with: {id}")
            txn.put("keys".encode(), pickle.dumps(id_list))