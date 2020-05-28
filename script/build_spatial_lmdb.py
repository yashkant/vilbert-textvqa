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
    # "../datasets/VQA/cache/VQA_trainval_23_heads_new.pkl",
    # "../datasets/VQA/cache/VQA_minval_23_heads_new.pkl",
    # "../datasets/VQA/cache/VQA_val_23_heads_new.pkl",
    "../datasets/VQA/cache/VQA_test_23_heads_new.pkl",
]

CACHE_LMDBS = [
    # "../datasets/VQA/cache/VQA_trainval_23_heads_new.lmdb",
    # "../datasets/VQA/cache/VQA_minval_23_heads_new.lmdb",
    # "../datasets/VQA/cache/VQA_mal_23_heads_new.lmdb",
    "../datasets/VQA/cache/VQA_test_23_heads_new.lmdb",
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

                    if id in set(id_list):
                        continue

                    id_list.append(id)
                    item["spatial_adj_matrix"] = entry["spatial_adj_matrix"]
                    item["spatial_adj_matrix_share3_1"] = entry["spatial_adj_matrix_share3_1"]
                    item["spatial_adj_matrix_share3_2"] = entry["spatial_adj_matrix_share3_2"]
                    item["spatial_adj_matrix_random1"] = entry["spatial_adj_matrix_random1"]
                    item["spatial_adj_matrix_random3"] = entry["spatial_adj_matrix_random3"]

                    txn.put(id, pickle.dumps(item))
                except:
                    print(f"Error Occurred with: {id}")
            txn.put("keys".encode(), pickle.dumps(id_list))