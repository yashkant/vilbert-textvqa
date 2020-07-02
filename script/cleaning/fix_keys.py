import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
import tqdm

MAP_SIZE = 1099511627776

def read_lmdb(path):
    env = lmdb.open(
        path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    with env.begin(write=False) as txn:
        image_ids = pickle.loads(txn.get("keys".encode()))
        item = pickle.loads(txn.get(image_ids[0]))
        import pdb
        pdb.set_trace()

    import pdb
    pdb.set_trace()


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


# TextVQA
features_h5path1 = "/srv/share/ykant3/vilbert-mt/features/obj/trainval_obj.lmdb"  # obj-features (use holder to different splits)
features_h5path2 = "/srv/share/ykant3/vilbert-mt/features/ocr/trainval_ocr.lmdb"  # ocr-features

# STVQA
stvqa_features_h5path1 = "/srv/share/ykant3/scene-text/features/obj/lmdbs/train_task_fixed.lmdb"  # object features
stvqa_features_h5path2 = "/srv/share/ykant3/scene-text/features/ocr/lmdbs/train_task_fixed.lmdb"  # ocr features


paths = {
    "stvqa_obj_trainval": "/srv/share/ykant3/scene-text/features/obj/lmdbs/train_task_fixed.lmdb",
    "stvqa_ocr_trainval": "/srv/share/ykant3/scene-text/features/ocr/lmdbs/train_task_fixed.lmdb",

    "stvqa_obj_test": "/srv/share/ykant3/scene-text/features/obj/lmdbs/test_task3_fixed.lmdb",
    "stvqa_ocr_test": "/srv/share/ykant3/scene-text/features/ocr/lmdbs/test_task3_fixed.lmdb",
}

fixed_paths = {
    "stvqa_obj_trainval": "/srv/share/ykant3/scene-text/features/obj/lmdbs/stvqa_train_task.lmdb",
    "stvqa_ocr_trainval": "/srv/share/ykant3/scene-text/features/ocr/lmdbs/stvqa_train_task.lmdb",

    "stvqa_obj_test": "/srv/share/ykant3/scene-text/features/obj/lmdbs/stvqa_test_task3.lmdb",
    "stvqa_ocr_test": "/srv/share/ykant3/scene-text/features/ocr/lmdbs/stvqa_test_task3.lmdb",
}


for tag, path in paths.items():

    env = lmdb.open(
        path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    with env.begin(write=False) as txn:
        image_ids = pickle.loads(txn.get("keys".encode()))

    save_path = fixed_paths[tag]
    new_env = lmdb.open(save_path, map_size=MAP_SIZE)

    new_id_list = []
    with new_env.begin(write=True) as new_txn:
        for image_id in tqdm.tqdm(image_ids):

            with env.begin(write=False) as txn:
                item = pickle.loads(txn.get(image_id))

            image_id_new = splitall(image_id.decode())
            image_id_new = f"{image_id_new[-2]}/{image_id_new[-1].split('.')[0]}"
            new_id_list.append(image_id_new)
            new_txn.put(image_id_new.encode(), pickle.dumps(item))
        new_txn.put("keys".encode(), pickle.dumps(new_id_list))






































