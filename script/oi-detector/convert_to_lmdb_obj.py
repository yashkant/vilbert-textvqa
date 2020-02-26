# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
import tqdm
from PIL import Image

MAP_SIZE = 1099511627776


OI_OBJ_FEATURES = [
    "/srv/share/ykant3/vilbert-mt/features/oi-features/obj/train/",
    "/srv/share/ykant3/vilbert-mt/features/oi-features/obj/test/"
]

OI_OBJ_LMDBS = [
    "/srv/share/ykant3/vilbert-mt/features/oi-features/lmdb/oi_train.lmdb",
    "/srv/share/ykant3/vilbert-mt/features/oi-features/lmdb/oi_test.lmdb"
]

IMAGES_TEXTVQA = [
    "/srv/share/ykant3/pythia/dataset_images/train_images/",
    "/srv/share/ykant3/pythia/dataset_images/test_images/",
]

if __name__ == "__main__":
    for feature_dir, lmdb_file, images_dir in zip(OI_OBJ_FEATURES, OI_OBJ_LMDBS, IMAGES_TEXTVQA):
        infiles = glob.glob(os.path.join(feature_dir, "*"))
        id_list = []
        env = lmdb.open(lmdb_file, map_size=MAP_SIZE)
        with env.begin(write=True) as txn:
            for infile in tqdm.tqdm(infiles):
                reader = np.load(infile, allow_pickle=True).item()
                item = {}
                image_id = reader.get("image_url").split(".")[0]
                image_path = os.path.join(images_dir, image_id + ".jpg")
                assert os.path.exists(image_path)
                image_w, image_h = Image.open(image_path).size
                item["image_id"] = image_id
                img_id = str(item["image_id"]).encode()
                id_list.append(img_id)
                item["image_h"] = image_h
                item["image_w"] = image_w
                item["boxes"] = reader.get("bounding_boxes")
                item["num_boxes"] = len(item["boxes"])
                item["features"] = reader.get("features")
                txn.put(img_id, pickle.dumps(item))
            txn.put("keys".encode(), pickle.dumps(id_list))
