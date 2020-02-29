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

OCR_LMDB_OCRVQA = "/srv/share/ykant3/ocr-vqa/features/lmdbs/ocrvqa_ocr.lmdb"
OCR_FEATURES_OCRVQA = "/srv/share/ykant3/ocr-vqa/features/ocr/"
IMAGES_OCRVQA = "/srv/share/ykant3/ocr-vqa/images/"


# NOTE: Some of the images that are present in the dataset are not annotated and do not exist in provided json file.

if __name__ == "__main__":

    print(f"Reading from: {OCR_FEATURES_OCRVQA}")
    print(f"Storing to: {OCR_LMDB_OCRVQA}")

    feature_files = glob.glob(OCR_FEATURES_OCRVQA + "/**", recursive=True)
    feature_files = [path for path in feature_files if path.endswith(".npy")]

    image_files = glob.glob(IMAGES_OCRVQA + "/**", recursive=True)
    image_files = [path for path in image_files if not os.path.isdir(path)]
    image_id_vs_path_dict = {}

    for image_path in image_files:
        image_id = os.path.split(image_path)[-1].split(".")[0]
        assert image_id not in image_id_vs_path_dict
        image_id_vs_path_dict[image_id] = image_path


    # Image-ids collide for VisualGenomes!
    # Using feature-paths as indices instead!
    id_list = []
    env = lmdb.open(OCR_LMDB_OCRVQA, map_size=MAP_SIZE)
    with env.begin(write=True) as txn:
        for infile in tqdm.tqdm(feature_files):
            try:
                reader = np.load(infile, allow_pickle=True).item()
            except:
                print(f"Couldn't read/find: {infile}")

            image_id = os.path.split(infile)[1].split(".")[0]
            assert image_id in image_id_vs_path_dict
            image_path = image_id_vs_path_dict[image_id]
            image_w, image_h = Image.open(image_path).size
            id = str(infile).encode()
            id_list.append(id)
            item = {}
            item["boxes"] = reader["ocr_boxes"]
            item["features"] = reader["features"]
            item["image_h"] = image_h
            item["image_w"] = image_w
            item["num_boxes"] = len(item["features"])
            txn.put(id, pickle.dumps(item))
        txn.put("keys".encode(), pickle.dumps(id_list))