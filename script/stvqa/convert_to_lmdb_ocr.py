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

MAP_SIZE = 1099511627776

OCR_LMDB_SCENETEXT = [
    "/srv/share/ykant3/scene-text/features/ocr/lmdbs/train_task_fixed2.lmdb",
    "/srv/share/ykant3/scene-text/features/ocr/lmdbs/test_task3_fixed2.lmdb"
]

OCR_FEATURES_SCENETEXT = [
    "/srv/share/ykant3/scene-text/features/ocr/train/train_task/",
    "/srv/share/ykant3/scene-text/features/ocr/test/test_task3/"
]

IMDB_SCENETEXT_RESPONSE_FIXED_PROCESSED_SPLIT = [
    ["/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_train.npy",
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_val.npy"],
    ["/srv/share/ykant3/scene-text/test/imdb/test_task3_response_meta_fixed_processed.npy"],
]

IMDB_M4C = [
    ["/srv/share/ykant3/m4c-release/data/imdb/m4c_stvqa/imdb_subtrain.npy",
    "/srv/share/ykant3/m4c-release/data/imdb/m4c_stvqa/imdb_subval.npy"],
    "/srv/share/ykant3/m4c-release/data/imdb/m4c_stvqa/imdb_test_task3.npy",
]


# NOTE: Some of the images that are present in the dataset are not annotated and do not exist in provided json file.

if __name__ == "__main__":
    for feature_dir, lmdb_file, imdb_files in zip(OCR_FEATURES_SCENETEXT, OCR_LMDB_SCENETEXT, IMDB_SCENETEXT_RESPONSE_FIXED_PROCESSED_SPLIT):
        print(f"Reading from: {feature_dir} and {imdb_files}")
        print(f"Storing to: {lmdb_file}")
        imdb_dict = {}
        imdb_data = []

        for imdb_file in imdb_files:
            _data = np.load(imdb_file, allow_pickle=True)
            imdb_data.extend(_data[1:])

        for instance in imdb_data:
            imdb_dict[instance["image_path"].split(".")[0]] = instance

        feature_files = glob.glob(feature_dir + "/**", recursive=True)
        feature_files = [path for path in feature_files if path.endswith(".npy")]

        # Image-ids collide for VisualGenomes!
        # Using feature-paths as indices instead!
        id_list = []
        env = lmdb.open(lmdb_file, map_size=MAP_SIZE)
        with env.begin(write=True) as txn:
            for infile in tqdm.tqdm(feature_files):
                reader = np.load(infile, allow_pickle=True).item()
                try:
                    image_path_key = infile.replace("features/ocr/", "").split(".")[0]
                    instance = imdb_dict[image_path_key]
                except:
                    print(f"Image path not found in imdb: {image_path_key}")
                    continue
                try:
                    # assert that ocr-boxes are consistent
                    assert len(instance['google_ocr_info_filtered']) == len(reader["features"])
                except:
                    import pdb
                    pdb.set_trace()
                    continue

                id = str(infile).encode()
                id_list.append(id)
                item = {}
                item["boxes"] = reader.get("ocr_boxes")
                item["features"] = reader.get("features")
                item["image_h"] = instance.get("image_height")
                item["image_w"] = instance.get("image_width")
                item["num_boxes"] = len(item["features"])
                txn.put(id, pickle.dumps(item))
            txn.put("keys".encode(), pickle.dumps(id_list))
