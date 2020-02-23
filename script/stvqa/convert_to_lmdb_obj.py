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


OBJ_FEATURES_SCENETEXT = [
    "/srv/share/ykant3/scene-text/features/obj/train/train_task/",
    "/srv/share/ykant3/scene-text/features/obj/test/test_task3/"
]

OBJ_LMDB_SCENETEXT = [
    "/srv/share/ykant3/scene-text/features/obj/lmdbs/train_task_fixed2.lmdb",
    "/srv/share/ykant3/scene-text/features/obj/lmdbs/test_task3_fixed2.lmdb"
]

IMAGES_SCENETEXT = [
    "/srv/share/ykant3/scene-text/train/train_task/",
    "/srv/share/ykant3/scene-text/test/test_task3/"
]



if __name__ == "__main__":

    for feature_dir, lmdb_file, images_dir in zip(OBJ_FEATURES_SCENETEXT, OBJ_LMDB_SCENETEXT, IMAGES_SCENETEXT):
        print(f"Reading from: {feature_dir}")
        print(f"Storing to: {lmdb_file}")
        feature_files = glob.glob(feature_dir + "/**", recursive=True)
        feature_files = [path for path in feature_files if path.endswith(".npy")]
        id_list = []
        env = lmdb.open(lmdb_file, map_size=MAP_SIZE)
        with env.begin(write=True) as txn:
            for infile in tqdm.tqdm(feature_files):
                try:
                    reader = np.load(infile, allow_pickle=True).item()
                    item = {}
                    id = str(infile).encode()
                    id_list.append(id)

                    if os.path.split(infile)[-1] == "COCO_train2014_000000084731.npy":
                        import pdb
                        pdb.set_trace()

                    # Image-ids collide for VisualGenomes!
                    # Using feature-paths as indices instead!
                    item["image_id"] = reader.get("image_id")
                    item["image_h"] = reader.get("image_height")
                    item["image_w"] = reader.get("image_width")
                    item["num_boxes"] = reader.get("num_boxes")
                    item["boxes"] = reader.get("bbox")
                    item["features"] = reader.get("features")
                    item["cls_prob"] = reader.get("cls_prob")
                    item["objects"] = reader.get("objects")
                    txn.put(id, pickle.dumps(item))
                except:
                    print(f"Error Occurred with: {infile}")
            txn.put("keys".encode(), pickle.dumps(id_list))
