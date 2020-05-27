# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import json
import os
import pickle

import lmdb
import numpy as np
import tqdm

MAP_SIZE = 1099511627776

IMDB_SCENETEXT_RESPONSE_FIXED_PROCESSED_SPLIT = [
    "/srv/share/ykant3/scene-text/test/imdb/test_task1_response_meta_fixed_processed.npy",
    "/srv/share/ykant3/scene-text/test/imdb/test_task2_response_meta_fixed_processed.npy",
]

# json_paths
JSON_SCENETEXT = [
    "/srv/share/ykant3/scene-text/test/test_task_1.json",
    "/srv/share/ykant3/scene-text/test/test_task_2.json",

]

# image_folders
IMAGES_SCENETEXT = [
    "/srv/share/ykant3/scene-text/test/test_task1/",
    "/srv/share/ykant3/scene-text/test/test_task2/"

]

IMDB_SCENETEXT_PROCESSED = [
    "/srv/share/ykant3/scene-text/test/imdb/test_task1_response_meta_fixed.npy",
    "/srv/share/ykant3/scene-text/test/imdb/test_task2_response_meta_fixed.npy",
]


IMDB_COCOTEXT = "/srv/share/ykant3/coco-text/cocotext_processed.npy"


# Gist: Delete 'response', Add ['answers', 'train_val_ids', 'question', 'question_ids']
# Check what is the difference between 'valid_answers' and 'answers'
# Also fixing the rescaling bullshit from coco-text images!

if __name__ == "__main__":

    import copy
    imdb_coco_text = np.load(IMDB_COCOTEXT, allow_pickle=True)

    # replace ocr-tokens and ocr-info, add new key replace (for ocr and obj extractors)
    add_keys = [
        "google_ocr_tokens",
        "google_ocr_info",
        "google_ocr_tokens_filtered",
        "google_ocr_info_filtered",
        "image_height",
        "image_width"
    ]

    cocotext_train_val_path_instance = {}
    for instance in imdb_coco_text[1:]:
        cocotext_train_val_path_instance[instance["file_name"]] = instance

    for json_path, imdb_response, save_imdb_path, image_fol in zip(JSON_SCENETEXT, IMDB_SCENETEXT_PROCESSED, IMDB_SCENETEXT_RESPONSE_FIXED_PROCESSED_SPLIT, IMAGES_SCENETEXT):

        json_data = json.load(open(json_path, "r"))
        save_imdb = [{
            "json_path": json_path,
            "dataset_name": json_data["dataset_name"],
            "task_name": json_data["task_name"]
        }]
        json_data = json_data["data"]

        response_data = np.load(imdb_response, allow_pickle=True)
        response_path_instance = {}
        for instance in response_data[1:]:
            response_path_instance[instance["image_path"]] = instance

        for instance in tqdm.tqdm(json_data):
            # check where to add keys from
            if "COCO" in instance["file_path"]:
                # take from cocotext-imdb
                coco_key = os.path.split(instance["file_path"])[-1]
                coco_instance = cocotext_train_val_path_instance[coco_key]
                for key in add_keys:
                    instance[key] = copy.deepcopy(coco_instance[key])
            else:
                response_key = f"{image_fol}{instance['file_path']}"
                response_instance = response_path_instance[response_key]
                for key in add_keys:
                    instance[key] = copy.deepcopy(response_instance[key])


            save_imdb.append(instance)

        print("Lens: ", len(save_imdb))
        print(f"Save path: {save_imdb_path}")
        np.save(save_imdb_path, save_imdb)

