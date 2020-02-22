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

IMDB_SCENETEXT_RESPONSE_FIXED = [
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed.npy",
    "/srv/share/ykant3/scene-text/test/imdb/test_task3_response_meta_fixed.npy",
]

IMDB_SCENETEXT_RESPONSE_FIXED_PROCESSED_SPLIT = [
    ["/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_train.npy",
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_val.npy"],
    "/srv/share/ykant3/scene-text/test/imdb/test_task3_response_meta_fixed_processed.npy",
]

IMDB_M4C = [
    ["/srv/share/ykant3/m4c-release/data/imdb/m4c_stvqa/imdb_subtrain.npy",
    "/srv/share/ykant3/m4c-release/data/imdb/m4c_stvqa/imdb_subval.npy"],
    "/srv/share/ykant3/m4c-release/data/imdb/m4c_stvqa/imdb_test_task3.npy",
]

# image_folders
IMAGES_SCENETEXT = [
    "/srv/share/ykant3/scene-text/train/train_task/",
    "/srv/share/ykant3/scene-text/test/test_task3/"

]

IMDB_COCOTEXT = "/srv/share/ykant3/coco-text/cocotext_processed.npy"





# Gist: Delete 'response', Add ['answers', 'train_val_ids', 'question', 'question_ids']
# Check what is the difference between 'valid_answers' and 'answers'
# Todo: Rescaling bullshit from coco-text images!

if __name__ == "__main__":

    import copy
    imdb_train_val = np.load(IMDB_SCENETEXT_RESPONSE_FIXED[0], allow_pickle=True)
    imdb_coco_text = np.load(IMDB_COCOTEXT, allow_pickle=True)

    train_val_path_instance = {}
    for instance in imdb_train_val[1:]:
        del instance["response"]
        train_val_path_instance[instance["image_path"]] = instance

    cocotext_train_val_path_instance = {}
    for instance in imdb_coco_text[1:]:
        cocotext_train_val_path_instance[instance["file_name"]] = instance

    for m4c_imdb_path, save_imdb_path in zip(IMDB_M4C[0], IMDB_SCENETEXT_RESPONSE_FIXED_PROCESSED_SPLIT[0]):

        m4c_imdb = np.load(m4c_imdb_path, allow_pickle=True)
        save_imdb = [m4c_imdb[0]]

        count_replace = []

        question_ids = []
        for instance in tqdm.tqdm(m4c_imdb[1:]):
            assert instance['answers'] == instance['valid_answers']

            train_val_key = "/srv/share/ykant3/scene-text/train/train_task/" + instance["image_path"]
            assert train_val_key in train_val_path_instance
            train_val_instance = copy.deepcopy(train_val_path_instance[train_val_key])

            train_val_instance.update({
                "answers": instance["answers"],
                "question": instance["question"],
                "question_id": instance["question_id"],
            })

            # check for image-resizing in coco-text!
            if "coco" in train_val_key and \
                    (instance["image_height"], instance["image_width"]) != (train_val_instance["image_height"], train_val_instance["image_width"]):
                count_replace.append(train_val_key)
                coco_key = os.path.split(instance["image_path"])[-1]
                coco_instance = cocotext_train_val_path_instance[coco_key]
                assert (instance["image_height"], instance["image_width"]) == \
                       (coco_instance["image_height"], coco_instance["image_width"])

                # replace ocr-tokens and ocr-info, add new key replace (for ocr and obj extractors)
                replace_keys = [
                    "google_ocr_tokens",
                    "google_ocr_info",
                    "google_ocr_tokens_filtered",
                    "google_ocr_info_filtered",
                    "image_height",
                    "image_width"
                ]

                for key in replace_keys:
                    train_val_instance[key] = coco_instance[key]
                train_val_instance["replaced"] = True
            # append to imdb

            question_ids.append(train_val_instance["question_id"])
            save_imdb.append(train_val_instance)

        import pdb
        pdb.set_trace()
        print("Lens: ", len(set(question_ids)), len(question_ids))
        save_imdb[0]["replaced_paths"] = list(set(count_replace))
        print(f"Save path: {save_imdb_path}")
        np.save(save_imdb_path, save_imdb)

