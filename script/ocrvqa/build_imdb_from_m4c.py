# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
from tqdm import tqdm
from PIL import Image

MAP_SIZE = 1099511627776

IMDB_OCRVQA_RAW = "/srv/share/ykant3/ocr-vqa/intermediate/ocrvqa.npy"

IMDB_M4C = [
    "/nethome/ykant3/m4c-release/data/imdb/m4c_ocrvqa/imdb_train.npy",
    "/nethome/ykant3/m4c-release/data/imdb/m4c_ocrvqa/imdb_val.npy",
    "/nethome/ykant3/m4c-release/data/imdb/m4c_ocrvqa/imdb_test.npy",
]

IMDB_OCRVQA_SPLITS = [
    "/srv/share/ykant3/ocr-vqa/intermediate/ocrvqa_train.npy",
    "/srv/share/ykant3/ocr-vqa/intermediate/ocrvqa_val.npy",
    "/srv/share/ykant3/ocr-vqa/intermediate/ocrvqa_test.npy",
]

IMAGES_OCRVQA = "/srv/share/ykant3/ocr-vqa/images/"

# image_folders

if __name__ == "__main__":
    raw_imdb_data = np.load(IMDB_OCRVQA_RAW, allow_pickle=True)
    image_path_vs_instance = {}

    for instance in raw_imdb_data:
        if "image_path" in instance:
            image_path_vs_instance[instance["image_path"]] = instance

    for m4c_imdb_path, save_imdb_path in zip(IMDB_M4C, IMDB_OCRVQA_SPLITS):
        m4c_imdb_data = np.load(m4c_imdb_path, allow_pickle=True)
        save_imdb_data = [
            m4c_imdb_data[0]
        ]

        print(f"Saving to: {save_imdb_path}")
        for instance in tqdm(m4c_imdb_data[1:]):
            abs_image_path = os.path.join(IMAGES_OCRVQA, instance["image_path"])
            image_w, image_h = Image.open(abs_image_path).size
            assert os.path.exists(abs_image_path)
            raw_instance = image_path_vs_instance[abs_image_path]
            save_instance = {"image_height": image_h, "image_width": image_w}

            put_keys_raw = [
                "google_ocr_tokens_filtered",
                "google_ocr_info_filtered",
            ]
            for key in put_keys_raw:
                save_instance[key] = raw_instance[key]

            put_keys_m4c = [
                "image_path",
                "answers",
                "question",
                "question_id"
            ]
            for key in put_keys_m4c:
                save_instance[key] = instance[key]

            save_imdb_data.append(save_instance)

        np.save(save_imdb_path, save_imdb_data)
