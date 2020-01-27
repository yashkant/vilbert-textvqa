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


def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--features_dir", default="/srv/share/ykant3/vilbert-mt/features/obj/test/", type=str, help="Path to extracted features file"
    # )
    parser.add_argument(
        "--features_dir", default="/srv/share/ykant3/vilbert-mt/features/obj/train/", type=str, help="Path to extracted features file"
    )
    parser.add_argument(
        "--lmdb_file", default="/srv/share/ykant3/vilbert-mt/features/obj/trainval_obj.lmdb", type=str, help="Path to generated LMDB file"
    )
    # parser.add_argument(
    #     "--imdb_file", default="/nethome/ykant3/pythia/data/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_info_test.npy",
    #     type=str, help="Path to IMDB file"
    # )
    # parser.add_argument(
    #     "--imdb_file", default="/nethome/ykant3/pythia/data/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_train.npy",
    #     type=str, help="Path to IMDB file"
    # )
    # parser.add_argument(
    #     "--imdb_file", default="/nethome/ykant3/pythia/data/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_val.npy",
    #     type=str, help="Path to IMDB file"
    # )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    infiles = glob.glob(os.path.join(args.features_dir, "*"))
    id_list = []
    env = lmdb.open(args.lmdb_file, map_size=MAP_SIZE)
    # imdb_data = np.load(args.imdb_file, allow_pickle=True)
    #
    # imdb_dict = {}
    # for instance in imdb_data[1:]:
    #     imdb_dict[instance["image_id"]] = instance

    with env.begin(write=True) as txn:
        for infile in tqdm.tqdm(infiles):
            reader = np.load(infile, allow_pickle=True)
            item = {}
            item["image_id"] = reader.item().get("image_id")

            # try:
            #     instance = imdb_dict[item["image_id"]]
            # except:
            #     print("Couldn't find", item["image_id"])

            img_id = str(item["image_id"]).encode()
            id_list.append(img_id)

            item["image_h"] = reader.item().get("image_height")
            item["image_w"] = reader.item().get("image_width")
            item["num_boxes"] = reader.item().get("num_boxes")
            item["boxes"] = reader.item().get("bbox")
            item["features"] = reader.item().get("features")

            txn.put(img_id, pickle.dumps(item))
        txn.put("keys".encode(), pickle.dumps(id_list))
