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

imdb_paths = {
    "train": "/nethome/ykant3/pythia/data/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_train.npy",
    "val": "/nethome/ykant3/pythia/data/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_val.npy",
    "test": "/nethome/ykant3/pythia/data/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_info_test.npy",
}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_dir", default="/srv/share/ykant3/vilbert-mt/features/ocr/train/", type=str, help="Path to extracted features file"
    )
    parser.add_argument(
        "--lmdb_file", default="/srv/share/ykant3/vilbert-mt/features/ocr/trainval_ocr.lmdb", type=str, help="Path to generated LMDB file"
    )
    return parser


if __name__ == "__main__":

    imdb_datum = [np.load(imdb_paths[split], allow_pickle=True) for split in ["train", "val"]]
    imdb_dict = {}

    for imdb_data in imdb_datum:
        for instance in imdb_data[1:]:
            imdb_dict[instance["image_id"]] = instance

    print("IMDB Dict len: ", len(imdb_dict))

    import pdb
    pdb.set_trace()

    args = get_parser().parse_args()
    infiles = glob.glob(os.path.join(args.features_dir, "*"))
    id_list = []
    env = lmdb.open(args.lmdb_file, map_size=MAP_SIZE)

    with env.begin(write=True) as txn:
        for infile in tqdm.tqdm(infiles):
            reader = np.load(infile, allow_pickle=True)
            item = {}
            item["image_id"] = reader.item().get("image_id")

            try:
                instance = imdb_dict[item["image_id"]]
            except:
                import pdb
                pdb.set_trace()

            img_id = str(item["image_id"]).encode()
            id_list.append(img_id)

            item["boxes"] = reader.item().get("ocr_boxes")
            item["features"] = reader.item().get("features")

            # assert that ocr-boxes are consistent
            assert len(instance['google_ocr_info_filtered']) == len(item["features"])

            item["image_h"] = instance.get("image_height")
            item["image_w"] = instance.get("image_width")
            item["num_boxes"] = len(item["features"])

            txn.put(img_id, pickle.dumps(item))
        txn.put("keys".encode(), pickle.dumps(id_list))
