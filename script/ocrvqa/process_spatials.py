# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import pickle
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import numpy as np
import tqdm
from PIL import Image
import multiprocessing as mp
import logging
from vilbert.spatial_utils import build_graph_using_normalized_boxes_new

MAP_SIZE = 1099511627776

# Read from spatials
# OCR_SPATIAL_LMDB_OCRVQA = "/srv/share/ykant3/ocr-vqa/features/lmdbs/ocrvqa_ocr_spatial.lmdb"
OCR_LMDB_OCRVQA = "/srv/share/ykant3/ocr-vqa/features/lmdbs/ocrvqa_ocr.lmdb"
OBJ_LMDB_OCRVQA = "/srv/share/ykant3/ocr-vqa/features/lmdbs/ocrvqa.lmdb"

OCR_FEATURES_OCRVQA = "/srv/share/ykant3/ocr-vqa/features/ocr/"
OCR_SPATIAL_FEATURES_OCRVQA = "/srv/share/ykant3/ocr-vqa/features/ocr-spatial/"
OBJ_FEATURES_OCRVQA = "/srv/share/ykant3/ocr-vqa/features/obj/"
IMAGES_OCRVQA = "/srv/share/ykant3/ocr-vqa/images/"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
ocr_feature_reader = ImageFeaturesH5Reader(OCR_LMDB_OCRVQA)
obj_feature_reader = ImageFeaturesH5Reader(OBJ_LMDB_OCRVQA)


def pad_features(features, bboxes, num_boxes, max_feat_num, tensorize=False):
    mix_num_boxes = min(int(num_boxes), max_feat_num)
    mask = [1] * (int(mix_num_boxes))
    while len(mask) < max_feat_num:
        mask.append(0)

    mix_boxes_pad = np.zeros((max_feat_num, 5))
    mix_boxes_pad[:mix_num_boxes] = bboxes[:mix_num_boxes]

    mix_features_pad = np.zeros((max_feat_num, 2048))
    mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

    return mix_features_pad, mask, mix_boxes_pad


def process_spatial_ocr(item):
    if not os.path.exists(OCR_SPATIAL_FEATURES_OCRVQA):
        os.makedirs(OCR_SPATIAL_FEATURES_OCRVQA)
    image_id = item["image_id"]
    save_path = os.path.join(OCR_SPATIAL_FEATURES_OCRVQA, image_id + ".npy")
    if os.path.exists(save_path):
        return

    ocr_feat_path = item["ocr_feature_path"]
    obj_key = os.path.join(OBJ_FEATURES_OCRVQA, image_id + ".npy")
    ocr_key = os.path.join(OCR_FEATURES_OCRVQA, image_id + ".npy")

    obj_features, obj_num_boxes, obj_bboxes, _ = obj_feature_reader[obj_key]
    obj_features, obj_num_boxes, obj_bboxes = obj_features[1:], obj_num_boxes - 1, obj_bboxes[1:]
    _, _, pad_obj_bboxes = pad_features(
        obj_features, obj_bboxes, obj_num_boxes, 100, tensorize=False
    )
    ocr_features, ocr_num_boxes, ocr_bboxes, _ = ocr_feature_reader[ocr_key]
    ocr_features, ocr_num_boxes, ocr_bboxes = ocr_features[1:], ocr_num_boxes - 1, ocr_bboxes[1:]
    _, _, pad_ocr_bboxes = pad_features(
        ocr_features, ocr_bboxes, ocr_num_boxes, 50, tensorize=False
    )

    # Append bboxes to the list
    pad_obj_ocr_bboxes = np.concatenate([pad_obj_bboxes[:, :-1], pad_ocr_bboxes[:, :-1]], axis=0)
    adj_matrix, adj_matrix_share3_1, adj_matrix_share3_2 \
        = build_graph_using_normalized_boxes_new(pad_obj_ocr_bboxes, distance_threshold=0.5)

    try:
        data = np.load(ocr_feat_path, allow_pickle=True).item()
    except:
        print(f"Error with: {ocr_feat_path}")
        return

    data["adj_matrix"] = adj_matrix
    data["adj_matrix_share3_1"] = adj_matrix_share3_1
    data["adj_matrix_share3_2"] = adj_matrix_share3_2
    np.save(save_path, data)


# NOTE: Some of the images that are present in the dataset are not annotated and do not exist in provided json file.

if __name__ == "__main__":

    print(f"Reading from: {OCR_FEATURES_OCRVQA}")
    print(f"Storing to: {OCR_SPATIAL_FEATURES_OCRVQA}")

    # ocr_feature_files = glob.glob(OCR_FEATURES_OCRVQA + "/**", recursive=True)
    # ocr_feature_files = [path for path in ocr_feature_files if path.endswith(".npy")]
    image_files = glob.glob(IMAGES_OCRVQA + "/**", recursive=True)
    image_files = [path for path in image_files if not os.path.isdir(path)]
    image_id_vs_path_dict = {}
    for image_path in image_files:
        image_id = os.path.split(image_path)[-1].split(".")[0]
        ocr_spatial_feature_path = os.path.join(OCR_SPATIAL_FEATURES_OCRVQA, image_id + ".npy")
        
        if os.path.exists(ocr_spatial_feature_path):
            continue
        
        import pdb
        pdb.set_trace()
        
        assert image_id not in image_id_vs_path_dict
        image_id_vs_path_dict[image_id] = {
            "image_path": image_path,
            "image_id": image_id,
            "ocr_feature_path": os.path.join(OCR_FEATURES_OCRVQA, image_id + ".npy")
        }

    # # assert len(ocr_feature_reader) == len(ocr_feature_files)
    # value_list = list(image_id_vs_path_dict.values())

    # # Add mapping for images vs
    # sp_pool = mp.Pool(64)
    # # map is synchronous (ordered)
    # result_list = list(tqdm.tqdm(sp_pool.imap(process_spatial_ocr, value_list),
    #                         total=len(value_list), desc="Spatial Relations"))
    # sp_pool.close()
    # sp_pool.join()
    # logger.info(f"Done Processsing Quadrant Spatial Relations with {64} threads")
