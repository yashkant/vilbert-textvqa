# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Requires vqa-maskrcnn-benchmark to be built and installed. See Readme
# for more details.
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from tqdm import tqdm
import os

from typing import List
import csv
import h5py
import numpy as np
import copy
import pickle
import lmdb  # install lmdb by "pip install lmdb"
import base64
import pdb


IMDB_OCRVQA_SPLITS = [
    "/srv/share/ykant3/ocr-vqa/intermediate/ocrvqa_train.npy",
    "/srv/share/ykant3/ocr-vqa/intermediate/ocrvqa_val.npy",
    "/srv/share/ykant3/ocr-vqa/intermediate/ocrvqa_test.npy",
]

IMAGES_OCRVQA = "/srv/share/ykant3/ocr-vqa/images/"

OCR_FEATURES_OCRVQA = "/srv/share/ykant3/ocr-vqa/features/"

def preprocess_ocr_tokens(info, prefix, suffix):
    """CRUX: given an entry return list of bboxes [x1,y1,x2,y2] and tokens"""
    height = info['image_height']
    width = info['image_width']
    ocr_info = info['{}ocr_info{}'.format(prefix, suffix)]

    boxes = np.zeros((len(ocr_info), 4), np.float32)
    tokens = [None]*len(ocr_info)
    for idx, entry in enumerate(ocr_info):
        tokens[idx] = entry['word']
        box = entry['bounding_box']
        x1 = box['top_left_x'] * width
        y1 = box['top_left_y'] * height
        w = box['width'] * width
        h = box['height'] * height
        x2 = x1 + w
        y2 = y1 + h
        boxes[idx] = [x1, y1, x2, y2]
    return boxes, tokens

class FeatureExtractor:
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.detection_model = self._build_detection_model()

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_file", default="../data/detectron_model.pth", type=str, help="Detectron model file"
        )
        parser.add_argument(
            "--config_file", default="../data/detectron_config.yaml", type=str, help="Detectron config file"
        )
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        parser.add_argument(
            "--partition", type=int, default=0, help="Partition to download."
        )
        return parser

    def preprocess_ocr_tokens(info, prefix, suffix):
        """CRUX: given an entry return list of bboxes [x1,y1,x2,y2] and tokens"""
        height = info['image_height']
        width = info['image_width']
        ocr_info = info['{}ocr_info{}'.format(prefix, suffix)]

        boxes = np.zeros((len(ocr_info), 4), np.float32)
        tokens = [None] * len(ocr_info)
        for idx, entry in enumerate(ocr_info):
            tokens[idx] = entry['word']
            box = entry['bounding_box']
            x1 = box['top_left_x'] * width
            y1 = box['top_left_y'] * height
            w = box['width'] * width
            h = box['height'] * height
            x2 = x1 + w
            y2 = y1 + h
            boxes[idx] = [x1, y1, x2, y2]
        return boxes, tokens

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def _image_transform(self, image_path):
        """Handles 2-dim and n-dim images. Resize between (800, 1333) and output scale"""
        img = Image.open(image_path)
        im = np.array(img).astype(np.float32)
        if im.ndim == 2:
            im = np.tile(im[:, :, None], (1, 1, 3))
        if im.shape[2] > 3:
            im = im[:, :, :3]
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(800) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > 1333:
            im_scale = float(1333) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        return img, im_scale, im_shape

    def get_detectron_features(self, image_path, input_boxes):
        # img_tensor, im_scales, im_infos = [], [], []
        im, im_scale, im_shape = self._image_transform(image_path)

        if input_boxes is not None:
            if isinstance(input_boxes, np.ndarray):
                input_boxes = torch.from_numpy(input_boxes.copy()).to("cuda")
            input_boxes *= im_scale
            input_boxes = [input_boxes]

        img_tensor, im_scales = [im], [im_scale]
        # im.shape: (C,H,W)
        # batches the images and decides the max_size
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to('cuda')

        with torch.no_grad():
            output = self.detection_model(current_img_list, input_boxes=input_boxes)

        feat = output[0][self.args.feature_name].cpu().numpy()
        bbox = output[0]["proposals"][0].bbox.cpu().numpy() / im_scale

        return feat, bbox

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def extract_features(self, imdb_data, image_ids, save_dir, image_dir):
        prefix, suffix = "google_", "_filtered"
        import pdb
        pdb.set_trace()

        image_ids_dict = {}
        for image_id in image_ids:
            image_ids_dict[image_id.decode()] = None
        
        import pdb
        pdb.set_trace()

        image_paths = []
        for instance in tqdm(imdb_data[1:]):
            image_id = instance["image_id"]
            if image_id not in image_ids_dict:
                image_path = os.path.join(IMAGES_OCRVQA, instance["image_path"])
                image_paths.append(image_path)
        import pdb
        pdb.set_trace()


        for image_path in tqdm(image_paths):
            image_paths.append(image_path)
            assert os.path.exists(image_path)
            image_w, image_h = Image.open(image_path).size
            instance["image_width"], instance["image_height"] = image_w, image_h
            ocr_boxes, ocr_tokens = preprocess_ocr_tokens(instance, prefix, suffix)
            try:
                if len(ocr_boxes) > 0:
                    extracted_feat, _ = self.get_detectron_features(image_path, input_boxes=ocr_boxes)
                else:
                    extracted_feat = np.zeros((0, 2048), np.float32)

                assert len(extracted_feat) == len(ocr_boxes)
                save_data = {
                    "ocr_boxes": ocr_boxes,
                    "ocr_tokens": ocr_tokens,
                    "features": extracted_feat
                }

                _save_dir = os.path.split(save_path)[0]
                if not os.path.exists(_save_dir):
                    os.makedirs(_save_dir)
                    print(f"Creating Dir: {_save_dir}")

                np.save(
                    save_path,
                    save_data
                )
            except BaseException:
                print("Couldn't extract from: ", instance["image_path"])


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    # with h5py.File(self.features_h5path, "r", libver='latest', swmr=True) as features_h5:
    # self._image_ids = list(features_h5["image_ids"])
    # If not loaded in memory, then list of None.
    env = lmdb.open(
        "/srv/share/ykant3/ocr-vqa/features/lmdbs/ocrvqa_ocr.lmdb",
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    with env.begin(write=False) as txn:
        _image_ids = pickle.loads(txn.get("keys".encode()))
        id = b'/srv/share/ykant3/ocr-vqa/features/ocr/1599215322.npy'
        import pdb
        pdb.set_trace()
        item = pickle.loads(txn.get(id))


    print(f"Extracting for: {IMDB_OCRVQA_SPLITS}")
    for imdb_file in IMDB_OCRVQA_SPLITS:
        print(f"Reading from: {imdb_file} \nSaving to: {OCR_FEATURES_OCRVQA}")
        imdb_data = np.load(imdb_file, allow_pickle=True)
        feature_extractor.extract_features(imdb_data, image_ids, OCR_FEATURES_OCRVQA, IMAGES_OCRVQA)

    # Todo: Check from LMDB the missing keys and only replace and handle them! 
