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


IMDB_SCENETEXT_RESPONSE_FIXED = [
    "/srv/share/ykant3/scene-text/test/imdb/test_task1_response_meta_fixed_processed.npy",
    "/srv/share/ykant3/scene-text/test/imdb/test_task2_response_meta_fixed_processed.npy",
]

OCR_FEATURES_SCENETEXT = [
    "/srv/share/ykant3/scene-text/features/ocr/test/test_task1/",
    "/srv/share/ykant3/scene-text/features/ocr/test/test_task2/",
]

# image_folders
IMAGES_SCENETEXT = [
    "/srv/share/ykant3/scene-text/test/test_task1/",
    "/srv/share/ykant3/scene-text/test/test_task2/",
]

IMAGES_COCOTEXT = "/srv/share/datasets/coco/train2014"

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
            "--model_file", default="../../data/detectron_model.pth", type=str, help="Detectron model file"
        )
        parser.add_argument(
            "--config_file", default="../../data/detectron_config.yaml", type=str, help="Detectron config file"
        )
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        # parser.add_argument(
        #     "--output_folder", type=str, default="/srv/share/ykant3/vilbert-mt/features/ocr/train/", help="Output folder"
        # )

        # parser.add_argument(
        #     "--output_folder", type=str, default="/srv/share/ykant3/vilbert-mt/features/ocr/test/", help="Output folder"
        # )
        #
        # parser.add_argument("--imdb_file",
        #                     type=str,
        #                     default="/nethome/ykant3/pythia/data/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_info_test.npy",
        #                     help="Image directory or file")

        # parser.add_argument("--imdb_file",
        #                     type=str,
        #                     default="/nethome/ykant3/pythia/data/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_train.npy",
        #                     help="Image directory or file")

        # parser.add_argument("--imdb_file",
        #                     type=str,
        #                     default="/nethome/ykant3/pythia/data/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_val.npy",
        #                     help="Image directory or file")

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

    def extract_features(self, imdb_data, save_dir, image_dir):
        # split = imdb_data[0]["dataset_type"]
        prefix, suffix = "google_", "_filtered"
        for instance in tqdm(imdb_data[1:]):
            image_path = instance["file_path"]

            if "COCO" in image_path:
                image_path = os.path.join(IMAGES_COCOTEXT, os.path.split(image_path)[-1])
            else:
                image_path = os.path.join(image_dir, image_path)

            assert os.path.exists(image_path)

            image_w, image_h = Image.open(image_path).size
            instance["image_width"], instance["image_height"] = image_w, image_h
            assert os.path.exists(image_path)
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

                if "COCO" in image_path:
                    save_path = os.path.join(image_dir, "coco-text", os.path.split(image_path)[-1].split(".")[0]) + ".npy"
                    save_path = save_path.replace(image_dir, save_dir)
                else:
                    save_path = image_path.replace(image_dir, save_dir)
                    save_path = save_path + ".npy"

                _save_dir = os.path.split(save_path)[0]
                if not os.path.exists(_save_dir):
                    os.makedirs(_save_dir)
                    print(f"Creating Dir: {_save_dir}")

                np.save(
                    save_path,
                    save_data
                )
            except BaseException:
                print("Couldn't extract from: ", instance["file_path"])


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    for imdb_file, save_dir, image_dir in zip(IMDB_SCENETEXT_RESPONSE_FIXED, OCR_FEATURES_SCENETEXT, IMAGES_SCENETEXT):
        print(f"Reading from: {imdb_file} \nSaving to: {save_dir}")
        imdb_data = np.load(imdb_file, allow_pickle=True)
        feature_extractor.extract_features(imdb_data, save_dir, image_dir)
        # np.save(imdb_file, imdb_data)
