# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Requires vqa-maskrcnn-benchmark to be built and installed. See Readme
# for more details.
import argparse
import glob
import os
import json
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


IMAGES_TEXTVQA = [
    "/srv/share/ykant3/pythia/dataset_images/train_images/",
    "/srv/share/ykant3/pythia/dataset_images/test_images/",
]

OI_OBJ_JSON = [
    "/srv/share3/hagrawal9/data/TextVQA_train_images_oidetector.json",
    "/srv/share3/hagrawal9/data/TextVQA_test_images_oidetector.json"
]

OI_OBJ_FEATURES = [
    "/srv/share/ykant3/vilbert-mt/features/oi-features/obj/train/",
    "/srv/share/ykant3/vilbert-mt/features/oi-features/obj/test/"
]


class FeatureExtractor:
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def preprocess_boxes(self, boxes, height, width):
        boxes[:, 0] = boxes[:, 0]*width
        boxes[:, 2] = boxes[:, 2]*width
        boxes[:, 1] = boxes[:, 1]*height
        boxes[:, 3] = boxes[:, 3]*height
        return boxes

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.detection_model = self._build_detection_model()

        os.makedirs(self.args.output_folder, exist_ok=True)

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
        parser.add_argument(
            "--output_folder", type=str, default="/srv/share/ykant3/", help="Output folder"
        )

        parser.add_argument(
            "--image_dir", type=str, default=None
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

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][start_index:], dim=1)
            cls_prob = torch.max(scores[keep_boxes][start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    "cls_prob": scores[keep_boxes].cpu().numpy(),
                }
            )

        return feat_list, info_list

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

    def _save_feature(self, save_path, feature, info):
        file_base_name = save_path.split(".")[0]
        info["image_id"] = os.path.split(file_base_name)[-1]
        file_base_name = file_base_name + ".npy"
        info["features"] = feature.cpu().numpy()
        np.save(file_base_name, info)

    def extract_features(self, json_path, save_dir, image_dir):

        with open(json_path) as file:
            json_data = json.load(file)

        # paths to all the items images
        files = glob.glob(image_dir + "/*", recursive=True)
        assert os.path.exists(image_dir)
        assert len(files) == len(json_data["image_url"])

        for idx, image_path in tqdm(json_data["image_url"].items()):
            try:
                abs_image_path = os.path.join(image_dir, image_path)
                assert os.path.exists(abs_image_path)
                image_w, image_h = Image.open(abs_image_path).size
                normalized_boxes = np.array(json_data["detection_boxes"][idx])
                bounding_boxes = self.preprocess_boxes(normalized_boxes, image_h, image_w)

                if len(bounding_boxes) > 0:
                    extracted_feat, _ = self.get_detectron_features(abs_image_path, input_boxes=bounding_boxes)
                else:
                    extracted_feat = np.zeros((0, 2048), np.float32)

                assert len(extracted_feat) == len(bounding_boxes)

                save_data = {
                    "bounding_boxes": bounding_boxes,
                    "features": extracted_feat,
                    "detection_class_entities": json_data["detection_class_entities"][idx],
                    "detection_class_names": json_data["detection_class_names"][idx],
                    "detection_boxes": json_data["detection_boxes"][idx],
                    "detection_scores": json_data["detection_scores"][idx],
                    "detection_class_labels": json_data["detection_class_labels"][idx],
                    "image_url": json_data["image_url"][idx],
                }
                save_path = abs_image_path.replace(image_dir, save_dir).split(".")[0]
                save_path = save_path + ".npy"

                _save_dir = os.path.split(save_path)[0]
                if not os.path.exists(_save_dir):
                    os.makedirs(_save_dir)
                    print(f"Creating Dir: {_save_dir}")

                np.save(save_path, save_data)
            except BaseException:
                print("Couldn't extract from: ", abs_image_path)


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()

    for json_path, save_dir, image_dir in zip(OI_OBJ_JSON, OI_OBJ_FEATURES, IMAGES_TEXTVQA):
        print(f"Extracting from: {image_dir} \nSaving to: {save_dir}")
        feature_extractor.extract_features(json_path, save_dir, image_dir)

