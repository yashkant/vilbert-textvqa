# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import logging
from tools.registry import registry
from tools.objects_to_byte_tensor import enc_obj2bytes, dec_bytes2obj
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .textvqa_processors import *
from easydict import EasyDict as edict
from ._image_features_reader import ImageFeaturesH5Reader
from vilbert.spatial_utils_regat import torch_broadcast_adj_matrix, build_graph_using_normalized_boxes_new, \
    random_spatial_processor
import multiprocessing as mp
from vilbert.datasets.textvqa_dataset import TextVQADataset, Processors, ImageDatabase

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_dataset(name, debug):
    """Load entries from Imdb

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'

    (YK): We load questions and answers corresponding to
        the splits, and return entries.
    """

    if name == "train" or name == "val" or name == "test":
        imdb_path = f"/srv/share/ykant3/ocr-vqa/intermediate/ocrvqa_{name}.npy"
        if debug:
            imdb_path = f"/srv/share/ykant3/ocr-vqa/intermediate/ocrvqa_debug.npy"
        logger.info(f"Loading IMDB for {name}" if not debug else f"Loading IMDB for {name} in debug mode")
        imdb_data = ImageDatabase(imdb_path)
    else:
        assert False, "data split is not recognized."

    # build entries with only the essential keys
    entries = []
    store_keys = [
        "question",
        "question_id",
        "image_path",
        "answers",
        "image_height",
        "image_width",
        "google_ocr_tokens_filtered",
        # "google_ocr_info_filtered",
    ]

    logger.info(f"Building Entries for {name}")
    for instance in imdb_data:
        entry = dict([(key, instance[key]) for key in store_keys])
        # Also need to add features-dir
        entry["image_id"] = entry["image_path"].split(".")[0] + ".npy"
        entries.append(entry)

    return entries, imdb_data.metadata


class OCRVQADataset(TextVQADataset):
    def __init__(
        self,
        split,
        tokenizer,
        bert_model,
        task="OCRVQA",
        padding_index=0,
        max_seq_length=16,
        max_region_num=101,
        processing_threads=32,
        extra_args=None
    ):
        """
        (YK): Builds self.entries by reading questions and answers and caches them.
        """

        # Just initialize the grand-parent classs
        Dataset.__init__(self)

        dataroot = extra_args["ocrvqa_dataroot"]
        self.split = split
        self._max_seq_length = max_seq_length
        self.obj_features_reader = ImageFeaturesH5Reader(
            features_path=extra_args["ocrvqa_features_h5path1"], in_memory=True
        )
        self.ocr_features_reader = ImageFeaturesH5Reader(
            features_path=extra_args["ocrvqa_features_h5path2"], in_memory=True
        )
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.max_obj_num = extra_args["max_obj_num"]
        self.max_ocr_num = extra_args["max_ocr_num"]
        assert self.max_obj_num == 100
        assert self.max_ocr_num == 50
        self.max_resnet_num = extra_args["max_resnet_num"]
        self.debug = extra_args.get("debug", False)
        self.vocab_type = extra_args["vocab_type"]
        self.dynamic_sampling = extra_args.get("dynamic_sampling", True)
        self.distance_threshold = extra_args.get("distance_threshold", 0.5)
        self.processing_threads = processing_threads
        self.heads_type = extra_args.get("heads_type", "none")
        self.clean_answers = extra_args.get("clean_answers", True)
        self.randomize = extra_args.get("randomize", -1)
        self.spatials = {}

        registry.vocab_type = self.vocab_type
        registry.distance_threshold = self.distance_threshold
        registry.randomize = self.randomize

        logger.info(f"Dynamic Sampling is {self.dynamic_sampling}")
        logger.info(f"distance_threshold is {self.distance_threshold}")
        logger.info(f"heads_type: {self.heads_type}")
        logger.info(f"Clean Answers is {self.clean_answers}")
        logger.info(f"Randomize is {self.randomize}")

        # We only randomize the spatial-adj-matrix
        if self.heads_type != "none":
            assert self.randomize <= 0

        clean_train = ""

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + clean_train
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task + "_" + split + "_" + str(max_seq_length) + clean_train + f"_vocab_type{self.vocab_type}"
                + f"_dynamic_{self.dynamic_sampling}" + ".pkl",
            )
        logger.info(f"Cache Name:  {cache_path}")

        if self.heads_type != "none" or self.randomize > 0:
            spatial_cache_path = cache_path.split(".")[0]
            spatial_cache_path = spatial_cache_path + f"_spatials" + ".pkl"
            logger.info(f"Spatial Cache Name:  {spatial_cache_path}")


        if not os.path.exists(cache_path) or self.debug:
            # Initialize Processors

            if "processors" not in registry:
                self.processors = Processors(self._tokenizer, vocab_type=self.vocab_type)
                registry.processors = self.processors
            else:
                self.processors = registry.processors

            self.entries, _ = _load_dataset(split, self.debug)
            # convert questions to tokens, create masks, segment_ids
            self.process()

            if self.randomize > 0 or self.heads_type != "none":
                self.process_spatials()

            if not self.debug:
                cPickle.dump(self.entries, open(cache_path, "wb"))
                if self.randomize > 0 or self.heads_type != "none":
                    cPickle.dump(self.spatials, open(spatial_cache_path, "wb"))
        else:
            if "processors_only_registry" not in registry:
                self.processors = Processors(
                    self._tokenizer,
                    only_registry=True,
                    vocab_type=self.vocab_type
                )  # only initialize the M4C processor (for registry)
                registry.processors_only_registry = self.processors
            else:
                self.processors = registry.processors_only_registry

            # otherwise load cache!
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

            if self.randomize > 0 or self.heads_type != "none":
                self.spatials = cPickle.load(open(spatial_cache_path, "rb"))

    def process_spatials(self):
        logger.info(f"Processsing Share/Single/Random Spatial Relations with {self.processing_threads} threads")
        import multiprocessing as mp

        image_id_obj_list = {}

        for entry in tqdm(self.entries, desc="Reading Entries"):
            if entry["image_id"] in image_id_obj_list:
                continue

            obj_features, obj_num_boxes, obj_bboxes, _ = self.obj_features_reader[entry["image_id"]]
            obj_features, obj_num_boxes, obj_bboxes = obj_features[1:], obj_num_boxes - 1, obj_bboxes[1:]
            _, _, pad_obj_bboxes = self._pad_features(
                obj_features, obj_bboxes, obj_num_boxes, self.max_obj_num, tensorize=False
            )
            ocr_features, ocr_num_boxes, ocr_bboxes, _ = self.ocr_features_reader[entry["image_id"]]
            ocr_features, ocr_num_boxes, ocr_bboxes = ocr_features[1:], ocr_num_boxes - 1, ocr_bboxes[1:]
            _, _, pad_ocr_bboxes = self._pad_features(
                ocr_features, ocr_bboxes, ocr_num_boxes, self.max_ocr_num, tensorize=False
            )

            # Append bboxes to the list
            pad_obj_ocr_bboxes = np.concatenate([pad_obj_bboxes[:, :-1], pad_ocr_bboxes[:, :-1]], axis=0)
            image_id_obj_list[entry["image_id"]] = pad_obj_ocr_bboxes

        image_ids = []
        obj_lists = []
        for image_id, obj_list in image_id_obj_list.items():
            image_ids.append(image_id)
            obj_lists.append(obj_list)



        # Add mapping for images vs
        sp_pool = mp.Pool(self.processing_threads)
        # map is synchronous (ordered)
        result_list = list(tqdm(sp_pool.imap(OCRVQADataset.process_all_spatials, obj_lists),
                                     total=len(obj_lists), desc="Spatial Relations"))
        sp_pool.close()
        sp_pool.join()
        logger.info(f"Done Processsing Quadrant Spatial Relations with {self.processing_threads} threads")
        assert len(result_list) == len(obj_lists)


        for image_id, (adj_matrix, adj_matrix_share3_1, adj_matrix_share3_2, adj_matrix_random1, adj_matrix_random3) \
                in zip(image_ids, result_list):
            self.spatials[image_id] = {}
            self.spatials[image_id]["spatial_adj_matrix"] = adj_matrix
            self.spatials[image_id]["spatial_adj_matrix_share3_1"] = adj_matrix_share3_1
            self.spatials[image_id]["spatial_adj_matrix_share3_2"] = adj_matrix_share3_2
            self.spatials[image_id]["spatial_adj_matrix_random1"] = adj_matrix_random1
            self.spatials[image_id]["spatial_adj_matrix_random3"] = adj_matrix_random3


    @staticmethod
    def process_all_spatials(pad_obj_bboxes):
        adj_matrix, adj_matrix_share3_1, adj_matrix_share3_2 \
            = build_graph_using_normalized_boxes_new(pad_obj_bboxes, distance_threshold=0.5)
        adj_matrix_random1, adj_matrix_random3 = random_spatial_processor(pad_obj_bboxes)
        return adj_matrix, adj_matrix_share3_1, adj_matrix_share3_2, adj_matrix_random1, adj_matrix_random3


    def __getitem__(self, index):
        """
        1. Get image-features/bboxes and image mask (as nump-arrays), tensorize them.
        2. Get question, input_mask, segment_ids and coattention mask
        3. Build target (vocab-dim) with VQA scores scattered at label-indices
        4. Return
        """
        entry = self.entries[index]
        image_id = entry["image_id"]

        # add object-features and bounding boxes
        obj_features, obj_num_boxes, obj_bboxes, _ = self.obj_features_reader[image_id]
        # remove avg-features
        obj_features, obj_num_boxes, obj_bboxes = obj_features[1:], obj_num_boxes - 1, obj_bboxes[1:]
        pad_obj_features, pad_obj_mask, pad_obj_bboxes = self._pad_features(
            obj_features, obj_bboxes, obj_num_boxes, self.max_obj_num
        )

        # add ocr-features and bounding boxes
        ocr_features, ocr_num_boxes, ocr_bboxes, _ = self.ocr_features_reader[image_id]
        # remove avg-features
        ocr_features, ocr_num_boxes, ocr_bboxes = ocr_features[1:], ocr_num_boxes - 1, ocr_bboxes[1:]
        pad_ocr_features, pad_ocr_mask, pad_ocr_bboxes = self._pad_features(
            ocr_features, ocr_bboxes, ocr_num_boxes, self.max_ocr_num
        )

        co_attention_mask = torch.cat((pad_obj_mask, pad_ocr_mask))
        co_attention_mask = co_attention_mask.unsqueeze(1).repeat(1, self._max_seq_length)
        segment_ids = torch.zeros_like(entry["question_mask"])

        item = edict({
            "pad_obj_features": pad_obj_features,
            "pad_obj_mask": pad_obj_mask,
            "pad_obj_bboxes": pad_obj_bboxes,
            "pad_ocr_features": pad_ocr_features,
            "pad_ocr_mask": pad_ocr_mask,
            "pad_ocr_bboxes": pad_ocr_bboxes,
            "segment_ids": segment_ids
        })

        item["co_attention_mask"] = co_attention_mask

        # process answers (dynamic sampling)
        if self.clean_answers:
            cleaned_answers = [Processors.word_cleaner(word) for word in entry["answers"]]
        else:
            cleaned_answers = entry["answers"]

        cleaned_ocr_tokens = entry["cleaned_ocr_tokens"]
        processed_answers = self.processors.answer_processor({
            "answers": cleaned_answers,
            "context_tokens": cleaned_ocr_tokens,
        })
        entry.update(processed_answers)


        if self.heads_type in ["mix", "share3", "quad4"] or self.randomize > 0:
            spatials = self.spatials[image_id]
            # In the first iteration expand all the spatial relation matrices

            if not isinstance(spatials["spatial_adj_matrix"], torch.Tensor):
                if self.randomize > 0:
                    rows, cols, slices = spatials["spatial_adj_matrix"].shape
                    assert slices == self.randomize
                    adj_matrices = []

                    # Expand each slice
                    for slice_idx in range(slices):
                        adj_matrix_slice = torch_broadcast_adj_matrix(
                            torch.from_numpy(spatials["spatial_adj_matrix"][:, :, slice_idx]),
                            label_num=12
                        )
                        adj_matrices.append(adj_matrix_slice)

                    # Aggregate each slice
                    spatials["spatial_adj_matrix"] = adj_matrices[0]
                    for matrix_slice in adj_matrices[1:]:
                        spatials["spatial_adj_matrix"] = torch.max(spatials["spatial_adj_matrix"], matrix_slice)

                    entry["spatial_loss_mask"] = entry["spatial_ocr_relations"] = None
                else:
                    spatials["spatial_adj_matrix"] = torch_broadcast_adj_matrix(
                        torch.from_numpy(spatials["spatial_adj_matrix"]),
                        label_num=12
                    )

                if self.heads_type == "mix":
                    spatial_adj_matrix_share3_1 = torch_broadcast_adj_matrix(
                        torch.from_numpy(spatials["spatial_adj_matrix_share3_1"]),
                        label_num=12
                    )
                    spatial_adj_matrix_share3_2 = torch_broadcast_adj_matrix(
                        torch.from_numpy(spatials["spatial_adj_matrix_share3_2"]),
                        label_num=12
                    )
                    spatials["spatial_adj_matrix_share3"] = torch.max(spatials["spatial_adj_matrix"],
                                                                   spatial_adj_matrix_share3_1)
                    spatials["spatial_adj_matrix_share3"] = torch.max(spatials["spatial_adj_matrix"],
                                                                   spatial_adj_matrix_share3_2)
            else:
                try:
                    assert len(spatials["spatial_adj_matrix"].shape) == 3
                except:
                    import pdb
                    pdb.set_trace()
            item.update(spatials)
        item.update(entry)


        # remove unwanted keys
        unwanted_keys = ['spatial_adj_matrix_share3_1',
                         'spatial_adj_matrix_share3_2',
                         'spatial_adj_matrix_random1',
                         'spatial_adj_matrix_random3',
                         'cleaned_ocr_tokens',
                         'image_id',
                         'image_path']

        if self.heads_type == "share3":
            unwanted_keys.append("spatial_adj_matrix_quad4")

        for key in unwanted_keys:
            if key in item:
                item.pop(key, None)

        # Collate Function doesn't work correctly with lists
        for key, value in item.items():
            if not isinstance(value, torch.Tensor):
                try:
                    item[key] = enc_obj2bytes(value)
                except:
                    import pdb
                    pdb.set_trace()

        return item
