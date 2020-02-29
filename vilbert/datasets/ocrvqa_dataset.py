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

        if self.heads_type != "none" or self.randomize > 0:
            spatial_cache_path = cache_path.split(".")[0]
            spatial_cache_path = spatial_cache_path + f"_spatials" + ".pkl"

        logger.info(f"Cache Name:  {cache_path}")

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
            # Adding spatial graph matrix
            obj_features, obj_num_boxes, obj_bboxes, _ = self._image_features_reader[entry["image_id"]]
            obj_features, obj_num_boxes, obj_bboxes = obj_features[1:], obj_num_boxes - 1, obj_bboxes[1:]
            _, _, pad_obj_bboxes = self._pad_features(
                obj_features, obj_bboxes, obj_num_boxes, self._max_region_num, tensorize=False
            )
            pad_obj_bboxes = pad_obj_bboxes[:, :-1]
            # Append bboxes to the list
            image_id_obj_list[entry["image_id"]] = pad_obj_bboxes

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

