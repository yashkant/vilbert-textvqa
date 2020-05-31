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
from vilbert.spatial_utils_regat import torch_broadcast_adj_matrix
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
        imdb_path = f"/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_{name}.npy"
        if name == "test":
            imdb_path = "/srv/share/ykant3/scene-text/test/imdb/test_task3_response_meta_fixed_processed.npy"
        if debug:
            imdb_path = f"/srv/share/ykant3/scene-text/train/imdb/debug_train_task_response_meta_fixed_processed_{name}.npy"
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
        entry = dict([(key, instance[key]) for key in store_keys if key in instance])
        # Also need to add features-dir
        entry["image_id"] = entry["image_path"].split(".")[0] + ".npy"
        entries.append(entry)

    return entries, imdb_data.metadata


class STVQADataset(TextVQADataset):
    def __init__(
        self,
        split,
        tokenizer,
        bert_model,
        task="STVQA",
        padding_index=0,
        processing_threads=32,
        task_cfg=None
    ):
        """
        (YK): Builds self.entries by reading questions and answers and caches them.
        """

        # Just initialize the grand-parent classs
        Dataset.__init__(self)

        dataroot = task_cfg["stvqa_dataroot"]
        self.split = split
        self._max_seq_length = task_cfg["max_seq_length"]

        if self.split == "test":
            self.obj_features_reader = ImageFeaturesH5Reader(
                features_path="/srv/share/ykant3/scene-text/features/obj/lmdbs/test_task3_fixed.lmdb", in_memory=True
            )
            self.ocr_features_reader = ImageFeaturesH5Reader(
                features_path="/srv/share/ykant3/scene-text/features/ocr/lmdbs/test_task3_fixed.lmdb", in_memory=True
            )
        else:
            self.obj_features_reader = ImageFeaturesH5Reader(
                features_path=task_cfg["stvqa_features_h5path1"], in_memory=True
            )
            self.ocr_features_reader = ImageFeaturesH5Reader(
                features_path=task_cfg["stvqa_features_h5path2"], in_memory=True
            )

        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.max_obj_num = task_cfg["max_obj_num"]
        self.max_ocr_num = task_cfg["max_ocr_num"]
        assert self.max_obj_num == 100
        assert self.max_ocr_num == 50
        self.max_resnet_num = task_cfg["max_resnet_num"]
        self.debug = task_cfg.get("debug", False)
        self.vocab_type = task_cfg.get("vocab_type", "4k")
        self.dynamic_sampling = task_cfg.get("dynamic_sampling", True)
        self.distance_threshold = task_cfg.get("distance_threshold", 0.5)
        self.processing_threads = processing_threads
        self.heads_type = task_cfg.get("heads_type", "none")
        self.clean_answers = task_cfg.get("clean_answers", True)
        self.randomize = task_cfg.get("randomize", -1)
        self.needs_spatial = False
        self.use_gauss_bias = task_cfg.get("use_gauss_bias", False)
        self.gauss_bias_dev_factor = task_cfg.get("gauss_bias_dev_factor", -1.0)
        self.use_attention_bins = task_cfg.get("use_attention_bins", False)
        self.attention_bins = task_cfg.get("attention_bins", [-1])
        self.matrix_type_map = {
            "share3": ["3"],
            "share5": ["3", "5"],
            "share7": ["3", "5", "7"],
            "share9": ["3", "5", "7", "9"],
        }
        self.restrict_oo = task_cfg.get("restrict_oo", False)
        self.extra_args = task_cfg

        if ( ("num_spatial_layers" in task_cfg and task_cfg["num_spatial_layers"] > 0) or
             ("layer_type_list" in task_cfg and "s" in task_cfg["layer_type_list"])):
            self.needs_spatial = True


        registry.vocab_type = self.vocab_type
        registry.distance_threshold = self.distance_threshold
        registry.randomize = self.randomize
        registry.use_gauss_bias = self.use_gauss_bias
        registry.gauss_bias_dev_factor = self.gauss_bias_dev_factor
        registry.mix_list = task_cfg.get("mix_list", ["none"])
        registry.use_attention_bins = self.use_attention_bins
        registry.attention_bins = self.attention_bins
        registry.restrict_oo = self.restrict_oo

        logger.info(f"Dynamic Sampling is {self.dynamic_sampling}")
        logger.info(f"distance_threshold is {self.distance_threshold}")
        logger.info(f"heads_type: {self.heads_type}")
        logger.info(f"Clean Answers is {self.clean_answers}")
        logger.info(f"Randomize is {self.randomize}")
        logger.info(f"needs_spatial is {self.needs_spatial}")
        logger.info(f"use_gauss_bias is {self.use_gauss_bias}")
        logger.info(f"gauss_bias_dev is {self.gauss_bias_dev_factor}")
        logger.info(f"restrict_oo is {self.restrict_oo}")

        head_types = []
        if "mix_list" in self.extra_args:
            for head_type in set(self.extra_args["mix_list"]):
                if head_type in self.matrix_type_map:
                    head_types.extend(self.matrix_type_map[head_type])

        self.head_types = list(set(head_types))
        self.head_types.sort()

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
                + str(self._max_seq_length)
                + clean_train
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task + "_" + split + "_" + str(self._max_seq_length) + clean_train + f"_vocab_type{self.vocab_type}"
                + f"_dynamic_{self.dynamic_sampling}" + ".pkl",
            )

        if self.distance_threshold != 0.5:
            raise AssertionError
            cache_path = cache_path.split(".")[0]
            cache_path = cache_path + f"_threshold_{self.distance_threshold}" + ".pkl"

        if self.randomize > 0:
            cache_path = cache_path.split(".")[0]
            cache_path = cache_path + f"_randomize_{self.randomize}" + ".pkl"
        elif self.needs_spatial:
            cache_path = cache_path.split(".")[0]
            cache_path = cache_path + f"_spatial" + ".pkl"
            if self.use_gauss_bias:
                _types = list(set(registry.mix_list))
                _types.sort()
                _types = "-".join(_types)
                cache_path = cache_path.split(".")[0]
                cache_path = cache_path + f"_gauss_factor_{self.gauss_bias_dev_factor}_heads_type_{_types}" + ".pkl"

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

            if self.randomize > 0:
                raise AssertionError
                self.process_random_spatials()
            elif self.needs_spatial:
                self.process_spatials()

            if not self.debug:
                cPickle.dump(self.entries, open(cache_path, "wb"))
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