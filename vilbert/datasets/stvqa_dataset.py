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
from ._image_features_reader import ImageFeaturesH5Reader, CacheH5Reader
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

        dataroot = extra_args["stvqa_dataroot"]
        self.split = split
        self._max_seq_length = max_seq_length

        if self.split == "test":
            self.obj_features_reader = ImageFeaturesH5Reader(
                features_path="/srv/share/ykant3/scene-text/features/obj/lmdbs/test_task3_fixed.lmdb", in_memory=False
            )
            self.ocr_features_reader = ImageFeaturesH5Reader(
                features_path="/srv/share/ykant3/scene-text/features/ocr/lmdbs/test_task3_fixed.lmdb", in_memory=False
            )
        else:
            self.obj_features_reader = ImageFeaturesH5Reader(
                features_path=extra_args["stvqa_features_h5path1"], in_memory=False
            )
            self.ocr_features_reader = ImageFeaturesH5Reader(
                features_path=extra_args["stvqa_features_h5path2"], in_memory=False
            )

        self.spatial_reader = CacheH5Reader(
            features_path = f"/srv/share/ykant3/vilbert-mt/stvqa/cache/STVQA_{self.split}_20_vocab_type5k_stvqa_dynamic_True_spatial.lmdb",
            in_memory=False
        )

        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.max_obj_num = extra_args["max_obj_num"]
        self.max_ocr_num = extra_args["max_ocr_num"]
        assert self.max_obj_num == 100
        assert self.max_ocr_num == 50
        self.max_resnet_num = extra_args["max_resnet_num"]
        self.debug = extra_args.get("debug", False)
        self.vocab_type = extra_args.get("vocab_type", "4k")
        self.dynamic_sampling = extra_args.get("dynamic_sampling", True)
        self.distance_threshold = extra_args.get("distance_threshold", 0.5)
        self.processing_threads = processing_threads
        self.heads_type = extra_args.get("heads_type", "none")
        self.clean_answers = extra_args.get("clean_answers", True)
        self.randomize = extra_args.get("randomize", -1)
        self.needs_spatial = False
        self.use_gauss_bias = extra_args.get("use_gauss_bias", False)
        self.gauss_bias_dev_factor = extra_args.get("gauss_bias_dev_factor", -1.0)
        self.use_attention_bins = extra_args.get("use_attention_bins", False)
        self.attention_bins = extra_args.get("attention_bins", [-1])
        self.matrix_type_map = {
            "share3": ["3"],
            "share5": ["3", "5"],
            "share7": ["3", "5", "7"],
            "share9": ["3", "5", "7", "9"],
        }
        self.restrict_oo = extra_args.get("restrict_oo", False)
        self.extra_args = extra_args
        self.stvqa_task_type = extra_args.get("stvqa_task_type", 3)

        if ( ("num_spatial_layers" in extra_args and extra_args["num_spatial_layers"] > 0) or
             ("layer_type_list" in extra_args and "s" in extra_args["layer_type_list"]) ):
            self.needs_spatial = True


        registry.vocab_type = self.vocab_type
        registry.distance_threshold = self.distance_threshold
        registry.randomize = self.randomize
        registry.use_gauss_bias = self.use_gauss_bias
        registry.gauss_bias_dev_factor = self.gauss_bias_dev_factor
        registry.mix_list = extra_args.get("mix_list", ["none"])
        registry.use_attention_bins = self.use_attention_bins
        registry.attention_bins = self.attention_bins
        registry.restrict_oo = self.restrict_oo
        registry.stvqa_task_type = self.stvqa_task_type

        logger.info(f"Dynamic Sampling is {self.dynamic_sampling}")
        logger.info(f"distance_threshold is {self.distance_threshold}")
        logger.info(f"heads_type: {self.heads_type}")
        logger.info(f"Clean Answers is {self.clean_answers}")
        logger.info(f"Randomize is {self.randomize}")
        logger.info(f"needs_spatial is {self.needs_spatial}")
        logger.info(f"use_gauss_bias is {self.use_gauss_bias}")
        logger.info(f"gauss_bias_dev is {self.gauss_bias_dev_factor}")
        logger.info(f"restrict_oo is {self.restrict_oo}")
        logger.info(f"stvqa_task_type is {self.stvqa_task_type}")

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

        # if self.distance_threshold != 0.5:
        #     raise AssertionError
        #     cache_path = cache_path.split(".")[0]
        #     cache_path = cache_path + f"_threshold_{self.distance_threshold}" + ".pkl"

        # if self.randomize > 0:
        #     cache_path = cache_path.split(".")[0]
        #     cache_path = cache_path + f"_randomize_{self.randomize}" + ".pkl"
        # elif self.needs_spatial:
        #     cache_path = cache_path.split(".")[0]
        #     cache_path = cache_path + f"_spatial" + ".pkl"
        #     if self.use_gauss_bias:
        #         _types = list(set(registry.mix_list))
        #         _types.sort()
        #         _types = "-".join(_types)
        #         cache_path = cache_path.split(".")[0]
        #         cache_path = cache_path + f"_gauss_factor_{self.gauss_bias_dev_factor}_heads_type_{_types}" + ".pkl"

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

            # if self.randomize > 0:
            #     raise AssertionError
            #     self.process_random_spatials()
            # elif self.needs_spatial:
            #     self.process_spatials()

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

    def __getitem__(self, index):
        """
        1. Get image-features/bboxes and image mask (as nump-arrays), tensorize them.
        2. Get question, input_mask, segment_ids and coattention mask
        3. Build target (vocab-dim) with VQA scores scattered at label-indices
        4. Return
        """
        entry = self.entries[index]
        image_id = entry["image_id"]

        if self.needs_spatial:
            entry["spatial_adj_matrix_shared"] = self.spatial_reader[entry["image_id"]]

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

        if "answers" in entry:
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
        else:
            # Empty placeholder
            entry["train_prev_inds"] = torch.zeros(12, dtype=torch.long)

        if self.needs_spatial:
            # In the first iteration expand all the spatial relation matrices
            if "spatial_adj_matrices" not in entry:
                if self.randomize > 0:
                    rows, cols, slices = entry["spatial_adj_matrix"].shape
                    assert slices == self.randomize
                    adj_matrices = []

                    # Expand each slice
                    for slice_idx in range(slices):
                        adj_matrix_slice = torch_broadcast_adj_matrix(
                            torch.from_numpy(entry["spatial_adj_matrix"][:, :, slice_idx]),
                            label_num=12
                        )
                        adj_matrices.append(adj_matrix_slice)

                    # Aggregate each slice
                    entry["spatial_adj_matrix"] = adj_matrices[0]
                    for matrix_slice in adj_matrices[1:]:
                        entry["spatial_adj_matrix"] = torch.max(entry["spatial_adj_matrix"], matrix_slice)

                    entry["spatial_loss_mask"] = entry["spatial_ocr_relations"] = None
                else:
                    entry["spatial_adj_matrices"] = {}
                    entry["gauss_bias_matrices"] = {}
                    entry["bins_matrices"] = {}

                    build_map = {
                        "3": ["1", "31", "32"],
                        "5": ["3", "51", "52"],
                        "7": ["5", "71", "72"],
                        "9": ["7", "91", "92"],
                    }

                    entry["spatial_adj_matrices"]["1"] = torch_broadcast_adj_matrix(
                        torch.from_numpy(entry["spatial_adj_matrix_shared"]["1"]),
                        label_num=12
                    )

                    entry["spatial_adj_matrices"]["full_spatial"] = \
                        (torch.from_numpy(entry["spatial_adj_matrix_shared"]["1"]) != 0).int()

                    if self.use_gauss_bias:
                        entry["gauss_bias_matrices"]["1"] = torch_broadcast_gauss_bias(
                            torch.from_numpy(entry["spatial_adj_matrix_shared"]["1"]),
                            torch.from_numpy(entry["spatial_gauss_bias_shared"]["1"])
                        )

                    if self.use_attention_bins:
                        entry["bins_matrices"]["1"] = torch_broadcast_bins(
                            torch.from_numpy(entry["spatial_adj_matrix_shared"]["1"]),
                            1
                        )

                    for head_type in self.head_types:
                        use_matrix_types = build_map[head_type]
                        assert use_matrix_types[0] in entry["spatial_adj_matrices"]
                        init_matrix = entry["spatial_adj_matrices"][use_matrix_types[0]]
                        first_matrix = torch_broadcast_adj_matrix(
                            torch.from_numpy(entry["spatial_adj_matrix_shared"][use_matrix_types[1]]),
                            label_num=12
                        )
                        second_matrix = torch_broadcast_adj_matrix(
                            torch.from_numpy(entry["spatial_adj_matrix_shared"][use_matrix_types[2]]),
                            label_num=12
                        )
                        init_matrix = torch.max(init_matrix, first_matrix)
                        init_matrix = torch.max(init_matrix, second_matrix)
                        entry["spatial_adj_matrices"][head_type] = init_matrix

                        if self.use_gauss_bias:
                            assert use_matrix_types[0] in entry["spatial_gauss_bias_shared"]
                            gauss_init_matrix = entry["gauss_bias_matrices"][use_matrix_types[0]]
                            gauss_first_matrix = torch_broadcast_gauss_bias(
                                torch.from_numpy(entry["spatial_adj_matrix_shared"][use_matrix_types[1]]),
                                torch.from_numpy(entry["spatial_gauss_bias_shared"][use_matrix_types[1]]),
                            )
                            gauss_second_matrix = torch_broadcast_gauss_bias(
                                torch.from_numpy(entry["spatial_adj_matrix_shared"][use_matrix_types[2]]),
                                torch.from_numpy(entry["spatial_gauss_bias_shared"][use_matrix_types[2]]),
                            )
                            gauss_init_matrix = torch.max(gauss_init_matrix, gauss_first_matrix)
                            gauss_init_matrix = torch.max(gauss_init_matrix, gauss_second_matrix)
                            entry["gauss_bias_matrices"][head_type] = gauss_init_matrix

                        if self.use_attention_bins:
                            bins_init_matrix = entry["bins_matrices"][use_matrix_types[0]]
                            bins_first_matrix = torch_broadcast_bins(
                                torch.from_numpy(entry["spatial_adj_matrix_shared"][use_matrix_types[1]]),
                                (int(use_matrix_types[1][0]) + 1) / 2
                            )
                            bins_second_matrix = torch_broadcast_bins(
                                torch.from_numpy(entry["spatial_adj_matrix_shared"][use_matrix_types[2]]),
                                (int(use_matrix_types[2][0]) + 1) / 2
                            )
                            bins_init_matrix = torch.max(bins_init_matrix, bins_first_matrix)
                            bins_init_matrix = torch.max(bins_init_matrix, bins_second_matrix)
                            entry["bins_matrices"][head_type] = bins_init_matrix

        item.update(entry)

        # remove unwanted keys
        unwanted_keys_item = ['spatial_gauss_bias_shared',
                              'spatial_adj_matrix_shared',
                              'cleaned_ocr_tokens',
                              'image_id',
                              'image_path']

        for key in unwanted_keys_item:
            if key in item:
                item.pop(key, None)

        unwanted_keys_entry = [
            'spatial_adj_matrices',
            'gauss_bias_matrices',
            'bin_matrices',
        ]

        for key in unwanted_keys_entry:
            if key in entry:
                entry.pop(key, None)

        # Collate Function doesn't work correctly with lists
        for key, value in item.items():
            if not isinstance(value, torch.Tensor) and not isinstance(value, dict):
                try:
                    item[key] = enc_obj2bytes(value)
                except:
                    print(key)
                    import pdb
                    pdb.set_trace()

        return item
