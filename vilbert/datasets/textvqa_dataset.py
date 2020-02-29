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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_dataset(dataroot, name, debug):
    """Load entries from Imdb

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'

    (YK): We load questions and answers corresponding to
        the splits, and return entries.
    """

    if name == "train" or name == "val" or name == "test":
        imdb_holder = "imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_{}.npy"
        if name == "test":
            imdb_holder = "imdb_google_det_bbox_textvqa_info_{}.npy"

        if debug:
            imdb_holder = "debug_" + imdb_holder

        imdb_path = os.path.join(dataroot, "imdb/textvqa_0.5/", imdb_holder.format(name))
        logger.info(f"Loading IMDB for {name}" if not debug else f"Loading IMDB for {name} in debug mode")
        imdb_data = ImageDatabase(imdb_path)
    else:
        assert False, "data split is not recognized."

    # build entries with only the essential keys
    entries = []
    store_keys = [
        "question",
        "question_id",
        "image_id",
        "answers",
        "image_height",
        "image_width",
        "google_ocr_tokens_filtered",
        # "google_ocr_info_filtered",
    ]

    logger.info(f"Building Entries for {name}")
    for instance in imdb_data:
        entry = dict([(key, instance[key]) for key in store_keys])
        entries.append(entry)

    return entries, imdb_data.metadata


class TextVQADataset(Dataset):
    def __init__(
            self,
            split,
            tokenizer,
            bert_model,
            task="TextVQA",
            padding_index=0,
            max_seq_length=16,
            max_region_num=101,
            processing_threads=32,
            extra_args=None
    ):
        """
        (YK): Builds self.entries by reading questions and answers and caches them.
        """
        super().__init__()
        dataroot = extra_args["dataroot"]
        self.split = split
        self._max_seq_length = max_seq_length
        self.obj_features_reader = ImageFeaturesH5Reader(
            features_path=extra_args["features_h5path1"], in_memory=True
        )
        self.ocr_features_reader = ImageFeaturesH5Reader(
            features_path=extra_args["features_h5path2"], in_memory=True
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

        if self.distance_threshold != 0.5:
            cache_path = cache_path.split(".")[0]
            cache_path = cache_path + f"_threshold_{self.distance_threshold}" + ".pkl"

        if self.heads_type != "none":
            cache_path = cache_path.split(".")[0]
            cache_path = cache_path + f"_heads_new" + ".pkl"

        if self.randomize > 0:
            cache_path = cache_path.split(".")[0]
            cache_path = cache_path + f"_randomize_{self.randomize}" + ".pkl"

        logger.info(f"Cache Name:  {cache_path}")

        if not os.path.exists(cache_path) or self.debug:
            # Initialize Processors

            if "processors" not in registry:
                self.processors = Processors(self._tokenizer, vocab_type=self.vocab_type)
                registry.processors = self.processors
            else:
                self.processors = registry.processors

            self.entries, _ = _load_dataset(dataroot, split, self.debug)
            # convert questions to tokens, create masks, segment_ids
            self.process()

            if self.randomize > 0:
                self.process_random_spatials()
            else:
                self.process_spatials()

            if self.heads_type != "none":
                self.process_spatial_extras()
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

    def process_spatial_extras(self):
        logger.info(f"Processsing Quadrant Spatial Relations with {self.processing_threads} threads")
        mp_input = [entry["spatial_adj_matrix"] for entry in self.entries]
        sp_pool = mp.Pool(self.processing_threads)
        # map is synchronous (ordered)
        result_list = sp_pool.map(TextVQADataset.process_four_quadrant_spatials, mp_input)
        sp_pool.close()
        sp_pool.join()
        logger.info(f"Done Processsing Quadrant Spatial Relations with {self.processing_threads} threads")
        assert len(result_list) == len(mp_input)
        for entry, (quad4, share3_1, share3_2) in zip(self.entries, result_list):
            entry["spatial_adj_matrix_quad4"] = quad4
            entry["spatial_adj_matrix_share3_1"] = share3_1
            entry["spatial_adj_matrix_share3_2"] = share3_2

    def process(self):
        # Fill the boxes from readers
        for entry in tqdm(self.entries, desc="Processing Entries"):
            # tensorize
            entry["question_id"] = torch.tensor(entry["question_id"])
            entry["image_height"] = torch.tensor(entry["image_height"])
            entry["image_width"] = torch.tensor(entry["image_width"])

            # process question
            processed_question = self.processors.bert_processor({"question": entry["question"]})
            entry["question_indices"] = processed_question['token_inds']
            entry["num_question_tokens"] = processed_question['token_num']
            entry["question_mask"] = processed_question["tokens_mask"]

            # process ocr-tokens
            cleaned_ocr_tokens = [Processors.word_cleaner(word) for word in entry["google_ocr_tokens_filtered"]]

            # fasttext features
            ft_processed_tokens = self.processors.fasttext_processor({"tokens": cleaned_ocr_tokens})
            entry["ocr_fasttext"] = ft_processed_tokens["padded_token_indices"]
            entry["ocr_tokens"] = ft_processed_tokens["padded_tokens"]
            entry["ocr_length"] = ft_processed_tokens["length"]
            entry["cleaned_ocr_tokens"] = cleaned_ocr_tokens

            # phoc features
            phoc_processed_tokens = self.processors.phoc_processor({"tokens": cleaned_ocr_tokens})
            entry["ocr_phoc"] = phoc_processed_tokens["padded_phoc_features"]

            # biggest keys are: ocr_phoc, ocr_fasttext and targets (that goes into caching)
            remove_keys = ["sampled_idx_seq", "google_ocr_info_filtered", "google_ocr_tokens_filtered"]
            for key in remove_keys:
                entry.pop(key, None)

    def process_spatials(self):
        pad_obj_ocr_bboxes_list = []
        for entry in tqdm(self.entries, desc="Reading Entries"):
            # Adding spatial graph matrix
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
            pad_obj_ocr_bboxes_list.append(np.concatenate([pad_obj_bboxes[:, :-1], pad_ocr_bboxes[:, :-1]], axis=0))

        logger.info(f"Processsing Spatial Relations with {self.processing_threads} threads")
        pool = mp.Pool(self.processing_threads)
        # map is synchronous (ordered)
        results = pool.map(SpatialProcessor, pad_obj_ocr_bboxes_list)
        pool.close()
        pool.join()
        assert len(results) == len(self.entries)
        for result, entry in zip(results, self.entries):
            entry["spatial_adj_matrix"] = result

    def process_random_spatials(self):
        pad_obj_ocr_bboxes_list = []
        for entry in tqdm(self.entries, desc="Reading Entries"):
            # Adding spatial graph matrix
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
            pad_obj_ocr_bboxes_list.append(np.concatenate([pad_obj_bboxes[:, :-1], pad_ocr_bboxes[:, :-1]], axis=0))

        logger.info(f"Processsing Random Spatial Relations with {self.processing_threads} threads")
        pool = mp.Pool(self.processing_threads)
        # map is synchronous (ordered)
        results = pool.map(RandomSpatialProcessor, pad_obj_ocr_bboxes_list)
        pool.close()
        pool.join()
        assert len(results) == len(self.entries)
        for result, entry in zip(results, self.entries):
            entry["spatial_adj_matrix"] = result

    @staticmethod
    def process_four_quadrant_spatials(spatial_adj_matrix):
        replace_dict = {}
        for quad in [4, 6, 8, 10]:
            replace_dict[quad] = quad + 1
            replace_dict[quad + 1] = quad

        spatial_adj_matrix_share3_1_replace_dict = {}
        for quad in [4, 5, 6, 7, 8, 9, 10]:
            spatial_adj_matrix_share3_1_replace_dict[quad] = quad + 1
        spatial_adj_matrix_share3_1_replace_dict[11] = 4

        spatial_adj_matrix_share3_2_replace_dict = {}
        for quad in [5, 6, 7, 8, 9, 10, 11]:
            spatial_adj_matrix_share3_2_replace_dict[quad] = quad - 1
        spatial_adj_matrix_share3_2_replace_dict[4] = 11

        spatial_adj_matrix_quad4 = np.copy(spatial_adj_matrix) * 0
        spatial_adj_matrix_share3_1 = np.copy(spatial_adj_matrix) * 0
        spatial_adj_matrix_share3_2 = np.copy(spatial_adj_matrix) * 0

        assert len(spatial_adj_matrix.shape) == 2
        rows, cols = spatial_adj_matrix.shape
        for row in range(rows):
            for col in range(cols):
                spatial_adj_matrix_quad4[row][col] = replace_dict.get(spatial_adj_matrix[row][col], 0)
                spatial_adj_matrix_share3_1[row][col] = spatial_adj_matrix_share3_1_replace_dict.get(
                    spatial_adj_matrix[row][col], 0)
                spatial_adj_matrix_share3_2[row][col] = spatial_adj_matrix_share3_2_replace_dict.get(
                    spatial_adj_matrix[row][col], 0)
        return spatial_adj_matrix_quad4, spatial_adj_matrix_share3_1, spatial_adj_matrix_share3_2

    def __len__(self):
        return len(self.entries)

    def _pad_features(self, features, bboxes, num_boxes, max_feat_num, tensorize=True):
        mix_num_boxes = min(int(num_boxes), max_feat_num)
        mask = [1] * (int(mix_num_boxes))
        while len(mask) < max_feat_num:
            mask.append(0)

        mix_boxes_pad = np.zeros((max_feat_num, 5))
        mix_boxes_pad[:mix_num_boxes] = bboxes[:mix_num_boxes]

        mix_features_pad = np.zeros((max_feat_num, 2048))
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        if not tensorize:
            return mix_features_pad, mask, mix_boxes_pad

        # tensorize
        pad_features = torch.tensor(mix_features_pad).float()
        mask_features = torch.tensor(mask).long()
        pad_bboxes = torch.tensor(mix_boxes_pad).float()

        return pad_features, mask_features, pad_bboxes

    def _debug_inputs(self):
        for entry in self.entries:
            if entry["question_id"] == 24823:
                return entry

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
            # In the first iteration expand all the spatial relation matrices
            if not isinstance(entry["spatial_adj_matrix"], torch.Tensor):
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
                    entry["spatial_loss_mask"] = torch.from_numpy((entry["spatial_adj_matrix"] != 0).astype(np.float))
                    entry["spatial_ocr_relations"] = torch.from_numpy(
                        entry["spatial_adj_matrix"][-self.max_ocr_num:, -self.max_ocr_num:].astype(np.float)
                    )
                    # label_num = 12 classifies self-relationship as label=12
                    entry["spatial_adj_matrix"] = torch_broadcast_adj_matrix(
                        torch.from_numpy(entry["spatial_adj_matrix"]),
                        label_num=12
                    )

                if self.heads_type == "quad4":
                    spatial_adj_matrix_quad4 = torch_broadcast_adj_matrix(
                        torch.from_numpy(entry["spatial_adj_matrix_quad4"]),
                        label_num=12
                    )
                    entry["spatial_adj_matrix"] = torch.max(entry["spatial_adj_matrix"], spatial_adj_matrix_quad4)

                if self.heads_type == "share3":
                    spatial_adj_matrix_share3_1 = torch_broadcast_adj_matrix(
                        torch.from_numpy(entry["spatial_adj_matrix_share3_1"]),
                        label_num=12
                    )

                    spatial_adj_matrix_share3_2 = torch_broadcast_adj_matrix(
                        torch.from_numpy(entry["spatial_adj_matrix_share3_2"]),
                        label_num=12
                    )
                    entry["spatial_adj_matrix"] = torch.max(entry["spatial_adj_matrix"], spatial_adj_matrix_share3_1)
                    entry["spatial_adj_matrix"] = torch.max(entry["spatial_adj_matrix"], spatial_adj_matrix_share3_2)

                if self.heads_type == "mix":
                    spatial_adj_matrix_quad4 = torch_broadcast_adj_matrix(
                        torch.from_numpy(entry["spatial_adj_matrix_quad4"]),
                        label_num=12
                    )
                    entry["spatial_adj_matrix_quad4"] = torch.max(entry["spatial_adj_matrix"], spatial_adj_matrix_quad4)

                    spatial_adj_matrix_share3_1 = torch_broadcast_adj_matrix(
                        torch.from_numpy(entry["spatial_adj_matrix_share3_1"]),
                        label_num=12
                    )
                    spatial_adj_matrix_share3_2 = torch_broadcast_adj_matrix(
                        torch.from_numpy(entry["spatial_adj_matrix_share3_2"]),
                        label_num=12
                    )
                    entry["spatial_adj_matrix_share3"] = torch.max(entry["spatial_adj_matrix"],
                                                                   spatial_adj_matrix_share3_1)
                    entry["spatial_adj_matrix_share3"] = torch.max(entry["spatial_adj_matrix"],
                                                                   spatial_adj_matrix_share3_2)
            else:
                try:
                    assert len(entry["spatial_adj_matrix"].shape) == 3
                    assert "spatial_loss_mask" in entry
                    assert "spatial_ocr_relations" in entry
                except:
                    import pdb
                    pdb.set_trace()
        item.update(entry)

        # remove unwanted keys
        unwanted_keys = ['spatial_adj_matrix_share3_1',
                         'spatial_adj_matrix_share3_2',
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


class Processors:
    """
    Contains static-processors used for processing question/ocr-tokens, image/ocr features,
        decoding answer.
    """

    def __init__(self, bert_tokenizer, vocab_type="4k", only_registry=False):
        logger.info("Loading Processors")
        logger.info(f"Vocab Type: {vocab_type}")
        # decode-answers
        answer_config = edict()
        answer_config.max_copy_steps = 12
        answer_config.num_answers = 10
        answer_config.max_ocr_tokens = 50
        answer_config.vocab_type = vocab_type
        self.answer_processor = M4CAnswerProcessor(answer_config)
        self.only_registry = only_registry

        # Attach bert-tokenizer
        registry["bert_tokenizer"] = bert_tokenizer

        if only_registry:
            logger.info("Only registry processor initialized")
            return

        # question
        question_config = edict()
        question_config.max_length = 20
        self.bert_processor = BertTokenizerProcessor(question_config, bert_tokenizer)

        # ocr-tokens
        ocr_config = edict()
        ocr_config.max_length = 50
        self.fasttext_processor = FastTextProcessor(ocr_config)
        self.phoc_processor = PhocProcessor(ocr_config)

    @staticmethod
    def word_cleaner(word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    @staticmethod
    def word_cleaner_lower(word):
        word = word.lower()
        return word.strip()


class ImageDatabase(torch.utils.data.Dataset):
    """
    Dataset for IMDB used in Pythia
    General format that we have standardize follows:
    {
        metadata: {
            'version': x
        },
        data: [
            {
                'id': DATASET_SET_ID,
                'set_folder': <directory>,
                'feature_path': <file_path>,
                'info': {
                    // Extra information
                    'questions_tokens': [],
                    'answer_tokens': []
                }
            }
        ]
    }
    """

    def __init__(self, imdb_path):
        super().__init__()
        self.metadata = {}
        self._load_imdb(imdb_path)

    def _load_imdb(self, imdb_path):
        if imdb_path.endswith(".npy"):
            self._load_npy(imdb_path)
        else:
            raise ValueError("Unknown file format for imdb")

    def _load_npy(self, imdb_path):
        self.db = np.load(imdb_path, allow_pickle=True)
        assert isinstance(self.db, np.ndarray)
        assert "image_id" not in self.db
        self.metadata = {"version": 1}
        self.data = self.db
        self.start_idx = 1
        self.metadata.update(self.db[0])
        self._sort()

    def __len__(self):
        return len(self.data) - self.start_idx

    def __getitem__(self, idx):
        data = self.data[idx + self.start_idx]
        return data

    def get_version(self):
        return self.metadata.get("version", None)

    def _sort(self):
        sorted_data = sorted(self.data[self.start_idx:], key=lambda x: x["question_id"])
        self.data[self.start_idx:] = sorted_data
