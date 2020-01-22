# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .textvqa_processors import *
from easydict import EasyDict as edict
from ._image_features_reader import ImageFeaturesH5Reader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(question, answer):
    answer.pop("image_id")
    answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],
        "question": question["question"],
        "answer": answer,
    }
    return entry


def _load_dataset(dataroot, name, clean_datasets):
    """Load entries from Imdb

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'

    (YK): We load questions and answers corresponding to
        the splits, and return entries.
    """

    if name == "train" or name == "val" or name =="test":

        imdb_holder = "imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_{}.npy"

        if name == "test":
            imdb_holder = "imdb_google_det_bbox_textvqa_info_{}.npy"

        imdb_path = os.path.join(
            dataroot, imdb_holder.format(name)
        )
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
        "google_ocr_info_filtered",
    ]

    for instance in imdb_data:
        entry = dict([(key, instance[key]) for key in store_keys])
        entries.append(entry)

    return entries, imdb_data.metadata


class VQAClassificationDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        clean_datasets,
        padding_index=0,
        max_seq_length=16,
        max_region_num=101,
    ):
        """
        (YK): Builds self.entries by reading questions and answers and caches them.
        """
        super().__init__()
        self.split = split
        # Todo: What are these?
        # ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
        # label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
        # self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        # self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        # self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index

        clean_train = "_cleaned" if clean_datasets else ""

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
                task + "_" + split + "_" + str(max_seq_length) + clean_train + ".pkl",
            )

        # Initialize Processors
        self.processors = Processors()

        if not os.path.exists(cache_path):
            self.entries, _ = _load_dataset(dataroot, split, clean_datasets)
            # convert questions to tokens, create masks, segment_ids
            self.process()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))


    def process(self):
        for entry in self.entries:
            # process question
            processed_question = self.processors.bert_processor({"question": entry["question"]})
            entry["question_indices"] = processed_question['token_inds']
            entry["num_question_tokens"] = processed_question['token_num']

            # process ocr-tokens
            cleaned_ocr_tokens = [Processors.word_cleaner(word) for word in entry["ocr_tokens"]]
            # fasttext features
            ft_processed_tokens = self.processors.fasttext_processor({"tokens": cleaned_ocr_tokens})
            entry["ocr_fasttext"] = ft_processed_tokens["padded_token_indices"]
            entry["ocr_tokens"] = ft_processed_tokens["padded_tokens"]
            entry["ocr_length"] = ft_processed_tokens["length"]
            # phoc features
            phoc_processed_tokens = self.processors.phoc_processor({"tokens": cleaned_ocr_tokens})
            entry["ocr_phoc"] = phoc_processed_tokens["padded_token_indices"]

            # process answers
            cleaned_answers = [Processors.word_cleaner(word) for word in entry["answers"]]
            processed_answers = self.processors.answer_processor({
                "answers": cleaned_answers,
                "context_tokens": cleaned_ocr_tokens,
            })
            entry["targets"] = processed_answers["answers_scores"]
            entry["sampled_idx_seq"] = processed_answers["sampled_idx_seq"]
            entry["train_prev_inds"] = processed_answers["train_prev_inds"]
            entry["train_loss_mask"] = processed_answers["train_loss_mask"]
            entry["train_acc_mask"] = processed_answers["train_acc_mask"]

        import pdb
        pdb.set_trace()


class Processors:
    """
    Contains static-processors used for processing question/ocr-tokens, image/ocr features,
        decoding answer.
    """

    def __init__(self):

        # question
        question_config = edict()
        question_config.max_length = 20
        self.bert_processor = BertTokenizerProcessor(question_config)

        # ocr-tokens
        ocr_config = edict()
        ocr_config.max_length = 50
        self.fasttext_processor = FastTextProcessor(ocr_config)
        self.phoc_processor = PhocProcessor(ocr_config)

        # decode-answers
        answer_config = edict()
        answer_config.max_copy_steps = 12
        answer_config.num_answers = 10j
        answer_config.max_ocr_tokens = 50
        self.answer_processor = M4CAnswerProcessor(answer_config)


    @staticmethod
    def word_cleaner(word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
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
        import pdb
        pdb.set_trace()
        soreted_data = sorted(self.data[self.start_idx:], key=lambda x: x["question_id"])
        self.data[self.start_idx:] = sorted_data
