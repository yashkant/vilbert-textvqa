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

from tools.registry import registry
from ._image_features_reader import ImageFeaturesH5Reader, CacheH5Reader
from ..spatial_utils_regat import build_graph_using_normalized_boxes, random_spatial_processor, \
    torch_broadcast_adj_matrix

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
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'

    (YK): We load questions and answers corresponding to
        the splits, and return entries.
    """

    if name == "train" or name == "val":
        question_path = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % name
        )
        questions = sorted(
            json.load(open(question_path))["questions"], key=lambda x: x["question_id"]
        )
        answer_path = os.path.join(dataroot, "cache", "%s_target.pkl" % name)
        answers = cPickle.load(open(answer_path, "rb"))
        answers = sorted(answers, key=lambda x: x["question_id"])

    elif name == "trainval":
        # LOAD TRAIN questions and answers
        question_path_train = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "train"
        )
        questions_train = sorted(
            json.load(open(question_path_train))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_train = os.path.join(dataroot, "cache", "%s_target.pkl" % "train")
        answers_train = cPickle.load(open(answer_path_train, "rb"))
        answers_train = sorted(answers_train, key=lambda x: x["question_id"])

        # LOAD VAL questions and answers
        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])
        # Todo: What is this?
        questions = questions_train + questions_val[:-3000]
        answers = answers_train + answers_val[:-3000]

    elif name == "minval":
        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])
        questions = questions_val[-3000:]
        answers = answers_val[-3000:]
        spatial_words = ["north",
                         "south",
                         "east",
                         "west",
                         "up",
                         "down",
                         "right",
                         "left",
                         "bottom",
                         "top",
                         "under",
                         "over",
                         "below",
                         "above",
                         "beside",
                         "beneath"]
        registry.q_ids = []
        for que in questions:
            for sw in spatial_words:
                if sw in que["question"]:
                    registry.q_ids.append(que["question_id"])
                    break

    elif name == "test":
        question_path_test = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2015_questions.json" % "test"
        )
        questions_test = sorted(
            json.load(open(question_path_test))["questions"],
            key=lambda x: x["question_id"],
        )
        questions = questions_test

    elif name == "mteval":
        question_path_train = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "train"
        )
        questions_train = sorted(
            json.load(open(question_path_train))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_train = os.path.join(dataroot, "cache", "%s_target.pkl" % "train")
        answers_train = cPickle.load(open(answer_path_train, "rb"))
        answers_train = sorted(answers_train, key=lambda x: x["question_id"])

        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])

        questions = questions_train
        answers = answers_train
    else:
        assert False, "data split is not recognized."

    if "test" in name:
        entries = []
        for question in questions:
            entries.append(question)
    elif name == "mteval":
        entries = []
        remove_ids = np.load(os.path.join(dataroot, "cache", "coco_test_ids.npy"))
        remove_ids = [int(x) for x in remove_ids]

        for question, answer in zip(questions, answers):
            if int(question["image_id"]) in remove_ids:
                entries.append(_create_entry(question, answer))
    else:
        assert_eq(len(questions), len(answers))
        entries = []
        remove_ids = []
        # Todo: What is this?
        # Removing ids that are present in test-set of other tasks
        if clean_datasets:
            remove_ids = np.load(os.path.join(dataroot, "cache", "coco_test_ids.npy"))
            remove_ids = [int(x) for x in remove_ids]
        for question, answer in zip(questions, answers):
            if "train" in name and int(question["image_id"]) in remove_ids:
                continue
            assert_eq(question["question_id"], answer["question_id"])
            assert_eq(question["image_id"], answer["image_id"])
            entries.append(_create_entry(question, answer))

    return entries


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
            extra_args=None,
            processing_threads=64
    ):
        """
        (YK): Builds self.entries by reading questions and answers and caches them.
        """
        super().__init__()
        self.split = split
        # Todo: What are these?
        ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.debug = extra_args.get("debug", False)
        self.heads_type = extra_args.get("heads_type", "none")
        registry.debug = self.debug
        self.randomize = extra_args.get("randomize", -1)
        self.processing_threads = processing_threads
        self.spatial_dict = {}

        logger.info(f"heads_type: {self.heads_type}")
        logger.info(f"Randomize is {self.randomize}")

        clean_train = "_cleaned" if clean_datasets else ""

        self.spatial_reader = CacheH5Reader(
            features_path=f"/srv/share/ykant3/vilbert-mt/cache/VQA_{self.split}_23_heads_new.lmdb",
            in_memory=False
        )
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

        self.use_spatial = False
        if self.heads_type != "none" or self.randomize > 0:
            self.use_spatial = True
            # cache_path = cache_path.split(".")[0]
            # cache_path = cache_path + "_heads_new.pkl"

        if self.debug:
            cache_path = "/nethome/ykant3/vilbert-multi-task/datasets/VQA/cache/VQA_trainval_23_debug.pkl"
            logger.info("Loading in debug mode from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))
            if self.heads_type != "none" or self.randomize > 0:
                self.process_spatials()
            return


        if True or not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split, clean_datasets)
            # convert questions to tokens, create masks, segment_ids
            self.tokenize(max_seq_length)
            if self.heads_type != "none" or self.randomize > 0:
                self.process_spatials()
            # convert all tokens to tensors
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

    def process_spatials(self):
        logger.info(f"Processsing Share/Single/Random Spatial Relations with {self.processing_threads} threads")
        import multiprocessing as mp

        pad_obj_list = []
        read_list = []

        for entry in tqdm(self.entries, desc="Reading Entries"):

            if entry["image_id"] in read_list:
                continue

            # Adding spatial graph matrix
            obj_features, obj_num_boxes, obj_bboxes, _ = self._image_features_reader[entry["image_id"]]
            obj_features, obj_num_boxes, obj_bboxes = obj_features[1:], obj_num_boxes - 1, obj_bboxes[1:]
            _, _, pad_obj_bboxes = self._pad_features(
                obj_features, obj_bboxes, obj_num_boxes, self._max_region_num, tensorize=False
            )
            pad_obj_bboxes = pad_obj_bboxes[:, :-1]

            # Append bboxes to the list
            pad_obj_list.append(pad_obj_bboxes)
            read_list.append(entry["image_id"])

        sp_pool = mp.Pool(self.processing_threads)
        # map is synchronous (ordered)
        result_list = list(tqdm(sp_pool.imap(VQAClassificationDataset.process_all_spatials, pad_obj_list),
                                total=len(pad_obj_list), desc="Spatial Relations"))
        sp_pool.close()
        sp_pool.join()
        logger.info(f"Done Processsing Quadrant Spatial Relations with {self.processing_threads} threads")
        assert len(result_list) == len(pad_obj_list)

        results_dict = {}

        for key, value in zip(read_list, result_list):
            results_dict[key] = value

        for entry in (self.entries):
            (adj_matrix, adj_matrix_share3_1, adj_matrix_share3_2, adj_matrix_random1, adj_matrix_random3) = \
                results_dict[entry["image_id"]]
            entry["spatial_adj_matrix"] = adj_matrix
            entry["spatial_adj_matrix_share3_1"] = adj_matrix_share3_1
            entry["spatial_adj_matrix_share3_2"] = adj_matrix_share3_2
            entry["spatial_adj_matrix_random1"] = adj_matrix_random1
            entry["spatial_adj_matrix_random3"] = adj_matrix_random3

    @staticmethod
    def process_all_spatials(pad_obj_bboxes):
        adj_matrix, adj_matrix_share3_1, adj_matrix_share3_2 \
            = build_graph_using_normalized_boxes(pad_obj_bboxes, distance_threshold=0.5)
        adj_matrix_random1, adj_matrix_random3 = random_spatial_processor(pad_obj_bboxes)
        return adj_matrix, adj_matrix_share3_1, adj_matrix_share3_2, adj_matrix_random1, adj_matrix_random3

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

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        (YK): This will add (q_token, q_input_mask, q_segment_ids) in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in tqdm(self.entries, desc="Tokenizing..."):
            tokens = self._tokenizer.encode(entry["question"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self):
        """
        Tensorizes `q_token`, `q_input_mask` and `q_segment_ids` and answer `labels` and `scores`.

        entry: {
            question_id,
            image_id,
            question,
            answer: {
                labels: 2542
                scores: 0.9
            }
            q_token,
            q_segment_ids,
            q_input_mask
        }

        """

        for entry in tqdm(self.entries, desc="Tensorizing..."):
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            if "test" not in self.split:
                answer = entry["answer"]
                labels = np.array(answer["labels"])
                scores = np.array(answer["scores"], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None

    def __getitem__(self, index):
        """
        1. Get image-features/bboxes and image mask (as nump-arrays), tensorize them.
        2. Get question, input_mask, segment_ids and coattention mask
        3. Build target (vocab-dim) with VQA scores scattered at label-indices
        4. Return
        """
        # import time
        # time_start = time.time()

        entry = self.entries[index]

        if self.use_spatial:
            spatials = self.spatial_reader[entry["image_id"]]
            transfer_keys = [
                "spatial_adj_matrix",
                "spatial_adj_matrix_share3_1",
                "spatial_adj_matrix_share3_2",
                "spatial_adj_matrix_random1",
                "spatial_adj_matrix_random3"
            ]

            for key in transfer_keys:
                entry[key] = spatials[key]

        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]

        target = torch.zeros(self.num_labels)

        item_dict = {}

        # In the first iteration expand all the spatial relation matrices
        if "spatial_adj_matrix" in entry:
            if not isinstance(entry["spatial_adj_matrix"], torch.Tensor):
                if self.randomize > 0:
                    if self.randomize == 1:
                        entry["spatial_adj_matrix"] = entry["spatial_adj_matrix_random1"]
                    elif self.randomize == 3:
                        entry["spatial_adj_matrix"] = entry["spatial_adj_matrix_random3"]
                    else:
                        raise AssertionError
                    entry["spatial_adj_matrix"] = None
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

                else:
                    # label_num = 12 classifies self-relationship as label=12
                    entry["spatial_adj_matrix"] = torch_broadcast_adj_matrix(
                        torch.from_numpy(entry["spatial_adj_matrix"]),
                        label_num=12
                    )

                if self.heads_type == "mix":
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

            move_keys = ["spatial_adj_matrix_share3", "spatial_adj_matrix", "spatial_adj_matrix_random1",
                         "spatial_adj_matrix_random3"]

            if image_id not in self.spatial_dict:
                self.spatial_dict[image_id] = {}

            for key in move_keys:
                if key in entry:
                    if key not in self.spatial_dict[image_id]:
                        self.spatial_dict[image_id][key] = entry[key]
                    # else:
                    #     try:
                    #         assert (self.spatial_dict[image_id][key] == entry[key]).all()
                    #     except:
                    #         import pdb
                    #         pdb.set_trace()
                    del entry[key]

        if self.heads_type == "mix":
            assert image_id in self.spatial_dict
            item_dict.update(self.spatial_dict[image_id])

        if "test" not in self.split:
            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            if labels is not None:
                target.scatter_(0, labels, scores)

        item_dict.update({
            "input_imgs": features,
            "image_mask": image_mask,
            "image_loc": spatials,
            "question_indices": question,
            "question_mask": input_mask,
            "image_id": image_id,
            "question_id": question_id,
            "target": target,
        })

        # time_end = time.time()

        # if time_end - time_start > 2:
        #     print(f"Time taken: {time_end - time_start}")
        #     import pdb
        #     pdb.set_trace()
        return item_dict

    def __len__(self):
        return len(self.entries)
