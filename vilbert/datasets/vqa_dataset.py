# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import logging
import time
from copy import deepcopy
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tools.registry import registry
from ._image_features_reader import ImageFeaturesH5Reader
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

    retain_keys = ["rephrasing_ids",
                   "top_k_questions",
                   "top_k_questions_neg",
                   "same_image_questions",
                   "same_image_questions_neg"
                   ]

    for key in retain_keys:
        if key in question:
            entry[key] = question[key]

    return entry


def filter_aug(questions_list, answers_list, split=None):
    questions, answers = [], []
    max_samples = registry.aug_filter["max_re_per_sample"]
    sim_threshold = registry.aug_filter["sim_threshold"]
    sampling = registry.aug_filter["sampling"]

    rephrasings_data = []
    assert len(questions_list) == len(answers_list)

    if not registry.use_rephrasings:
        if max_samples != 1:
            max_samples = 1
            registry.aug_filter["max_re_per_sample"] = 1
            print(f"Use rephrasings is False, setting max-samples to : {max_samples}")
        else:
            print(f"Use rephrasings is False w/ max-samples : {max_samples}")


    for idx, (que_list, ans_list) in tqdm(enumerate(zip(questions_list, answers_list)), total=len(questions_list),
                                          desc="Filtering Data"):

        assert len(que_list) == len(ans_list)
        if split is not None and "cc" in split and registry.allowed_only:
            que_list, ans_list = zip(*[(q,a) for q,a in zip(que_list, ans_list) if q["allowed"]])

        # filter for sim-threshold
        if sim_threshold > 0:
            que_list, ans_list = zip(*[(q,a) for q,a in zip(que_list, ans_list) if q["sim_score"] > sim_threshold])
        # filter for max-samples
        if max_samples > 0:
            if sampling == "top":
                que_list, ans_list = que_list[:max_samples], ans_list[:max_samples]
            elif sampling == "bottom":
                que_list, ans_list = que_list[-max_samples:], ans_list[-max_samples:]
            elif sampling == "random":
                # use only original question
                if len(que_list) == 1:
                    que_list, ans_list = que_list[0:1], ans_list[0:1]
                else:
                    rand_indices = np.random.choice(range(1, len(que_list)), min(max_samples - 1, len(que_list) - 1), replace=False)
                    # add original question
                    rand_indices = [0] + sorted(rand_indices)
                    que_list, ans_list = np.array(que_list), np.array(ans_list)
                    que_list, ans_list = que_list[rand_indices], ans_list[rand_indices]

            else:
                raise ValueError

        filtered_rephrasing_ids = [que["question_id"] for que in que_list]
        min_qid = min(filtered_rephrasing_ids)
        for que in que_list:
            que["rephrasing_ids"] = sorted([x for x in filtered_rephrasing_ids if x != que["question_id"]])
            if "rephrasing_of" not in que:
                que["rephrasing_of"] = min_qid
            else:
                assert min_qid == que["rephrasing_of"]

        # add them to main list
        questions.extend(que_list)
        answers.extend(ans_list)
        rephrasings_data.append(len(que_list))

    return questions, answers


def rephrasings_dict(split, questions):
    question_rephrase_dict = {}

    for question in questions:
        if "rephrasing_of" in question:
            question_rephrase_dict[question["question_id"]] = question["rephrasing_of"]
        elif "rephrasing_ids" in question:
            min_qid = min(question["rephrasing_ids"] + [question["question_id"]])
            question_rephrase_dict[question["question_id"]] = min_qid
        else:
            question_rephrase_dict[question["question_id"]] = question["question_id"]


    # used in evaluation, hack to set attribute
    from easydict import EasyDict
    super(EasyDict, registry).__setattr__(f"question_rephrase_dict_{split}", question_rephrase_dict)
    super(EasyDict, registry).__setitem__(f"question_rephrase_dict_{split}", question_rephrase_dict)
    print(f"Built dictionary: question_rephrase_dict_{split}")


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

    elif name == "re_overlap":
        question_path = "data/re-vqa/data/revqa_minval_intersect.json"
        questions = sorted(json.load(open(question_path))["questions"], key=lambda x: x["question_id"])

        question_rephrase_dict = {}
        for question in questions:
            if "rephrasing_of" in question:
                question_rephrase_dict[question["question_id"]] = question["rephrasing_of"]
            else:
                question_rephrase_dict[question["question_id"]] = question["question_id"]

        # used in evaluation, hack to set attribute
        from easydict import EasyDict
        super(EasyDict, registry).__setattr__("question_rephrase_dict", question_rephrase_dict)
        super(EasyDict, registry).__setitem__("question_rephrase_dict", question_rephrase_dict)

        answer_path_val = "datasets/VQA/cache/revqa_minval_intersect_target.pkl"
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers = sorted(answers_val, key=lambda x: x["question_id"])

        # d_q, d_a = [], []
        # assert len(questions) == len(answers)
        # for question, answer in zip(questions, answers):
        #     assert answer["question_id"] == question["question_id"]
        #     if "rephrasing_of" not in question:
        #         d_q.append(question)
        #         d_a.append(answer)
        #
        # questions, answers = d_q, d_a

    elif name in ["re_train", "re_val", "re_train_negs", "re_val_negs"]:

        split_path_dict = {
            "re_train": ["data/re-vqa/data/revqa_train_proc.json", "datasets/VQA/cache/revqa_train_target.pkl", "train"],
            "re_val": ["data/re-vqa/data/revqa_val_proc.json", "datasets/VQA/cache/revqa_val_target.pkl", "val"],
            "re_train_negs": ["data/re-vqa/data/revqa_train_proc_image_negs.json", "datasets/VQA/cache/revqa_train_target_image_negs.pkl", "train"],
            "re_val_negs": ["data/re-vqa/data/revqa_val_proc_image_negs.json", "datasets/VQA/cache/revqa_val_target_image_negs.pkl", "val"],
            "val_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
                        "datasets/VQA/back-translate/org2_bt_val_target.pkl", "val"],
        }

        questions_path = split_path_dict[name][0]
        answers_path = split_path_dict[name][1]
        split = split_path_dict[name][-1]

        questions = sorted(json.load(open(questions_path))["questions"], key=lambda x: x["question_id"])

        if registry.debug:
            questions = questions[:1000]

        rephrasings_dict(split, questions)

        answers = cPickle.load(open(answers_path, "rb"))
        answers = sorted(answers, key=lambda x: x["question_id"])

        if registry.debug:
            answers = answers[:1000]

        # replace human rephrasings questions w/ BT rephrasings
        if name == "re_train" and registry.use_bt_re:
            val_questions_path, val_answers_path, val_split = split_path_dict["val_aug"][0], split_path_dict["val_aug"][
                1], split_path_dict["val_aug"][-1]
            val_questions_list = cPickle.load(open(val_questions_path, "rb"))
            val_answers_list = cPickle.load(open(val_answers_path, "rb"))
            val_questions, val_answers = filter_aug(val_questions_list, val_answers_list)

            original_ids = (set(registry.question_rephrase_dict_train.values()))
            questions, answers = [], []

            for q,a in zip(val_questions, val_answers):
                if q["rephrasing_of"] in original_ids:
                    questions.append(q)
                    answers.append(a)
            rephrasings_dict(split, questions)

        if name == "re_train" and registry.use_no_re:
            fil_questions, fil_answers = [], []
            for q,a in zip(questions, answers):
                ref_qid = min([q["question_id"]] + q["rephrasing_ids"])
                if ref_qid == q["question_id"]:
                    fil_questions.append(q)
                    fil_answers.append(a)
            questions, answers = fil_questions, fil_answers
            rephrasings_dict(split, questions)


        assert len(questions) == len(answers)
        for question, answer in zip(questions, answers):
            try:
                assert answer["question_id"] == question["question_id"]
            except:
                import pdb
                pdb.set_trace()

    elif name in ["train_aug", "val_aug", "trainval_aug", "train_aug_cc", "train_aug_cc_v2" ,"minval_aug", "train_aug_fil"]:

        split_path_dict = {
            "train_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_train2014_questions.pkl",
                         "datasets/VQA/back-translate/org2_bt_train_target.pkl", "train"],
            "train_aug_cc": ["datasets/VQA/cc-re/cc_v2_OpenEnded_mscoco_train2014_questions_88.pkl",
                         "datasets/VQA/cc-re/cc_train_target_88.pkl", "train"],
            "train_aug_cc_v2": ["datasets/VQA/cc-re/cc_v2_OpenEnded_mscoco_train2014_questions_88_v2.pkl",
                         "datasets/VQA/cc-re/cc_train_target_88_v2.pkl", "train"],
            "train_aug_fil": ["datasets/VQA/back-translate/bt_fil_dcp_sampling_{}_v2_OpenEnded_mscoco_train2014_questions.pkl",
                         "datasets/VQA/back-translate/bt_fil_dcp_sampling_{}_train_target.pkl", "train"],
            "val_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
                         "datasets/VQA/back-translate/org2_bt_val_target.pkl", "val"],
            "trainval_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_trainval2014_questions.pkl",
                         "datasets/VQA/back-translate/org2_bt_trainval_target.pkl", "trainval"],
            "minval_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_minval2014_questions.pkl",
                         "datasets/VQA/back-translate/org2_bt_minval_target.pkl", "minval"],
        }
        questions_path, answers_path, split = split_path_dict[name][0], split_path_dict[name][1], split_path_dict[name][-1]

        if name == "train_aug_fil":
            questions_path = questions_path.format(registry.aug_filter["sampling"])
            answers_path = answers_path.format(registry.aug_filter["sampling"])


        if not registry.debug:
            questions_list = cPickle.load(open(questions_path, "rb"))
            
            if isinstance(questions_list, dict):
                questions_list = questions_list["questions"]

            answers_list = cPickle.load(open(answers_path, "rb"))
        else:
            questions_path = questions_path.split(".")[0] + "_debug.pkl"
            answers_path = answers_path.split(".")[0] + "_debug.pkl"
            questions_list = cPickle.load(open(questions_path, "rb"))
            answers_list = cPickle.load(open(answers_path, "rb"))

        # filter-mech
        questions, answers = filter_aug(questions_list, answers_list, name)
        assert len(questions) == len(answers)

        logger.info(f"Train Samples after filtering: {len(questions)}")
        # this is needed for evaluation
        rephrasings_dict(split, questions)

        for question, answer in zip(questions, answers):
            assert answer["question_id"] == question["question_id"]

    elif name in ["re_total", "re_total_bt", "re_total_cc"]:
        paths_dict = {
            "re_train": ["data/re-vqa/data/revqa_train_proc.json", "datasets/VQA/cache/revqa_train_target.pkl", "train"],
            "re_val": ["data/re-vqa/data/revqa_val_proc.json", "datasets/VQA/cache/revqa_val_target.pkl", "val"],
            "val_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
                        "datasets/VQA/back-translate/org2_bt_val_target.pkl", "val"],
            "val_cc": ["datasets/VQA/cc-re/cc_v2_OpenEnded_mscoco_val2014_questions_88.pkl",
                        "datasets/VQA/cc-re/cc_val_target_88.pkl", "val"],
            "val_cc_v2": ["datasets/VQA/cc-re/cc_v2_OpenEnded_mscoco_val2014_questions_88_v2.pkl",
                        "datasets/VQA/cc-re/cc_val_target_88_v2.pkl", "val"],
            "val_aug_fil": [
                "datasets/VQA/back-translate/bt_fil_dcp_sampling_{}_v2_OpenEnded_mscoco_train2014_questions.pkl"
                    .format(registry.aug_filter["sampling"]),
                "datasets/VQA/back-translate/bt_fil_dcp_sampling_{}_train_target.pkl"
                    .format(registry.aug_filter["sampling"]),
                "val"
            ],
        }

        questions, answers = [], []
        for key, value in paths_dict.items():
            if key not in ["re_train", "re_val"]:
                continue
            _questions, _answers = json.load(open(value[0]))["questions"], cPickle.load(open(value[1], "rb"))
            questions.extend(_questions)
            answers.extend(_answers)
        questions = sorted(questions, key=lambda x: x["question_id"])
        answers = sorted(answers, key=lambda x: x["question_id"])
        rephrasings_dict(name, questions)

        assert len(questions) == len(answers)
        for question, answer in zip(questions, answers):
            assert answer["question_id"] == question["question_id"]

        # replace human rephrasings questions w/ BT rephrasings
        if registry.use_bt_re or name in ["re_total_bt", "re_total_cc", "re_total_cc_v2"]:
            bt_eval_key = registry.bt_eval_key
            val_questions_path, val_answers_path, val_split = paths_dict[bt_eval_key][0], paths_dict[bt_eval_key][
                1], paths_dict[bt_eval_key][-1]
            logger.info(f"Using \n question path: {val_questions_path} \n answers path: {val_answers_path}")
            val_questions_list = cPickle.load(open(val_questions_path, "rb"))
            val_answers_list = cPickle.load(open(val_answers_path, "rb"))
            val_questions, val_answers = filter_aug(val_questions_list, val_answers_list, bt_eval_key)

            original_ids = (set(registry.question_rephrase_dict_re_total.values()))

            if name == "re_total_bt":
                dump_path = "/nethome/ykant3/vilbert-multi-task/data/re-vqa/data/non_overlap_ids.npy"
                qids = np.load(dump_path, allow_pickle=True)
                assert len(set(qids).intersection(set(original_ids))) == 0
                original_ids = qids

            questions, answers = [], []
            for q,a in zip(val_questions, val_answers):
                if q["rephrasing_of"] in original_ids:
                    questions.append(q)
                    answers.append(a)
            rephrasings_dict(name, questions)

    elif name == "trainval":
        # (YK): We use train + (val - minval) questions
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
        questions = questions_train + questions_val[:-3000]
        answers = answers_train + answers_val[:-3000]

    elif name == "minval":
        # (YK): We use the last 3000 samples of the validation set
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
        assert False, f"data split {name} is not recognized."

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
        # Removing ids that are present in test-set of other tasks
        if clean_datasets:
            remove_ids = np.load(os.path.join(dataroot, "cache", "coco_test_ids.npy"))
            remove_ids = [int(x) for x in remove_ids]
        for question, answer in tqdm(zip(questions, answers), total=len(questions), desc="Building Entries"):
            if "train" in name and int(question["image_id"]) in remove_ids:
                continue
            try:
                assert_eq(question["question_id"], answer["question_id"])
                assert_eq(question["image_id"], answer["image_id"])
            except:
                import pdb
                pdb.set_trace()
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
        ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        # attach to registry
        registry.ans2label = self.ans2label
        registry.label2ans = self.label2ans
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
        self.extra_args = extra_args
        self.mask_image = registry.mask_image

        logger.info(f"heads_type: {self.heads_type}")
        logger.info(f"Randomize is {self.randomize}")

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

        if self.heads_type != "none" or self.randomize > 0:
            cache_path = cache_path.split(".")[0]
            cache_path = cache_path + "_heads_new.pkl"

        # if self.debug:
        #     cache_path = "/nethome/ykant3/vilbert-multi-task/datasets/VQA/cache/VQA_trainval_23_debug.pkl"
        #     logger.info("Loading in debug mode from %s" % cache_path)
        #     self.entries = cPickle.load(open(cache_path, "rb"))
        #     if self.heads_type != "none" or self.randomize > 0:
        #         self.process_spatials()
        #     return

        if (not os.path.exists(cache_path) or extra_args.get("revqa_eval", False) or extra_args.get("contrastive", None) \
                in ["simclr", "better"]) or True:
            self.entries = _load_dataset(dataroot, split, clean_datasets)

            # convert questions to tokens, create masks, segment_ids
            self.tokenize(max_seq_length)
            if self.heads_type != "none" or self.randomize > 0:
                self.process_spatials()
            # convert all tokens to tensors
            if not registry.sdebug:
                self.tensorize()
            # cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

        self.mean_read_time = 0.0
        self.num_samples = 0

    #     if extra_args.get("revqa_eval", False) or extra_args.get("contrastive", None) in ["simclr", "better"]:
    #         self.build_map()
    #
    # def build_map(self):
    #     self.entry_map = {}
    #     for entry in self.entries:
    #         question_ids = list(set(deepcopy(entry["rephrasing_ids"]) + [entry["question_id"]]))
    #         question_ids.sort()
    #         self.entry_map[min(question_ids)] = {
    #             "ids": question_ids,
    #             "iter": cycle(question_ids)
    #         }
    #     self.map_items = sorted(self.entry_map.items(), key=lambda x: x[0])

    def process_spatials(self):
        logger.info(f"Processsing Share/Single/Random Spatial Relations with {self.processing_threads} threads")
        import multiprocessing as mp

        pad_obj_list = []

        for entry in tqdm(self.entries, desc="Reading Entries"):
            # Adding spatial graph matrix
            obj_features, obj_num_boxes, obj_bboxes, _ = self._image_features_reader[entry["image_id"]]
            obj_features, obj_num_boxes, obj_bboxes = obj_features[1:], obj_num_boxes - 1, obj_bboxes[1:]
            _, _, pad_obj_bboxes = self._pad_features(
                obj_features, obj_bboxes, obj_num_boxes, self._max_region_num, tensorize=False
            )
            pad_obj_bboxes = pad_obj_bboxes[:, :-1]

            # Append bboxes to the list
            pad_obj_list.append(pad_obj_bboxes)


        sp_pool = mp.Pool(self.processing_threads)
        # map is synchronous (ordered)
        result_list = list(tqdm(sp_pool.imap(VQAClassificationDataset.process_all_spatials, pad_obj_list),
                                     total=len(pad_obj_list), desc="Spatial Relations"))
        sp_pool.close()
        sp_pool.join()
        logger.info(f"Done Processsing Quadrant Spatial Relations with {self.processing_threads} threads")
        assert len(result_list) == len(pad_obj_list)


        for entry, (adj_matrix, adj_matrix_share3_1, adj_matrix_share3_2, adj_matrix_random1, adj_matrix_random3) \
                in zip(self.entries, result_list):
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
        self.question_map = {}
        for _idx, entry in enumerate(tqdm(self.entries, desc="Tokenizing...")):
            self.question_map[entry["question_id"]] = _idx

            if registry.sdebug:
                continue

            # these are just raw questions!
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
        import time

        if hasattr(self, "count"):
            self.count += 1
        else:
            self.count = 0

        # if self.count <= 4:
        #     import pdb
        #     pdb.set_trace()

        item_dict = {}

        start_time = time.time()

        entry = self.entries[index]

        entry_load_time = time.time()

        image_id = entry["image_id"]
        question_id = entry["question_id"]

        if registry.sdebug:
            item_dict.update({
                "idx": index,
                "image_id": image_id,
                "question_id": question_id,
            })

            if len(entry["rephrasing_ids"]) == 0:
                pos_entry = entry
            else:
                que_id = np.random.choice(entry["rephrasing_ids"])
                pos_entry = self.entries[self.question_map[que_id]]
            item_pos_dict = {}
            item_pos_dict.update({
                "idx": index,
                "question_id": pos_entry["question_id"],
                "image_id": entry["image_id"]
            })

            return (item_dict, item_pos_dict)


        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        features_load_time = time.time()
        # print(f"Feat load time: {features_load_time - entry_load_time}")

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
                    entry["spatial_adj_matrix_share3"] = torch.max(entry["spatial_adj_matrix"], spatial_adj_matrix_share3_1)
                    entry["spatial_adj_matrix_share3"] = torch.max(entry["spatial_adj_matrix"], spatial_adj_matrix_share3_2)

            move_keys = ["spatial_adj_matrix_share3", "spatial_adj_matrix", "spatial_adj_matrix_random1", "spatial_adj_matrix_random3"]

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

        # remove all the images via masking!
        if self.mask_image:
            image_mask = torch.zeros_like(image_mask)

        item_dict.update({
            "input_imgs": features,
            "image_mask": image_mask,
            "image_loc": spatials,
            "question_indices": question,
            "question_mask": input_mask,
            "image_id": image_id,
            "question_id": question_id,
            "target": target,
            "mask": torch.tensor(1)
        })

        time_end = time.time()
        total_time = (time_end-start_time)

        # print(f"Total Time: {total_time}")

        # if total_time > self.mean_read_time*3:
        #     import pdb
        #     pdb.set_trace()

        self.mean_read_time = ((self.mean_read_time*self.num_samples) + total_time)/(self.num_samples+1)
        self.num_samples += 1

        # don't use while evaluation loop
        if self.extra_args.get("contrastive", None) in ["simclr", "better"] \
                and registry.use_rephrasings \
                and self.split not in ["minval", "re_total", "re_val", "test", "val"]:
            common_indices = torch.zeros_like(entry['q_token'])
            common_indices_pos = torch.zeros_like(entry['q_token'])
            # item_dict["common_inds"] = common_indices

            return_list = [item_dict]
            item_pos_dicts = [deepcopy(item_dict) for _ in range(registry.num_rep-1)]
            # when there's no rephrasing available send the original
            if len(entry["rephrasing_ids"]) == 0:
                item_dict["mask"] = item_dict["mask"] * 0
                for id in item_pos_dicts:
                    id["mask"] = id["mask"] * 0
                return_list.extend(item_pos_dicts)
                return return_list

            que_ids = np.random.choice(entry["rephrasing_ids"], registry.num_rep-1)
            pos_entries = [self.entries[self.question_map[qid]] for qid in que_ids]

            for id, pe in zip(item_pos_dicts, pos_entries):
                id.update({
                    "question_indices": pe["q_token"],
                    "question_mask": pe["q_input_mask"],
                    "question_id": pe["question_id"],

                })

            return_list.extend(item_pos_dicts)
            return return_list





            # # when there's no rephrasing available send the original
            # if len(entry["rephrasing_ids"]) == 0:
            #     item_dict["mask"] = item_dict["mask"] * 0
            #     item_pos_dict["mask"] = item_pos_dict["mask"] * 0
            #     return (item_dict, item_pos_dict)
            #
            # que_id = np.random.choice(entry["rephrasing_ids"])
            # pos_entry = self.entries[self.question_map[que_id]]
            # try:
            #     assert pos_entry["image_id"] == entry["image_id"]
            #     if pos_entry["answer"]["labels"] is not None:
            #         assert all(pos_entry["answer"]["labels"] == entry["answer"]["labels"])
            # except:
            #     import pdb
            #     pdb.set_trace()
            #
            # # entry_tokens = set(entry['q_token'].tolist())
            # # pos_entry_tokens = set(pos_entry['q_token'].tolist())
            # # common_tokens = list(pos_entry_tokens.intersection(entry_tokens) - set([0, 101, 102]))
            # #
            # # for iter, tok in enumerate(common_tokens):
            # #     entry_idx = int(((tok == entry['q_token']) * 1).nonzero()[0][0])
            # #     pos_entry_idx = int(((tok == pos_entry['q_token']) * 1).nonzero()[0][0])
            # #     common_indices[iter] = entry_idx
            # #     common_indices_pos[iter] = pos_entry_idx
            #
            # item_pos_dict.update({
            #     "question_indices": pos_entry["q_token"],
            #     "question_mask": pos_entry["q_input_mask"],
            #     "question_id": pos_entry["question_id"],
            #     # "common_inds": common_indices_pos
            # })
            #
            # return (item_dict, item_pos_dict)

        return item_dict

    def __len__(self):
        # if self.extra_args.get("contrastive", None) in ["simclr", "better"]:
        #     return len(self.map_items)
        # else:
        return len(self.entries)
