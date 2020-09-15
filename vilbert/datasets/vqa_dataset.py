import _pickle as cPickle
import json
import logging
import os
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from tools.registry import registry

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def create_entry(question, answer):
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


def filter_aug(questions_list, answers_list):
    questions, answers = [], []
    max_samples = registry.aug_filter["num_rephrasings"]
    sim_threshold = registry.aug_filter["sim_threshold"]
    assert len(questions_list) == len(answers_list)

    for idx, (que_list, ans_list) in tqdm(enumerate(zip(questions_list, answers_list)),
                                          total=len(questions_list),
                                          desc="Filtering Data"):

        assert len(que_list) == len(ans_list)
        # filter for sim-threshold
        if sim_threshold > 0:
            que_list, ans_list = zip(*[(q,a) for q,a in zip(que_list, ans_list) if q["sim_score"] > sim_threshold])

        # filter for max-samples
        if max_samples > 0:
            que_list, ans_list = que_list[:max_samples], ans_list[:max_samples]

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


def load_qa(name, sort=True, use_filter=False, set_dict=False):
    split_path_dict = {
        "train_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_train2014_questions.pkl",
                      "datasets/VQA/back-translate/org2_bt_train_target.pkl", "train"],
        "train": ["datasets/VQA/v2_OpenEnded_mscoco_train2014_questions.json",
                      "datasets/VQA/cache/train_target.pkl", "train"],
        "val": ["datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json",
                      "datasets/VQA/cache/val_target.pkl", "val"],
        "test": ["datasets/VQA/v2_OpenEnded_mscoco_test2015_questions.json",
                      "", "test"],
        "val_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
                    "datasets/VQA/back-translate/org2_bt_val_target.pkl", "val"],
        "trainval_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_trainval2014_questions.pkl",
                         "datasets/VQA/back-translate/org2_bt_trainval_target.pkl", "trainval"],
        "minval_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_minval2014_questions.pkl",
                       "datasets/VQA/back-translate/org2_bt_minval_target.pkl", "minval"],
        "re_total": ["data/re-vqa/data/revqa_total_proc.json", "datasets/VQA/cache/revqa_total_target.pkl", "re_total"],


    }
    questions_path, answers_path, split = split_path_dict[name]
    questions = json.load(open(questions_path)) if questions_path.endswith(".json") \
        else cPickle.load(open(questions_path, "rb"))
    if isinstance(questions, dict):
        questions = questions["questions"]

    if name == "test":
        return questions

    answers = cPickle.load(open(answers_path, "rb"))

    if sort:
        questions = sorted(questions, key=lambda x: x["question_id"])
        answers = sorted(answers, key=lambda x: x["question_id"])

    if use_filter:
        questions, answers = filter_aug(questions, answers)

    if set_dict:
        rephrasings_dict(split, questions)

    return questions, answers


def load_entries(dataroot, name):
    """Load questions and answers.
    """

    import pdb
    pdb.set_trace()

    if name == "train" or name == "val":
        questions, answers = load_qa(name)
        if registry.debug:
            questions, answers = questions[:40000], answers[:40000]

    elif name in ["train_aug", "val_aug", "trainval_aug", "minval_aug", "re_total"]:
        questions, answers = load_qa(name, sort=False, use_filter=True, set_dict=True)
        assert len(questions) == len(answers)
        logger.info(f"Samples after filtering: {len(questions)}")

    # replace human rephrasings questions w/ BT rephrasings
    elif name == "re_total_bt":
        val_questions, val_answers = load_qa(name, sort=False, use_filter=True)
        dump_path = "/nethome/ykant3/vilbert-multi-task/data/re-vqa/data/non_overlap_ids.npy"
        non_overlap_ids = np.load(dump_path, allow_pickle=True)
        questions, answers = [], []
        for q, a in zip(val_questions, val_answers):
            if q["rephrasing_of"] in non_overlap_ids:
                questions.append(q)
                answers.append(a)
        rephrasings_dict(name, questions)

    elif name == "trainval":
        questions_train, answers_train = load_qa("train", sort=True)
        questions_val, answers_val = load_qa("val", sort=True)
        questions = questions_train + questions_val[:-3000]
        answers = answers_train + answers_val[:-3000]

    elif name == "minval":
        questions_val, answers_val = load_qa("val", sort=True)
        questions, answers = questions_val[-3000:], answers_val[-3000:]

    elif name == "test":
        questions = load_qa(name)

    else:
        assert False, f"data split {name} is not recognized."

    if "test" in name:
        entries = []
        for question in questions:
            entries.append(question)
    else:
        assert_eq(len(questions), len(answers))
        entries = []

        for question, answer in tqdm(zip(questions, answers), total=len(questions), desc="Building Entries"):
            assert_eq(question["question_id"], answer["question_id"])
            assert_eq(question["image_id"], answer["image_id"])
            entries.append(create_entry(question, answer))

    return entries


class VQAClassificationDataset(Dataset):
    def __init__(
        self,
        dataroot,
        split,
        image_features_reader,
        tokenizer,
        padding_index=0,
        max_seq_length=16,
        max_region_num=101,
        extra_args=None,
    ):
        """
        Builds self.entries by reading questions and answers and caches them.
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
        self.debug = registry.debug
        registry.debug = self.debug
        self.extra_args = extra_args
        self.mask_image = registry.mask_image

        self.entries = load_entries(dataroot, split)
        # convert questions to tokens, create masks, segment_ids
        self.tokenize(max_seq_length)
        self.tensorize()
        self.mean_read_time = 0.0
        self.num_samples = 0


    def tokenize(self, max_length=16):
        """Tokenizes the questions."""
        self.question_map = {}
        for _idx, entry in enumerate(tqdm(self.entries, desc="Tokenizing...")):
            self.question_map[entry["question_id"]] = _idx
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
        item_dict = {}
        entry = self.entries[index]
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

        if registry.debug:
            return [item_dict]

        # don't use while evaluation loop
        if self.extra_args.get("contrastive", None) in ["better"] \
                and self.split not in ["minval", "re_total", "re_val", "test", "val"]:
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

        return item_dict

    def __len__(self):
        return len(self.entries)
