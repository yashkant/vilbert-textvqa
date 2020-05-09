import json
import os
from collections import defaultdict, Counter
from copy import deepcopy
from itertools import cycle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import logging
from tools.registry import registry

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False

    __iter__() is called after each epoch to get batch indices
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)


class NegativeSampler(Sampler):

    def __init__(self,
                 data_source,
                 batch_size,
                 task_cfg,
                 args,
                 replacement=False,
                 num_samples=None,
                 split="train"):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.split = split
        self.task_cfg = task_cfg["TASK19"]
        self.epoch_idx = int(1e10)
        self.epochs = []
        self.num_positives = self.task_cfg.get("num_positives", -1)
        if self.num_positives > 0:
            self.read_annotations()
        self.bin_ans_threshold = self.task_cfg.get("bin_ans_threshold", None)
        self.freq_ans_threshold = self.task_cfg.get("freq_ans_threshold", None)
        self.iter_count = 0

        logger.info(f"Use GT answers is: {self.task_cfg.get('use_gt_answer', False)}")

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        self.entries = data_source.entries
        # map from question-id -> entry-idx
        self.question_map = data_source.question_map

    # todo: move to dataset.py
    def read_annotations(self):
        ann_path = "/nethome/ykant3/vilbert-multi-task/datasets/VQA/v2_mscoco_val2014_annotations.json"
        ann_data = json.load(open(ann_path))["annotations"]
        self.qid_ann_dict = {}

        for ann in ann_data:
            ans_counter = Counter()
            for ans in ann["answers"]:
                # only consider answers available in vocabulary
                if ans['answer'] in registry.ans2label:
                    ans_counter[registry.ans2label[ans['answer']]] += 1

            # build dict with qid -> [(ans_label, freq)...]
            self.qid_ann_dict[ann["question_id"]] = ans_counter.most_common()

    def build_maps(self):
        """
        1. Builds Map from min(repharsed_ids) = info
        2. Builds re_bins = items of above dict sorted by ids

        """

        self.entry_map = {}

        # maps gt_ans -> [(entry_map_key, ans_freq) ...]
        self.answer_map = {}

        for entry in tqdm(self.entries, desc="Building question and answer maps", total=len(self.entries)):
            question_ids = list(set(deepcopy(entry["rephrasing_ids"]) + [entry["question_id"]]))
            question_ids.sort()
            entry_map_key = min(question_ids)
            self.entry_map[entry_map_key] = {
                "question_ids": question_ids,
                "iter_idx": 0,
                "entry_inds": [self.question_map[x] if not registry.debug else 0 for x in question_ids ]
            }
            for (ans_label, freq) in self.qid_ann_dict[entry_map_key]:
                if freq >= self.freq_ans_threshold:
                    if ans_label in self.answer_map:
                        self.answer_map[ans_label].append((entry_map_key, freq))
                    else:
                        self.answer_map[ans_label] = [(entry_map_key, freq)]

        # post-process: remove duplicates, build sampling weights
        for key in tqdm(self.answer_map.keys(), desc="Post-processing", total=len(self.answer_map)):
            self.answer_map[key] = list(set(self.answer_map[key]))
            self.answer_map[key] = list(zip(*self.answer_map[key]))
            self.answer_map[key][1] = np.array(self.answer_map[key][1])/sum(self.answer_map[key][1])

        self.re_bins = sorted(self.entry_map.items(), key=lambda x: x[0])

    def get_entry_answers(self, entry):
        entry_answers = entry["answer"]["labels"]
        if entry_answers is None:
            entry_answers = []
        else:
            entry_answers = entry_answers.tolist()
        return entry_answers

    def check_bin(self, bin, use_gt_answer, batches, batches_extension, batches_answers_set, batch_idx):
        flag = False
        for nentry_idx in bin["entry_inds"]:
            # skip negatives that are already present in batch/batch_extension
            if nentry_idx in set(batches[batch_idx]) or nentry_idx in set(batches_extension[batch_idx]):
                flag = True
                break

            # skip negatives with same ground-truth as other negatives in the batch.
            if use_gt_answer:
                entry_answers = self.get_entry_answers(self.entries[nentry_idx])
                if len(set(batches_answers_set[batch_idx]).intersection(set(entry_answers))) > 0:
                    flag = True
                    break
        return flag

    def get_negative(self,
                     batch_idx,
                     negative_list,
                     batches,
                     batches_extension,
                     use_gt_answer,
                     batches_answers_set):

        neg_entry_idx = -1
        np.random.shuffle(negative_list)

        # return if negatives list is empty
        if len(negative_list) == 0:
            return neg_entry_idx, True

        # iterate over the negatives and pick one without adding a positive sample
        for qid in negative_list:
            source_id = getattr(registry, f"question_rephrase_dict_{self.split}")[qid]
            bin = self.entry_map[source_id]
            flag = self.check_bin(bin, use_gt_answer, batches, batches_extension, batches_answers_set, batch_idx)

            # if passes add the sample
            if not flag:
                neg_entry_idx = bin["entry_inds"][bin["question_ids"].index(qid)]
                break

        return neg_entry_idx, flag

    def build_better_batches(self):
        self.build_maps()

        neg_replace = self.task_cfg["neg_replace"]
        init_batch_size = self.task_cfg["init_batch_size"]
        neg_type_weights = self.task_cfg["neg_type_weights"]
        neg_question_thresh = self.task_cfg["neg_question_thresh"]
        use_gt_answer = self.task_cfg["use_gt_answer"]
        assert np.sum(neg_type_weights) == 1.0
        assert self.batch_size % init_batch_size == 0
        num_passes = int(self.batch_size / init_batch_size) - 1

        assert neg_replace

        # we ensure that we cover all the samples occur at least once
        # num_batches that will be created will be more than no. of batches in a single epoch
        # Todo: We can reduce this by dividing only by samples from re-bins.
        num_batches = int(len(self.entries)/init_batch_size)

        # actual no. of batches to return (for one epoch)
        self.num_batches = int(len(self.entries)/self.batch_size)
        batches_answers_set = []

        # we monitor all the bins using self.entry_map and self.re_bins (both are mapped to same data)
        np.random.shuffle(self.re_bins)

        batches = []
        cycle_bins = cycle(list(range(len(self.re_bins))))
        batches_extension = [[] for _ in range(num_batches)]


        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Initial Pass: "):
            batch_inds = []
            batch_answers = []
            batch_bin_inds = []

            while True:
                idx = next(cycle_bins)
                bin = self.re_bins[idx][1]

                # do not use image-negatives
                if len(bin["entry_inds"]) == 1:
                    continue

                # we are not exhausting bins with gt_answers
                if idx in batch_bin_inds:
                    continue

                # we are not exhausting
                # if item["iter_idx"] is None:
                #     continue

                entry_idx = bin["entry_inds"][bin["iter_idx"]]

                # skip negatives with same ground-truth
                if use_gt_answer:
                    entry_answers = self.get_entry_answers(self.entries[entry_idx])

                    if len(set(batch_answers).intersection(set(entry_answers))) > 0:
                        continue

                bin["iter_idx"] += 1
                batch_inds.append(entry_idx)
                batch_answers.extend(self.get_entry_answers(self.entries[entry_idx]))
                batch_bin_inds.append(idx)

                # exit initial batches
                if len(batch_inds) == init_batch_size:
                    # shuffle left and right parts
                    np.random.shuffle(self.re_bins[:idx+1])
                    np.random.shuffle(self.re_bins[idx+1:])

                    # iterator exhausted
                    if bin["iter_idx"] == len(bin["entry_inds"]):
                        bin["iter_idx"] = 0
                    break

                # iterator exhausted
                if bin["iter_idx"] == len(bin["entry_inds"]):
                    bin["iter_idx"] = 0

            # assert all are unique in a batch
            assert len(batch_inds) == len(set(batch_inds))
            batches.append(batch_inds)
            batches_answers_set.append(list(set(batch_answers)))

        # each pass adds init_batch_size negatives to each batch
        for pass_idx in tqdm(range(num_passes), total=num_passes, desc="Passes"):
            # select a batch
            for batch_idx in tqdm(range(len(batches)), total=len(batches)):
                # select a sample
                for biter_idx in range(len(batches[batch_idx])):

                    # passed is triggered if we run out of better samples
                    passed = False

                    entry_idx = batches[batch_idx][biter_idx]
                    entry = self.entries[entry_idx]

                    # pick a negative-type
                    neg_choice = np.random.choice(["image_neg", "question_neg", "random"], p=neg_type_weights)

                    image_neg_key, question_neg_key = "same_image_questions", "top_k_questions"
                    if use_gt_answer:
                        image_neg_key, question_neg_key = "same_image_questions", "top_k_questions_neg"


                    if neg_choice == "image_neg":
                        assert len(batches_answers_set) == len(batches)
                        # add better negatives
                        image_negatives = entry[image_neg_key]
                        neg_entry_idx, passed = self.get_negative(batch_idx,
                                                                  image_negatives,
                                                                  batches,
                                                                  batches_extension,
                                                                  use_gt_answer,
                                                                  batches_answers_set)

                    # if image-negatives are exhausted we use question-negatives
                    if neg_choice == "question_neg" or passed:
                        question_neg_topk = entry[question_neg_key][:neg_question_thresh]
                        neg_entry_idx, passed = self.get_negative(batch_idx,
                                                                  question_neg_topk,
                                                                  batches,
                                                                  batches_extension,
                                                                  use_gt_answer,
                                                                  batches_answers_set)

                    # if question-negatives are exhausted we use random negatives
                    if neg_choice == "random" or passed:
                        while True:
                            idx = next(cycle_bins)
                            bin = self.re_bins[idx][1]
                            flag = self.check_bin(bin,
                                                  use_gt_answer,
                                                  batches,
                                                  batches_extension,
                                                  batches_answers_set,
                                                  batch_idx)
                            if flag:
                                continue

                            iter_indices = list(range(len(bin["entry_inds"])))
                            np.random.shuffle(iter_indices)
                            neg_entry_idx = bin["entry_inds"][iter_indices[0]]
                            break

                    try:
                        assert neg_entry_idx != -1
                    except:
                        import pdb
                        pdb.set_trace()
                    batches_extension[batch_idx].append(neg_entry_idx)
                    batches_answers_set[batch_idx].extend(self.get_entry_answers(self.entries[neg_entry_idx]))
                    assert len(set(batches_extension[batch_idx])) == len(batches_extension[batch_idx])

        # join batches together
        for batch, batch_ext in zip(batches, batches_extension):
            batch.extend(batch_ext)

        num_epochs = int(len(batches)/self.num_batches)
        epochs = []

        # build epochs
        for epoch_idx in range(num_epochs):
            batch_start_idx = epoch_idx * self.num_batches
            batch_end_idx = (epoch_idx+1) * self.num_batches
            assert batch_end_idx <= len(batches)
            epoch = []
            for batch_idx in range(batch_start_idx, batch_end_idx):
                assert len(batches[batch_idx]) == len(set(batches[batch_idx]))
                epoch.extend(batches[batch_idx])
            epochs.append(epoch)

        self.epoch_idx = 0
        self.epochs = epochs

    def check_gt_condition(self, entry_idx, batch_answers):
        entry_answers = self.get_entry_answers(self.entries[entry_idx])
        return len(set(batch_answers).intersection(set(entry_answers))) > 0

    def shuffle(self, array, start_idx, end_idx):
        np.random.shuffle(array[start_idx:end_idx])
        for i, item in enumerate(array[start_idx:end_idx]):
            item[1]["bin_idx"] = i + start_idx
        self.assert_bin_inds()
        return array

    def assert_bin_inds(self):
        for i, item in enumerate(self.re_bins):
            assert item[1]["bin_idx"] == i

    def build_batches(self):
        """ Frequency Counter for negs: ({1: 152627, 4: 35500, 5: 4})"""
        self.build_maps()
        batches = []
        self.re_bins = self.shuffle(self.re_bins, start_idx=0, end_idx=len(self.re_bins))
        # add positives for SCL
        add_positives = self.num_positives > 0
        num_batches = int(len(self.data_source) / self.batch_size)
        exhausted_bins = []
        cycle_bins = cycle(list(range(len(self.re_bins))))
        use_gt_answer = self.task_cfg.get("use_gt_answer", False)
        if add_positives:
            assert use_gt_answer is not True
        # replace bins: all bins will be used equally irrespective of their sizes
        neg_replace = self.task_cfg["neg_replace"]
        desc_string = f"SimCLR with neg_replace: {neg_replace}, use_gt: {use_gt_answer}, add_pos: {add_positives} "

        for batch_idx in tqdm(range(num_batches), total=num_batches, desc=desc_string):
            batch_inds = []
            batch_answers = []
            batch_bin_inds = []

            while True:
                idx = next(cycle_bins)

                # skip exhausted bins (only used when use_gt and neg_replace are turned off)
                if idx in exhausted_bins:
                    continue

                # do not repeat bins if we are not exhausting bins with gt_answers
                if (use_gt_answer or neg_replace) and idx in batch_bin_inds:
                    continue

                # pick the value from (key,value) tuple
                item = self.re_bins[idx][1]
                entry_idx = item["entry_inds"][item["iter_idx"]]

                # skip negatives with same ground-truth
                if use_gt_answer and self.check_gt_condition(entry_idx, batch_answers):
                    continue

                item["iter_idx"] += 1
                batch_inds.append(entry_idx)
                batch_bin_inds.append(idx)

                # don't worry about batch_answers for SCL setting
                batch_answers.extend(self.get_entry_answers(self.entries[entry_idx]))

                if add_positives:
                    # only add the needed amount
                    num_pos = min(self.num_positives, self.batch_size - len(batch_inds))
                    self.add_positives(idx, num_pos, use_gt_answer, neg_replace, batch_inds, batch_bin_inds)


                # exit with complete batch
                if len(batch_inds) == self.batch_size:
                    # shuffle left and right parts
                    self.re_bins = self.shuffle(self.re_bins, start_idx=0, end_idx=idx+1)
                    self.re_bins = self.shuffle(self.re_bins, start_idx=idx+1, end_idx=len(self.re_bins))

                    # iterator exhausted
                    self.check_iterator(item, exhausted_bins, idx, neg_replace, use_gt_answer)
                    break

                # iterator exhausted
                self.check_iterator(item, exhausted_bins, idx, neg_replace, use_gt_answer)

            # assert all are unique in a batch
            assert len(batch_inds) == len(set(batch_inds))
            batches.append(batch_inds)

        # stitch all indices together
        epoch_indices = []
        for _batch in batches:
            epoch_indices.extend(_batch)

        return epoch_indices

    def check_iterator(self, item, exhausted_bins=None, bin_idx=-1, neg_replace=False, use_gt_answer=False):
        # iterator exhausted
        if item["iter_idx"] == len(item["entry_inds"]):
            if (use_gt_answer or neg_replace):
                item["iter_idx"] = 0
            else:
                assert bin_idx != -1
                assert exhausted_bins != None
                item["iter_idx"] = None
                exhausted_bins.append(bin_idx)

    def add_positives(self,
                      bin_idx,
                      num_positives,
                      use_gt_answer,
                      neg_replace,
                      batch_inds,
                      batch_bin_inds):

        if num_positives <= 0:
            return

        # sample bin-answer to select positive from
        bin_min_qid = min(self.re_bins[bin_idx][1]["question_ids"])
        bin_answers = self.qid_ann_dict[bin_min_qid]

        filtered_bin_answers = []
        filtered_bin_answers_weights = []

        for ans, freq in bin_answers:
            if freq >= self.bin_ans_threshold and int(ans) in self.answer_map:
                filtered_bin_answers.append(ans)
                filtered_bin_answers_weights.append(freq)

        if len(filtered_bin_answers) <= 0:
            return

        filtered_bin_answers_weights = np.array(filtered_bin_answers_weights)/sum(filtered_bin_answers_weights)
        answer = int(np.random.choice(filtered_bin_answers, 1, p=filtered_bin_answers_weights))

        # sample positives with this answer
        sample_ids, sample_weights = self.answer_map[int(answer)]

        filtered_sample_ids, filtered_sample_weights = [], []

        # filter questions from used-bins
        for sample_id, sample_weight in zip(sample_ids, sample_weights):
            bin_idx = self.entry_map[sample_id]["bin_idx"]
            if bin_idx not in batch_bin_inds:
                filtered_sample_ids.append(sample_id)
                filtered_sample_weights.append(sample_weight)

        if len(filtered_sample_ids) <= 0:
            return

        # re-scale weights
        filtered_sample_weights = np.array(filtered_sample_weights)/sum(filtered_sample_weights)

        # sample two question-ids
        if num_positives >= len(filtered_sample_ids):
            question_ids = filtered_sample_ids
        else:
            question_ids = np.random.choice(filtered_sample_ids, size=num_positives, replace=False, p=filtered_sample_weights)

        bin_inds = []
        entry_inds = []

        # get corresponding bins and update batch
        for qid in question_ids:
            item = self.entry_map[qid]
            bin_inds.append(item["bin_idx"])
            iter_idx = item["iter_idx"]
            try:
                entry_inds.append(item["entry_inds"][iter_idx])
            except:
                import pdb
                pdb.set_trace()
            item["iter_idx"] += 1
            self.check_iterator(item, use_gt_answer=use_gt_answer, neg_replace=neg_replace)

        batch_inds.extend(entry_inds)
        batch_bin_inds.extend(bin_inds)

    def __iter__(self):
        base_path = "datasets/VQA/cache/samplers/"
        assert os.path.exists(base_path)
        cache_name = f"cache_{self.task_cfg['contrastive']}_iter_{self.iter_count}_" \
                     f"split_{self.split}_bt{self.bin_ans_threshold}_ft{self.freq_ans_threshold}_" \
                     f"pos_{self.num_positives}_batch_size_{self.batch_size}.npy"
        cache_name = os.path.join(base_path, cache_name)
        logger.info(f"Sampler Cache Path: {cache_name}")

        if self.task_cfg["contrastive"] == "better":
            # if epochs are exhausted, replenish
            if self.epoch_idx >= len(self.epochs):
                self.build_better_batches()
            epoch_indices = self.epochs[self.epoch_idx]
            self.epoch_idx += 1

        elif self.task_cfg["contrastive"] == "simclr":
            if os.path.exists(cache_name):
                logger.info("Using cache")
                epoch_indices = list(np.load(cache_name, allow_pickle=True))
            else:
                logger.info("Not using cache, and dumping it!")
                epoch_indices = self.build_batches()
                np.save(cache_name, epoch_indices)
        else:
            raise ValueError

        self.iter_count += 1
        return iter(epoch_indices)

    def __len__(self):
        return len(self.data_source)



