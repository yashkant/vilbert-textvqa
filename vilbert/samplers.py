import json
import os
from collections import defaultdict, Counter
from copy import deepcopy
from itertools import cycle
import time
import random
from easydict import EasyDict as edict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import logging
from tools.registry import registry
import multiprocessing as mp

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
        self.better_counter = 0  # increases with every build-call
        self.processing_threads = 16

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
        ann_path = [
            "/nethome/ykant3/vilbert-multi-task/datasets/VQA/v2_mscoco_train2014_annotations.json",
            "/nethome/ykant3/vilbert-multi-task/datasets/VQA/v2_mscoco_val2014_annotations.json",
        ]
        ann_data = json.load(open(ann_path[0]))["annotations"] + json.load(open(ann_path[1]))["annotations"]
        self.qid_ans_dict = {}

        for ann in ann_data:
            ans_counter = Counter()
            for ans in ann["answers"]:
                # only consider answers available in vocabulary
                if ans['answer'] in registry.ans2label:
                    ans_counter[registry.ans2label[ans['answer']]] += 1

            # build dict with qid -> [(ans_label, freq)...]
            self.qid_ans_dict[ann["question_id"]] = ans_counter.most_common()

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

            # skip creating bins for rephrasings
            if entry_map_key in self.entry_map:
                continue

            self.entry_map[entry_map_key] = {
                "question_ids": question_ids,
                "iter_idx": 0,
                "entry_inds": [self.question_map[x] if not registry.debug else 0 for x in question_ids],
                # "answers": self.qid_ans_dict[entry_map_key]
            }

            for (ans_label, freq) in self.qid_ans_dict[entry_map_key]:
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
        for idx, bin in enumerate(self.re_bins):
            bin[1]["bin_idx"] = idx

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
        add_positives = self.num_positives > 0
        num_passes = int((self.batch_size - init_batch_size*(self.num_positives + 1))/init_batch_size)
        assert neg_replace
        init_pass_bs = init_batch_size + self.num_positives*init_batch_size

        # we ensure that we cover all the samples occur at least once
        # num_batches that will be created will be more than no. of batches in a single epoch
        # Todo: We can reduce this by dividing only by samples from re-bins.
        num_batches = int(len(self.entries)/init_batch_size)

        # actual no. of batches to return (for one epoch)
        self.num_batches = int(len(self.entries)/self.batch_size)
        batches_answers_set = []

        # we monitor all the bins using self.entry_map and self.re_bins (both are mapped to same data)
        # self.re_bins = self.shuffle(self.re_bins, start_idx=0, end_idx=len(self.re_bins))

        batches = []
        cycle_bins = cycle(list(range(len(self.re_bins))))
        batches_extension = [[] for _ in range(num_batches)]

        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Initial Pass: "):
            batch_inds = []
            batch_answers = []
            batch_bin_inds = []
            batch_pos_inds = []

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
                # if use_gt_answer:
                #     print("skipping use-gt-answer")
                #     entry_answers = self.get_entry_answers(self.entries[entry_idx])
                #     if len(set(batch_answers).intersection(set(entry_answers))) > 0:
                #         continue

                bin["iter_idx"] += 1
                batch_inds.append(entry_idx)
                batch_answers.extend(self.get_entry_answers(self.entries[entry_idx]))
                batch_bin_inds.append(idx)

                if add_positives:
                    # only add the needed amount
                    num_pos = min(self.num_positives, init_pass_bs - (len(batch_inds) + len(batch_pos_inds)))
                    self.add_positives(idx,
                                       num_pos,
                                       use_gt_answer,
                                       neg_replace,
                                       batch_inds,
                                       batch_bin_inds,
                                       batch_pos_inds=batch_pos_inds)

                # exit initial batches
                if (len(batch_inds) + len(batch_pos_inds)) == init_pass_bs:
                    # shuffle left and right parts
                    self.re_bins = self.shuffle(self.re_bins, start_idx=0, end_idx=idx+1)
                    self.re_bins = self.shuffle(self.re_bins, start_idx=idx+1, end_idx=len(self.re_bins))
                    # iterator exhausted
                    if bin["iter_idx"] == len(bin["entry_inds"]):
                        bin["iter_idx"] = 0
                    break

                # iterator exhausted
                if bin["iter_idx"] == len(bin["entry_inds"]):
                    bin["iter_idx"] = 0

            # append pos-inds at last
            batch_inds.extend(batch_pos_inds)

            # assert all are unique in a batch
            assert len(batch_inds) == len(set(batch_inds))
            batches.append(batch_inds)
            batches_answers_set.append(list(set(batch_answers)))

        # each pass adds init_batch_size negatives to each batch
        for pass_idx in tqdm(range(num_passes), total=num_passes, desc="Passes"):
            # select a batch
            for batch_idx in tqdm(range(len(batches)), total=len(batches)):
                # select a sample
                for biter_idx in range(init_batch_size):

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

    @staticmethod
    def shuffle(array, start_idx, end_idx):
        np.random.shuffle(array[start_idx:end_idx])
        for i, item in enumerate(array[start_idx:end_idx]):
            item[1]["bin_idx"] = i + start_idx
        NegativeSampler.assert_bins(array)
        return array

    def assert_bin_inds(self):
        for i, item in enumerate(self.re_bins):
            assert item[1]["bin_idx"] == i

    def build_batches(self):
        """ Frequency Counter for negs: ({1: 152627, 4: 35500, 5: 4})"""
        self.build_maps()
        num_batches = int(len(self.data_source) / self.batch_size)
        num_threads = 32
        extra_args = edict()
        extra_args.update({
            "batch_size": self.batch_size,
            "num_positives": self.num_positives,
            "bin_ans_threshold": self.bin_ans_threshold
        })

        import multiprocessing as mp
        sp_pool = mp.Pool(num_threads)
        batches_per_thread = [num_batches // num_threads + (1 if x < num_batches % num_threads else 0)
                              for x in range(num_threads)]
        args_list = []
        for _, thread_batches in zip(range(num_threads), batches_per_thread):
            _args = (self.entry_map, self.re_bins, self.answer_map, self.qid_ans_dict, extra_args, thread_batches)
            args_list.append(_args)
        try:
            batches_list = list(sp_pool.imap(NegativeSampler.get_batches, args_list))
        except:
            import pdb
            pdb.set_trace()

        sp_pool.close()
        sp_pool.join()

        import pdb
        pdb.set_trace()

        logger.info(f"Done Processsing Quadrant Spatial Relations with {self.processing_threads} threads")

        # batch = NegativeSampler.get_batch(
        #     deepcopy(self.entry_map),
        #     self.answer_map,
        #     self.qid_ans_dict,
        #     extra_args
        # )
        #
        #
        #
        # batches = []
        # self.re_bins = self.shuffle(self.re_bins, start_idx=0, end_idx=len(self.re_bins))
        # # add positives for SCL
        # add_positives = self.num_positives > 0
        # exhausted_bins = []
        # cycle_bins = cycle(list(range(len(self.re_bins))))
        # use_gt_answer = self.task_cfg.get("use_gt_answer", False)
        # if add_positives:
        #     assert use_gt_answer is not True
        # # replace bins: all bins will be used equally irrespective of their sizes
        # neg_replace = self.task_cfg["neg_replace"]
        # desc_string = f"SimCLR with neg_replace: {neg_replace}, use_gt: {use_gt_answer}, add_pos: {add_positives} "

        # for batch_idx in tqdm(range(num_batches), total=num_batches, desc=desc_string):
        #     batch_inds = []
        #     batch_answers = []
        #     batch_bin_inds = []
        #
        #     while True:
        #         start_time = time.time()
        #         idx = next(cycle_bins)
        #
        #         # skip exhausted bins (only used when use_gt and neg_replace are turned off)
        #         if idx in exhausted_bins:
        #             continue
        #
        #         # do not repeat bins if we are not exhausting bins with gt_answers
        #         if (use_gt_answer or neg_replace) and idx in batch_bin_inds:
        #             continue
        #
        #         # pick the value from (key,value) tuple
        #         item = self.re_bins[idx][1]
        #         entry_idx = item["entry_inds"][item["iter_idx"]]
        #
        #         # skip negatives with same ground-truth
        #         if use_gt_answer and self.check_gt_condition(entry_idx, batch_answers):
        #             continue
        #
        #         item["iter_idx"] += 1
        #         batch_inds.append(entry_idx)
        #         batch_bin_inds.append(idx)
        #
        #         # don't worry about batch_answers for SCL setting
        #         batch_answers.extend(self.get_entry_answers(self.entries[entry_idx]))
        #
        #         pick_entry_time = time.time()
        #         # print(f"pick entry time: {pick_entry_time-start_time}")
        #
        #         if add_positives:
        #             # only add the needed amount
        #             num_pos = min(self.num_positives, self.batch_size - len(batch_inds))
        #             self.add_positives(idx, num_pos, use_gt_answer, neg_replace, batch_inds, batch_bin_inds)
        #
        #         add_pos_time = time.time()
        #         # print(f"add pos time: {add_pos_time-pick_entry_time}")
        #
        #
        #         # exit with complete batch
        #         if len(batch_inds) == self.batch_size:
        #             # Todo: think if shuffling was important here?
        #             # shuffle left and right parts
        #             # self.re_bins = self.shuffle(self.re_bins, start_idx=0, end_idx=idx+1)
        #             # self.re_bins = self.shuffle(self.re_bins, start_idx=idx+1, end_idx=len(self.re_bins))
        #
        #             # iterator exhausted
        #             self.check_iterator(item, exhausted_bins, idx, neg_replace, use_gt_answer)
        #             break
        #
        #         # iterator exhausted
        #         self.check_iterator(item, exhausted_bins, idx, neg_replace, use_gt_answer)
        #         finish_time = time.time()
        #         # print(f"finish time: {finish_time - add_pos_time}")
        #
        #     # assert all are unique in a batch
        #     assert len(batch_inds) == len(set(batch_inds))
        #     batches.append(batch_inds)
        #
        # # stitch all indices together
        # epoch_indices = []
        # for _batch in batches:
        #     epoch_indices.extend(_batch)

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

    @staticmethod
    def get_batches(
        args
    ):
        start_time = time.time()
        entry_map, re_bins, answer_map, qid_ans_dict, extra_args, num_batches = args
        bin_indices = list(range(len(re_bins)))
        batches = []

        try:
            for _ in tqdm(range(num_batches), total=num_batches, desc="Building Batches"):
                # shuffle the bins
                np.random.shuffle(bin_indices)
                # NegativeSampler.shuffle(re_bins, start_idx=0, end_idx=len(re_bins))

                num_positives = extra_args.num_positives
                add_positives = num_positives > 0
                batch_size = extra_args.batch_size

                # start building a batch
                batch_inds = []
                # batch_answers = []
                batch_bin_inds = []

                begin_time = time.time()
                # print(f"Begin time: {begin_time - start_time}")

                for bin_idx in bin_indices:

                    # pick the value from (key,value) tuple
                    item = re_bins[bin_idx][1]

                    # randomly pick one entry from the bin
                    iter_idx = random.sample(range(len(item["question_ids"])), 1)[0]
                    entry_idx = item["entry_inds"][iter_idx]
                    batch_inds.append(entry_idx)
                    batch_bin_inds.append(bin_idx)

                    # entry_answers = entry["answer"]["labels"]
                    # if entry_answers is None:
                    #     entry_answers = []
                    # else:
                    #     entry_answers = entry_answers.tolist()
                    # batch_answers.extend(entry_answers)

                    before_pos = time.time()
                    if add_positives:
                        # only add the needed amount
                        num_pos = min(num_positives, batch_size - len(batch_inds))
                        NegativeSampler.add_positives(
                            re_bins,
                            entry_map,
                            qid_ans_dict,
                            answer_map,
                            bin_idx,
                            num_pos,
                            batch_inds,
                            batch_bin_inds,
                            extra_args,
                        )

                    after_pos = time.time()
                    # print(f"Positive time: {after_pos - before_pos}")

                    if len(batch_inds) == batch_size:
                        break

                batches.append(batch_inds)
        except Exception as e:
            import pdb
            pdb.set_trace()

        # assert all are unique in a batch
        # assert len(batch_inds) == len(set(batch_inds))
        return batches

    @staticmethod
    def assert_bins(array):
        for i, item in enumerate(array):
            assert item[1]["bin_idx"] == i

    @staticmethod
    def add_positives(re_bins,
                      entry_map,
                      qid_ans_dict,
                      answer_map,
                      bin_idx,
                      num_positives,
                      batch_inds,
                      batch_bin_inds,
                      extra_args,
                      batch_pos_inds=None):

        if num_positives <= 0:
            return

        # sample bin-answer to select positive from
        bin_min_qid = min(re_bins[bin_idx][1]["question_ids"])
        bin_answers = qid_ans_dict[bin_min_qid]

        filtered_bin_answers = []
        filtered_bin_answers_weights = []

        for ans, freq in bin_answers:
            if freq >= extra_args.bin_ans_threshold and int(ans) in answer_map:
                filtered_bin_answers.append(ans)
                filtered_bin_answers_weights.append(freq)

        if len(filtered_bin_answers) <= 0:
            return

        filtered_bin_answers_weights = np.array(filtered_bin_answers_weights) / sum(filtered_bin_answers_weights)
        answer = int(np.random.choice(filtered_bin_answers, 1, p=filtered_bin_answers_weights))

        # sample_time = time.time()
        # print(f"sample time: {sample_time - start_time}")

        # sample positives with this answer
        sample_ids, sample_weights = answer_map[int(answer)]
        filtered_sample_ids, filtered_sample_weights = sample_ids, sample_weights

        if len(filtered_sample_ids) <= 0:
            return

        # re-scale weights
        filtered_sample_weights = np.array(filtered_sample_weights) / sum(filtered_sample_weights)

        # sample question-ids
        if num_positives >= len(filtered_sample_ids):
            question_ids = filtered_sample_ids
        else:
            sample_size = min(num_positives + len(batch_inds), len(filtered_sample_ids))
            question_ids = np.random.choice(filtered_sample_ids, size=sample_size, replace=False,
                                            p=filtered_sample_weights)
            # approx 3x faster without weights
            # import random
            # question_ids = [filtered_sample_ids[x] for x in random.sample(range(len(filtered_sample_weights)), sample_size)]

        bin_inds = []
        entry_inds = []

        # get corresponding bins and update batch
        for qid in question_ids:
            item = entry_map[qid]

            # skip if already present in batch
            if item["bin_idx"] in batch_bin_inds:
                continue

            # we sample more than needed to compensate for repeated batch-bins
            # this condition breaks the loop after needed positives in such case
            if len(bin_inds) == num_positives:
                break

            bin_inds.append(item["bin_idx"])
            iter_idx = item["iter_idx"]
            entry_inds.append(item["entry_inds"][iter_idx])
            item["iter_idx"] += 1

        if batch_pos_inds is not None:
            batch_pos_inds.extend(entry_inds)
        else:
            batch_inds.extend(entry_inds)
        batch_bin_inds.extend(bin_inds)

    # def add_positives(self,
    #                   bin_idx,
    #                   num_positives,
    #                   use_gt_answer,
    #                   neg_replace,
    #                   batch_inds,
    #                   batch_bin_inds,
    #                   batch_pos_inds=None):
    #
    #     start_time = time.time()
    #     if num_positives <= 0:
    #         return
    #
    #     # sample bin-answer to select positive from
    #     bin_min_qid = min(self.re_bins[bin_idx][1]["question_ids"])
    #     bin_answers = self.qid_ans_dict[bin_min_qid]
    #
    #     filtered_bin_answers = []
    #     filtered_bin_answers_weights = []
    #
    #     for ans, freq in bin_answers:
    #         if freq >= self.bin_ans_threshold and int(ans) in self.answer_map:
    #             filtered_bin_answers.append(ans)
    #             filtered_bin_answers_weights.append(freq)
    #
    #     if len(filtered_bin_answers) <= 0:
    #         return
    #
    #     filtered_bin_answers_weights = np.array(filtered_bin_answers_weights)/sum(filtered_bin_answers_weights)
    #     answer = int(np.random.choice(filtered_bin_answers, 1, p=filtered_bin_answers_weights))
    #
    #     sample_time = time.time()
    #     # print(f"sample time: {sample_time - start_time}")
    #
    #     # sample positives with this answer
    #     sample_ids, sample_weights = self.answer_map[int(answer)]
    #     # sample_bin_ids = [self.entry_map[qid]["bin_idx"] for qid in sample_ids]
    #     # assert len(sample_bin_ids) == len(set(sample_bin_ids))
    #     #
    #     # remove_inds = []
    #     # for bin_idx in batch_bin_inds:
    #     #     if bin_idx in set_sample_bin_ids:
    #     #         remove_inds.append()
    #
    #
    #     # filtered_bin_inds = set([self.entry_map[sample_id]["bin_idx"] for sample_id in sample_ids]) \
    #     #                       - set(batch_bin_inds)
    #     #
    #     #
    #     filtered_sample_ids, filtered_sample_weights = sample_ids, sample_weights
    #     # # filter questions from used-bins
    #     # for sample_id, sample_weight in zip(sample_ids, sample_weights):
    #     #     bin_idx = self.entry_map[sample_id]["bin_idx"]
    #     #     if bin_idx not in set(batch_bin_inds):
    #     #         filtered_sample_ids.append(sample_id)
    #     #         filtered_sample_weights.append(sample_weight)
    #
    #     # filtered_bin_ids, filtered_bin_weights = [], []
    #
    #     filter_time = time.time()
    #     # print(f"filter time: {filter_time - sample_time}")
    #
    #     if len(filtered_sample_ids) <= 0:
    #         return
    #
    #     # re-scale weights
    #     filtered_sample_weights = np.array(filtered_sample_weights)/sum(filtered_sample_weights)
    #
    #     # sample question-ids
    #     if num_positives >= len(filtered_sample_ids):
    #         question_ids = filtered_sample_ids
    #     else:
    #         sample_size = min(num_positives + len(batch_inds), len(filtered_sample_ids))
    #         question_ids = np.random.choice(filtered_sample_ids, size=sample_size, replace=False, p=filtered_sample_weights)
    #
    #         # approx 3x faster without weights
    #         # import random
    #         # question_ids = [filtered_sample_ids[x] for x in random.sample(range(len(filtered_sample_weights)), sample_size)]
    #
    #     sample2_time = time.time()
    #     print(f"sample-2 time: {sample2_time - filter_time}")
    #
    #
    #     bin_inds = []
    #     entry_inds = []
    #
    #     # get corresponding bins and update batch
    #     for qid in question_ids:
    #         item = self.entry_map[qid]
    #
    #         # skip if already present in batch
    #         if item["bin_idx"] in batch_bin_inds:
    #             continue
    #
    #         # we sample more than needed to compensate for repeated batch-bins
    #         # this condition breaks the loop after needed positives in such case
    #         if len(bin_inds) == num_positives:
    #             break
    #
    #         bin_inds.append(item["bin_idx"])
    #         iter_idx = item["iter_idx"]
    #         entry_inds.append(item["entry_inds"][iter_idx])
    #         item["iter_idx"] += 1
    #         self.check_iterator(item, use_gt_answer=use_gt_answer, neg_replace=neg_replace)
    #
    #     if batch_pos_inds is not None:
    #         batch_pos_inds.extend(entry_inds)
    #     else:
    #         batch_inds.extend(entry_inds)
    #
    #     update_time = time.time()
    #     # print(f"update time: {update_time - sample2_time}")
    #     batch_bin_inds.extend(bin_inds)

    def __iter__(self):
        base_path = "datasets/VQA/cache/samplers/"
        assert os.path.exists(base_path)
        cache_name = f"cache_{self.task_cfg['contrastive']}_iter_{self.iter_count}_" \
                     f"split_{self.split}_bt{self.bin_ans_threshold}_ft{self.freq_ans_threshold}_" \
                     f"pos_{self.num_positives}_batch_size_{self.batch_size}.npy"

        if registry.aug_filter is not None:
            aug_filter_str = f"_aug_fil_max_samples_{registry.aug_filter['max_re_per_sample']}" \
                             f"_sim_thresh_{registry.aug_filter['sim_threshold']}" \
                             f"_sampling_{registry.aug_filter['sampling']}.npy"
            cache_name = cache_name.split(".")[0] + aug_filter_str

        cache_name = os.path.join(base_path, cache_name)

        if self.task_cfg["contrastive"] == "better":
            # if epochs are exhausted, replenish
            if self.epoch_idx >= len(self.epochs):
                # todo: add negs_weights
                cache_name = cache_name.split(".npy")[0] + f"better_cnt_{self.better_counter}_negw_{self.task_cfg['neg_type_weights']}" \
                                                           f"init_bs_{self.task_cfg['init_batch_size']}.npy"
                logger.info(f"Sampler Cache Path: {cache_name}")
                if os.path.exists(cache_name):
                    self.epochs = list(np.load(cache_name, allow_pickle=True))
                    self.epoch_idx = 0
                else:
                    self.build_better_batches()
                    self.better_counter += 1
                    np.save(cache_name, self.epochs)

            epoch_indices = self.epochs[self.epoch_idx]
            self.epoch_idx += 1

        elif self.task_cfg["contrastive"] == "simclr":
            logger.info(f"Sampler Cache Path: {cache_name}")
            # missing_iters = []
            # cache_holder = cache_name.split("_iter_")[0] + "_iter_{}_split_" + cache_name.split("_split_")[-1]
            # for _iter in range(registry.num_epoch):
            #     if not os.path.exists(cache_holder.format(_iter)):
            #         missing_iters.append(_iter)
            #
            # if len(missing_iters) > 0:
            #     logger.info(f"Found missing epochs: {missing_iters}, building them now...")
            #     sp_pool = mp.Pool(len(missing_iters))
            #     epoch_indices_list = list(tqdm(sp_pool.imap(self.build_batches())))
            #
            # import pdb
            # pdb.set_trace()
            # todo: try to optmize the sampling part, if not posible
            if os.path.exists(cache_name) and False:
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



