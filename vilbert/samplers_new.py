import itertools
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
import pickle as cPickle

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

        if "trainval" in split:
            self.split = "trainval"
        elif "train" in split:
            self.split = "train"

        if "re" in split:
            self.split = "val"

        self.arg_split = split

        self.task_cfg = task_cfg["TASK19"]
        self.epoch_idx = int(1e10)
        self.epochs = []
        self.epochs_negs = []
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

        if self.task_cfg["contrastive"] == "better":
            logger.info("Loading Hard Negatives")
            self.load_hard_negatives()

    def load_hard_negatives(self):
        # import pdb
        # pdb.set_trace()
        if "fil" in registry.train_split:
            negs_path = "datasets/VQA/back-translate/fil_dcp_sampling_{}_train_question_negs.pkl".format(registry.aug_filter["sampling"])
        elif "cc_v2" in registry.train_split:
            negs_path = "datasets/VQA/cc-re/cc_re_train_question_negs_v2.pkl"
        elif "cc" in registry.train_split:
            negs_path = "datasets/VQA/cc-re/cc_re_train_question_negs.pkl"
        else:
            negs_path = "datasets/VQA/back-translate/fil_{}_question_negs.pkl".format(self.split)

        logger.info(f"Hard negatives path: {negs_path}")
        assert os.path.exists(negs_path)
        self.negs_data = cPickle.load(open(negs_path, "rb"))
        self.negs_index_dict = {}
        for idx, qid in enumerate(self.negs_data["qids"]):
            self.negs_index_dict[qid] = idx

    def add_to_answer_map(self, entry_map_key):
        # maps gt_ans -> [(entry_map_key, ans_freq) ...]
        total_answers = len(self.qid_ans_dict[entry_map_key])

        for (ans_label, freq) in self.qid_ans_dict[entry_map_key]:
            if freq >= self.freq_ans_threshold and \
                    not (total_answers == 2 and freq in [4, 5, 6] and registry.remove_ambiguous):

                if ans_label in self.answer_map:
                    self.answer_map[ans_label].append((entry_map_key, freq))
                else:
                    self.answer_map[ans_label] = [(entry_map_key, freq)]

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
        self.answer_map = {}

        for entry in tqdm(self.entries, desc="Building question and answer maps", total=len(self.entries)):
            question_ids = list(set(deepcopy(entry["rephrasing_ids"]) + [entry["question_id"]]))
            question_ids.sort()
            entry_map_key = min(question_ids)
            # skip creating bins for rephrasings
            if entry_map_key in self.entry_map:
                continue
            try:
                self.entry_map[entry_map_key] = {
                    "question_ids": question_ids,
                    "iter_idx": 0,
                    "entry_inds": [self.question_map[x] if not registry.debug else 0 for x in question_ids],
                }

                self.add_to_answer_map(entry_map_key)
            except:
                import pdb
                pdb.set_trace()

        # post-process: remove duplicates, build sampling weights
        for key in tqdm(self.answer_map.keys(), desc="Post-processing", total=len(self.answer_map)):
            self.answer_map[key] = list(set(self.answer_map[key]))
            ans_labels, freqs = list(zip(*self.answer_map[key]))
            self.answer_map[key] = (cycle(ans_labels), len(ans_labels))
            # not using weights
            # self.answer_map[key][1] = np.array(self.answer_map[key][1])/sum(self.answer_map[key][1])

        self.re_bins = sorted(self.entry_map.items(), key=lambda x: x[0])
        for idx, bin in enumerate(self.re_bins):
            bin[1]["bin_idx"] = idx
            bin[1]["iter_idx"] = cycle(list(range(len(bin[1]["question_ids"]))))

    def get_entry_answers(self, entry):
        entry_answers = entry["answer"]["labels"]
        if entry_answers is None:
            entry_answers = []
        else:
            entry_answers = entry_answers.tolist()
        return entry_answers

    @staticmethod
    def get_hard_negative(negative_list, batch_bins, entry_map, question_rephrase_dict):
        if len(negative_list) == 0:
            return -1, True, -1

        for qid in negative_list:
            if qid < 0:
                assert qid == -1
                break

            # handle case when we don't use all the rephrasings
            if qid not in question_rephrase_dict:
                continue

            source_id = question_rephrase_dict[qid]
            item = entry_map[source_id]
            bin_idx = item["bin_idx"]
            if bin_idx not in batch_bins:
                iter_idx = next(item["iter_idx"])
                entry_idx = item["entry_inds"][iter_idx]
                return entry_idx, False, bin_idx

        return -1, True, -1


    def build_hard_batches(self):
        self.build_maps()
        self.re_bins = NegativeSampler.shuffle(self.re_bins, 0, len(self.re_bins))
        neg_replace = self.task_cfg["neg_replace"]
        init_batch_size = self.task_cfg["init_batch_size"]
        neg_type_weights = self.task_cfg["neg_type_weights"]
        neg_question_thresh = self.task_cfg["neg_question_thresh"]
        use_gt_answer = self.task_cfg["use_gt_answer"]
        assert np.sum(neg_type_weights) == 1.0
        # assert self.batch_size % init_batch_size == 0
        add_positives = self.num_positives > 0
        num_passes = int((self.batch_size - init_batch_size*(self.num_positives + 1))/init_batch_size)
        assert neg_replace
        init_pass_bs = init_batch_size + self.num_positives*init_batch_size
        # assert init_pass_bs > (self.batch_size - init_pass_bs)
        num_batches = int(len(self.entries)/init_batch_size)
        question_rephrase_dict = getattr(registry, f"question_rephrase_dict_{self.split}")

        if "re" in self.arg_split:
            question_rephrase_dict = getattr(registry, f"question_rephrase_dict_train")

        # actual no. of batches to return (for one epoch)
        self.num_batches = int(len(self.entries)/self.batch_size)

        extra_args = edict()
        extra_args.update({
            "num_positives": self.num_positives,
            "bin_ans_threshold": self.bin_ans_threshold
        })

        _args = [self.entry_map,
                 self.re_bins,
                 self.answer_map,
                 self.qid_ans_dict,
                 extra_args,
                 num_batches if num_batches < 20000 else 20000,
                 init_pass_bs]

        # shuffle bins
        self.re_bins = NegativeSampler.shuffle(self.re_bins, 0, len(self.re_bins))

        # intial-pass
        batches, batches_bins = NegativeSampler.get_batches(_args)

        # replace w/ original batch-size
        _args[-1] = self.batch_size
        _args += list([neg_type_weights, self.entries, self.negs_data, self.negs_index_dict])

        # shuffle bins
        self.re_bins = NegativeSampler.shuffle(self.re_bins, 0, len(self.re_bins))
        batches, batches_bins, negatives = NegativeSampler.add_hard_negatives(batches, batches_bins, _args, question_rephrase_dict)

        if registry.sdebug:
            self.debug(batches)

        num_epochs = int(len(batches)/self.num_batches)
        epochs = []
        epochs_negs = []
        # build epochs
        for epoch_idx in range(num_epochs):
            batch_start_idx = epoch_idx * self.num_batches
            batch_end_idx = (epoch_idx+1) * self.num_batches
            assert batch_end_idx <= len(batches)
            epoch = []
            epoch_negs = []
            for batch_idx in range(batch_start_idx, batch_end_idx):
                assert len(batches[batch_idx]) == len(set(batches[batch_idx]))
                epoch.extend(batches[batch_idx])
                epoch_negs.append(negatives[batch_idx])
            epochs.append(epoch)
            epochs_negs.append(epoch_negs)

        self.epoch_idx = 0
        self.epochs = epochs
        self.epochs_negs = epochs_negs


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

        _args = (self.entry_map,
                 self.re_bins,
                 self.answer_map,
                 self.qid_ans_dict,
                 extra_args,
                 num_batches,
                 self.batch_size)

        self.re_bins = NegativeSampler.shuffle(self.re_bins, 0, len(self.re_bins))

        batches, _ = NegativeSampler.get_batches(_args)
        epoch_indices = list(itertools.chain.from_iterable(batches))
        return epoch_indices

    @staticmethod
    def get_batches(
        args
    ):
        entry_map, re_bins, answer_map, qid_ans_dict, extra_args, num_batches, batch_size = args
        batches = []
        batches_bins = []
        num_positives = extra_args.num_positives
        add_positives = num_positives > 0
        bins_iterator = cycle(range(len(re_bins)))

        for _ in tqdm(zip(range(num_batches)), total=num_batches, desc="Building Batches"):

            # start building a batch
            batch_inds = []
            batch_bins = []
            while True:
                bin_idx = next(bins_iterator)

                # to account for bins-used by positive sampler
                if bin_idx in batch_bins:
                    continue

                # pick the value from (key,value) tuple
                item = re_bins[bin_idx][1]

                # randomly pick one entry from the bin
                iter_idx = next(item["iter_idx"])
                entry_idx = item["entry_inds"][iter_idx]
                batch_inds.append(entry_idx)
                batch_bins.append(bin_idx)

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
                        batch_bins,
                        extra_args,
                    )
                if len(batch_inds) == batch_size:
                    break

            assert len(batch_bins) == len(set(batch_bins)) == batch_size
            batches.append(batch_inds)
            batches_bins.append(batch_bins)

        return batches, batches_bins

    @staticmethod
    def add_neg(neg_inds, batch_inds, entry_idx):
        if entry_idx in batch_inds:
            ref_idx = batch_inds.index(entry_idx)
            neg_inds[len(batch_inds)] = [ref_idx]
        else:
            # random negative can be paired w/ any index
            neg_inds[len(batch_inds)] = list(range(len(batch_inds)))
            # np.random.shuffle(neg_inds[len(batch_inds)])

            if entry_idx > -1:
                ref_idx = batch_inds.index(entry_idx)
            else:
                ref_idx = -1

        if ref_idx > -1:
            if ref_idx not in neg_inds:
                neg_inds[ref_idx] = [len(batch_inds)]
            else:
                neg_inds[ref_idx].append(len(batch_inds))
                # np.random.shuffle(neg_inds[ref_idx])

    @staticmethod
    def add_hard_negatives(batches,
                           batches_bins,
                           args,
                           question_rephrase_dict):

        (
            entry_map,
            re_bins,
            answer_map,
            qid_ans_dict,
            extra_args,
            num_batches,
            batch_size,
            neg_type_weights,
            entries,
            negs_data,
            negs_index_dict
        ) = args

        bins_iterator = cycle(range(len(re_bins)))
        negatives = [{} for batch in batches]
        actual_negatives = []

        for batch_inds, neg_inds, batch_bins in tqdm(zip(batches, negatives, batches_bins), total=len(batches), desc="Adding Hard Negatives"):
            batch_inds_iter = cycle(batch_inds)

            while True:
                neg_choice = np.random.choice(["image_neg", "question_neg", "random"], p=neg_type_weights)
                passed = False

                if neg_choice in ["image_neg"]:
                    entry_idx = next(batch_inds_iter)
                    question_id = entries[entry_idx]["question_id"]
                    negs_idx = negs_index_dict[question_id]
                    negatives_list = negs_data["same_image_questions_neg"][negs_idx]
                    # add better negatives
                    neg_entry_idx, passed, bin_idx = NegativeSampler.get_hard_negative(negatives_list, batch_bins, entry_map, question_rephrase_dict)
                    if not passed:
                        NegativeSampler.add_neg(neg_inds, batch_inds, entry_idx)
                        batch_inds.append(neg_entry_idx)
                        batch_bins.append(bin_idx)


                if neg_choice in ["question_neg"] or passed:
                    entry_idx = next(batch_inds_iter)
                    entry = entries[entry_idx]
                    question_id = entry["question_id"]

                    if "top_k_questions_neg" in entry:
                        negatives_list = entry["top_k_questions_neg"]
                    else:
                        negs_idx = negs_index_dict[question_id]
                        negatives_list = negs_data["question_negs"][negs_idx]

                    # add better negatives
                    neg_entry_idx, passed, bin_idx = NegativeSampler.get_hard_negative(negatives_list, batch_bins, entry_map, question_rephrase_dict)
                    if not passed:
                        NegativeSampler.add_neg(neg_inds, batch_inds, entry_idx)
                        batch_inds.append(neg_entry_idx)
                        batch_bins.append(bin_idx)

                if neg_choice == "random" or passed:
                    prev_entry_idx = entry_idx if passed else -1
                    while True:
                        bin_idx = next(bins_iterator)
                        # to account for bins-used by positive sampler
                        if bin_idx in batch_bins:
                            continue
                        # pick the value from (key,value) tuple
                        item = re_bins[bin_idx][1]
                        # randomly pick one entry from the bin
                        iter_idx = next(item["iter_idx"])
                        entry_idx = item["entry_inds"][iter_idx]
                        NegativeSampler.add_neg(neg_inds, batch_inds, prev_entry_idx)
                        batch_inds.append(entry_idx)
                        batch_bins.append(bin_idx)
                        break

                if len(batch_inds) == batch_size:
                    assert len(batch_bins) == len(set(batch_bins)) == batch_size
                    break
                elif len(batch_inds) > batch_size:
                    import pdb
                    pdb.set_trace()

            # assert that we have negatives for all the samples
            try:
                assert len(neg_inds) == len(batch_inds)

            except:
                keys_left = list(set(range(len(batch_inds))) - set(neg_inds.keys()))
                for key in keys_left:
                    neg_inds[key] = [np.random.choice(list(neg_inds.keys()))]
                assert len(neg_inds) == len(batch_inds)

            # filter neg-inds
            actual_neg_inds = []
            for key in sorted(neg_inds):
                try:
                    actual_neg_inds.append(np.random.choice (neg_inds[key]))
                except:
                    import pdb
                    pdb.set_trace()

            actual_negatives.append(actual_neg_inds)

        return batches, batches_bins, actual_negatives

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
                      batch_bins,
                      extra_args):

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

        count_pos = 0
        qids_iter, qids_len = answer_map[int(answer)]
        start_qid = next(qids_iter)

        # get corresponding bins and update batch
        qid = start_qid
        while True:
            item = entry_map[qid]

            # skip if already present in batch
            if item["bin_idx"] in batch_bins:
                qid = next(qids_iter)
                # we have exhausted all positives
                if qid == start_qid:
                    break

                continue
            # this condition breaks the loop after needed positives in such case
            if count_pos == num_positives:
                break
            batch_bins.append(item["bin_idx"])
            iter_idx = next(item["iter_idx"])
            entry_idx = item["entry_inds"][iter_idx]
            batch_inds.append(entry_idx)
            count_pos += 1
            qid = next(qids_iter)
            # we have exhausted all positives
            if qid == start_qid:
                break

    @staticmethod
    def check_iterator(bin):
        if bin["iter_idx"] >= len(bin["entry_inds"]) - 1:
            bin["iter_idx"] = 0
        else:
            bin["iter_idx"] += 1


    def __iter__(self):
        base_path = "datasets/VQA/cache/samplers/"
        assert os.path.exists(base_path)
        cache_name = f"latest_cache_{self.task_cfg['contrastive']}_iter_{self.iter_count}_" \
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
                cache_name = cache_name.split(".npy")[0] + f"better_cnt_{self.better_counter}_negw_{self.task_cfg['neg_type_weights']}" \
                                                           f"init_bs_{self.task_cfg['init_batch_size']}.npy"
                logger.info(f"Sampler Cache Path: {cache_name}")
                if os.path.exists(cache_name)  and False:
                    self.epochs = list(np.load(cache_name, allow_pickle=True))
                    self.epoch_idx = 0
                else:
                    self.build_hard_batches()
                    self.better_counter += 1
                    np.save(cache_name, self.epochs)

            epoch_indices = self.epochs[self.epoch_idx]
            self.epoch_idx += 1

        elif self.task_cfg["contrastive"] == "simclr":
            logger.info(f"Sampler Cache Path: {cache_name}")
            if os.path.exists(cache_name) and False:
                logger.info("Using cache")
                epoch_indices = list(np.load(cache_name, allow_pickle=True))
            else:
                logger.info("Not using cache, and dumping it!")
                epoch_indices = self.build_batches()
                np.save(cache_name, epoch_indices)
        else:
            raise ValueError
        registry.sampler_cache_name = cache_name

        logger.info(f"No. of Unique Samples: {len(set(epoch_indices))} / {len(epoch_indices)}")
        self.iter_count += 1
        return iter(epoch_indices)

    def __len__(self):
        return len(self.data_source)

    def debug(self, batches):
        ans_freq_counters = []

        import pdb
        pdb.set_trace()

        for batch in batches:
            batch_answers = [self.entries[idx]["answer"]["labels"][:1] for idx in batch]
            batch_answers = list(itertools.chain.from_iterable(batch_answers))
            counter = Counter(batch_answers)
            ans_freq_counters.append(counter)

        import pdb
        pdb.set_trace()
