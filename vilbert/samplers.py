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

    def __init__(self, data_source, batch_size, task_cfg, args, replacement=False, num_samples=None, split="train"):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.split = split
        self.task_cfg = task_cfg["TASK19"]
        self.epoch_idx = int(1e10)
        self.epochs = []

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

    def build_map(self):
        self.entry_map = {}
        for entry in self.entries:
            question_ids = list(set(deepcopy(entry["rephrasing_ids"]) + [entry["question_id"]]))
            question_ids.sort()
            self.entry_map[min(question_ids)] = {
                "question_ids": question_ids,
                "iter_idx": 0,
                "entry_inds": [self.question_map[x] if not registry.debug else 0 for x in question_ids ]
            }

        self.re_bins = sorted(self.entry_map.items(), key=lambda x: x[0])

    def build_better_batches(self):
        self.build_map()
        neg_replace = self.task_cfg["neg_replace"]
        init_batch_size = self.task_cfg["init_batch_size"]
        neg_type_weights = self.task_cfg["neg_type_weights"]
        neg_question_thresh = self.task_cfg["neg_question_thresh"]
        use_gt_answer = self.task_cfg["use_gt_answer"]
        assert np.sum(neg_type_weights) == 1.0
        assert self.batch_size % init_batch_size == 0
        num_passes = int(self.batch_size / init_batch_size) - 1

        # we are dropping the last batch, so we create n-1 batches
        num_batches = int(len(self.entries)/self.batch_size) if not neg_replace else int(len(self.entries)/init_batch_size)
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
                item = self.re_bins[idx]

                # we are not exhausting bins with gt_answers
                if use_gt_answer and idx in batch_bin_inds:
                    continue

                # skip exhausted bins
                if item[1]["iter_idx"] is None:
                    continue

                entry_idx = item[1]["entry_inds"][item[1]["iter_idx"]]

                # skip negatives with same ground-truth
                if use_gt_answer:
                    entry_answers = self.get_entry_answers(self.entries[entry_idx])

                    if len(set(batch_answers).intersection(set(entry_answers))) > 0:
                        continue

                item[1]["iter_idx"] += 1
                batch_inds.append(entry_idx)
                batch_answers.extend(self.get_entry_answers(self.entries[entry_idx]))
                batch_bin_inds.append(idx)

                # exit initial batches
                if len(batch_inds) == init_batch_size:
                    # shuffle left and right parts
                    np.random.shuffle(self.re_bins[:idx+1])
                    np.random.shuffle(self.re_bins[idx+1:])

                    # iterator exhausted
                    if item[1]["iter_idx"] == len(item[1]["entry_inds"]):
                        if not use_gt_answer:
                            item[1]["iter_idx"] = None
                        else:
                            item[1]["iter_idx"] = 0
                    break

                # iterator exhausted
                if item[1]["iter_idx"] == len(item[1]["entry_inds"]):
                    if not use_gt_answer:
                        item[1]["iter_idx"] = None
                    else:
                        item[1]["iter_idx"] = 0

            # assert all are unique in a batch
            assert len(batch_inds) == len(set(batch_inds))
            batches.append(batch_inds)
            batches_answers_set.append(list(set(batch_answers)))

        assert len(batches_answers_set) == len(batches)

        # add better negatives
        if neg_replace:
            for pass_idx in tqdm(range(num_passes), total=num_passes, desc="Passes"):
                for batch_idx in tqdm(range(len(batches)), total=len(batches)):

                    for biter_idx in range(len(batches[batch_idx])):
                        passed = False
                        entry_idx = batches[batch_idx][biter_idx]
                        entry = self.entries[entry_idx]
                        neg_choice = np.random.choice(["image_neg", "question_neg", "random"], p=neg_type_weights)
                        image_neg_key, question_neg_key = "same_image_questions", "top_k_questions"
                        if use_gt_answer:
                            image_neg_key, question_neg_key = "same_image_questions", "top_k_questions_neg"

                        neg_entry_idx = -1
                        if neg_choice == "image_neg":
                            # Todo: Handle cases when all the negatives are exhausted
                            neg_qid = entry[image_neg_key]

                        elif neg_choice == "question_neg":
                            question_neg_topk = entry[question_neg_key][:neg_question_thresh]
                            # instead of randomly picking we shuffle randomly and pick the first available choice
                            np.random.shuffle(question_neg_topk)

                            # iterate over the negatives and pick one without adding a positive sample
                            for qid in question_neg_topk:
                                # flag entries if we have sampled from their bin already
                                source_id = getattr(registry, f"question_rephrase_dict_{self.split}")[qid]
                                bin = self.entry_map[source_id]
                                flag = False

                                for nentry_idx in bin["entry_inds"]:

                                    # check if the entry_inds are already present in batch/batch_extension
                                    if nentry_idx in set(batches[batch_idx]) or nentry_idx in set(batches_extension[batch_idx]):
                                        flag = True
                                        break

                                    # skip negatives with same ground-truth as other negatives in the batch.
                                    if use_gt_answer:
                                        entry_answers = self.get_entry_answers(self.entries[nentry_idx])
                                        if len(set(batches_answers_set[batch_idx]).intersection(set(entry_answers))) > 0:
                                            flag = True
                                            break

                                if not flag:
                                    neg_entry_idx = bin["entry_inds"][bin["question_ids"].index(qid)]
                                    break

                            if neg_entry_idx == -1:
                                passed = True

                        if neg_choice == "random" or passed:
                            patience = 0

                            while True:
                                idx = next(cycle_bins)
                                item = self.re_bins[idx][1]

                                flag = False

                                for nentry_idx in item["entry_inds"]:

                                    # check if the entry_inds are already present in batch/batch_extension
                                    if nentry_idx in set(batches[batch_idx]) or nentry_idx in set(batches_extension[batch_idx]):
                                        flag = True
                                        break

                                    # skip negatives with same ground-truth
                                    if use_gt_answer:
                                        entry_answers = self.get_entry_answers(self.entries[nentry_idx])

                                        if len(set(batches_answers_set[batch_idx]).intersection(set(entry_answers))) > 0:
                                            flag = True
                                            # print("GT filter used")
                                            break

                                if patience > len(self.re_bins):
                                    import pdb
                                    pdb.set_trace()

                                if flag:
                                    patience += 1
                                    continue

                                iter_indices = list(range(len(item["entry_inds"])))
                                np.random.shuffle(iter_indices)
                                neg_entry_idx = item["entry_inds"][iter_indices[0]]
                                patience = 0
                                break

                        try:
                            assert neg_entry_idx != -1
                        except:
                            import pdb
                            pdb.set_trace()
                        batches_extension[batch_idx].append(neg_entry_idx)
                        batches_answers_set[batch_idx].extend(self.get_entry_answers(self.entries[neg_entry_idx]))
                        assert len(set(batches_extension[batch_idx])) == len(batches_extension[batch_idx])
        # else:
        #     num_passes = int(self.batch_size/init_batch_size) - 1
        #     for pass_idx in tqdm(range(num_passes), total=num_passes):
        #         for batch_idx in tqdm(range(len(batches)), total=len(batches)):
        #             for entry_idx in range(len(batches[batch_idx])):
        #                 entry = self.entries[entry_idx]
        #                 neg_choice = np.random.choice(["image_neg", "question_neg", "random"], p=neg_type_weights)
        #
        #                 image_neg_key, question_neg_key = "same_image_questions", "top_k_questions"
        #                 if use_gt_answer:
        #                     image_neg_key, question_neg_key = "same_image_questions", "top_k_questions_neg"
        #
        #                 passed = False
        #                 neg_entry_idx = -1
        #
        #                 if neg_choice == "image_neg":
        #                     # Todo: Handle cases when all the negatives are exhausted
        #                     neg_qid = entry[image_neg_key]
        #
        #                 if neg_choice == "question_neg" or passed:
        #                     question_neg_topk = entry[question_neg_key][:neg_question_thresh]
        #                     # instead of randomly picking we shuffle randomly and pick the first available choice
        #                     np.random.shuffle(question_neg_topk)
        #                     for qid in question_neg_topk:
        #                         source_id = getattr(registry, f"question_rephrase_dict_{self.split}")[qid]
        #                         bin = self.entry_map[source_id]
        #
        #                         if bin["iter_idx"] == None:
        #                             continue
        #
        #                         # check if the qid is available
        #                         if bin["iter_idx"] <= bin["question_ids"].index(qid):
        #                             neg_entry_idx = bin["entry_inds"][bin["question_ids"].index(qid)]
        #                             bin["iter_idx"] += 1
        #                             # insert to front
        #                             bin["question_ids"].remove(qid)
        #                             bin["question_ids"].insert(0, qid)
        #                             # iterator exhausted
        #                             if bin["iter_idx"] == len(bin["entry_inds"]):
        #                                 bin["iter_idx"] = None
        #                             passed = False
        #                             break
        #                     else:
        #                         passed = True
        #
        #                 if neg_choice == "random" or passed:
        #                     while True:
        #                         if use_gt_answer:
        #                             import pdb
        #                             pdb.set_trace()
        #
        #                         idx = next(cycle_bins)
        #                         item = self.re_bins[idx]
        #
        #                         # skip exhausted bins
        #                         if item[1]["iter_idx"] is None:
        #                             continue
        #
        #                         neg_entry_idx = item[1]["entry_inds"][item[1]["iter_idx"]]
        #                         item[1]["iter_idx"] += 1
        #
        #                         # iterator exhausted
        #                         if item[1]["iter_idx"] == len(item[1]["entry_inds"]):
        #                             item[1]["iter_idx"] = None
        #                         passed = False
        #                         break
        #
        #                 assert neg_entry_idx != -1
        #                 batches_extension[batch_idx].append(neg_entry_idx)

        for batch, batch_ext in zip(batches, batches_extension):
            batch.extend(batch_ext)

        num_epochs = int(len(batches)/self.num_batches)
        epochs = []
        for epoch_idx in range(num_epochs):
            batch_start_idx = epoch_idx*self.num_batches
            batch_end_idx = (epoch_idx+1)*self.num_batches
            assert batch_end_idx <= len(batches)
            epoch = []
            for batch_idx in range(batch_start_idx, batch_end_idx):
                assert len(batches[batch_idx]) == len(set(batches[batch_idx]))
                epoch.extend(batches[batch_idx])
            epochs.append(epoch)

        self.epoch_idx = 0
        self.epochs = epochs

    def get_entry_answers(self, entry):
        entry_answers = entry["answer"]["labels"]
        if entry_answers is None:
            entry_answers = []
        else:
            entry_answers = entry_answers.tolist()
        return entry_answers

    def build_batches(self):
        self.build_map()
        batches = []
        np.random.shuffle(self.re_bins)
        num_batches = int(np.ceil(len(self.data_source)/self.batch_size))
        # batches_answers_set = [[] for _ in range(num_batches)]
        exhausted_bins = []
        cycle_bins = cycle(list(range(len(self.re_bins))))
        use_gt_answer = self.task_cfg.get("use_gt_answer", False)
        disobey_threshold = len(self.re_bins)

        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Creating Batches"):
            batch_inds = []
            batch_answers = []
            batch_bin_inds = []
            obey_count = 0

            while True:
                # exit with incomplete last batch
                if len(exhausted_bins) == len(self.re_bins):
                    assert batch_idx == num_batches - 1
                    break

                # don't care for the last batch
                if use_gt_answer and batch_idx == num_batches-1:
                    break

                idx = next(cycle_bins)
                # skip exhausted bins
                if idx in exhausted_bins:
                    continue

                # we are not exhausting bins with gt_answers
                if use_gt_answer and idx in batch_bin_inds:
                    continue

                try:
                    item = self.re_bins[idx]
                    entry_idx = item[1]["entry_inds"][item[1]["iter_idx"]]

                    # skip negatives with same ground-truth
                    if use_gt_answer:
                        # print("Used `GT filter`")
                        entry_answers = self.get_entry_answers(self.entries[entry_idx])
                        if len(set(batch_answers).intersection(set(entry_answers))) > 0:
                            obey_count += 1
                            continue

                    item[1]["iter_idx"] += 1
                    batch_inds.append(entry_idx)
                    batch_answers.extend(self.get_entry_answers(self.entries[entry_idx]))
                    obey_count = 0
                    batch_bin_inds.append(idx)
                except:
                    import pdb
                    pdb.set_trace()

                # exit with complete batch
                if len(batch_inds) == self.batch_size:
                    # shuffle left and right parts
                    np.random.shuffle(self.re_bins[:idx+1])
                    np.random.shuffle(self.re_bins[idx+1:])
                    batch_bin_inds = []
                    # iterator exhausted
                    if item[1]["iter_idx"] == len(item[1]["entry_inds"]):
                        if not use_gt_answer:
                            item[1]["iter_idx"] = None
                            exhausted_bins.append(idx)
                        else:
                            item[1]["iter_idx"] = 0
                    break

                # iterator exhausted
                if item[1]["iter_idx"] == len(item[1]["entry_inds"]):
                    if not use_gt_answer:
                        item[1]["iter_idx"] = None
                        exhausted_bins.append(idx)
                    else:
                        item[1]["iter_idx"] = 0

            # assert all are unique in a batch
            assert len(batch_inds) == len(set(batch_inds))
            batches.append(batch_inds)
            # batches_answers_set.append(set(batch_answers))

        epoch_indices = []
        for _batch in batches:
            epoch_indices.extend(_batch)

        return epoch_indices

    def __iter__(self):
        if self.task_cfg["contrastive"] == "better":

            # if epochs are exhausted, replenish
            if self.epoch_idx >= len(self.epochs):
                self.build_better_batches()

            epoch_indices = self.epochs[self.epoch_idx]
            self.epoch_idx += 1

        elif self.task_cfg["contrastive"] == "simclr":
            epoch_indices = self.build_batches()
        else:
            raise ValueError

        return iter(epoch_indices)

    def __len__(self):
        return len(self.data_source)



