from copy import deepcopy
from itertools import cycle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from tools.registry import registry


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
        neg_replace = False
        init_batch_size = 26
        neg_type_weights = [0.0, 0.8, 0.2]
        neg_question_thresh = 200
        use_gt_answer = False

        # we are dropping the last batch, so we create n-1 batches
        num_batches = int(len(self.entries)/self.batch_size) if not neg_replace else int(len(self.entries)/init_batch_size)

        # we monitor all the bins using self.entry_map and self.re_bins (both are mapped to same data)
        np.random.shuffle(self.re_bins)

        batches = []
        cycle_bins = cycle(list(range(len(self.re_bins))))
        batches_extension = [[]] * len(batches)

        for batch_idx in (range(num_batches)):
            batch_inds = []
            while True:
                # # exit with incomplete last batch
                # if len(exhausted_bins) == len(self.re_bins):
                #     assert batch_idx == num_batches - 1
                #     break

                idx = next(cycle_bins)
                item = self.re_bins[idx]

                # skip exhausted bins
                if item[1]["iter_idx"] is None:
                    continue

                entry_idx = item[1]["entry_inds"][item[1]["iter_idx"]]

                item[1]["iter_idx"] += 1
                batch_inds.append(entry_idx)

                # exit initial batches
                if len(batch_inds) == init_batch_size:
                    # shuffle left and right parts
                    np.random.shuffle(self.re_bins[:idx+1])
                    np.random.shuffle(self.re_bins[idx+1:])

                    # iterator exhausted
                    if item[1]["iter_idx"] == len(item[1]["entry_inds"]):
                        item[1]["iter_idx"] = None
                    break

                # iterator exhausted
                if item[1]["iter_idx"] == len(item[1]["entry_inds"]):
                    item[1]["iter_idx"] = None

            batches.append(batch_inds)

        # add better negatives
        if neg_replace:
            pass
        else:
            num_passes = int(self.batch_size/init_batch_size) - 1
            for pass_idx in range(num_passes):
                for batch_idx in range(len(batches)):
                    for entry_idx in range(len(batches[batch_idx])):
                        entry = self.entries[entry_idx]
                        neg_choice = np.random.choice(["image_neg", "question_neg", "random"], p=neg_type_weights)

                        image_neg_key, question_neg_key = None, None
                        if use_gt_answer:
                            image_neg_key, question_neg_key = None, None

                        passed = False
                        neg_entry_idx = -1

                        if neg_choice == "image_neg":
                            # Todo: Handle cases when all the negatives are exhausted
                            neg_qid = entry[image_neg_key]

                        if neg_choice == "question_neg" or passed:
                            question_neg_topk = entry[question_neg_key][:neg_question_thresh]
                            # instead of randomly picking we shuffle randomly and pick the first available choice
                            np.random.shuffle(question_neg_topk)
                            for qid in question_neg_topk:
                                source_id = getattr(registry, f"question_rephrase_dict_{self.split}")[qid]
                                bin = self.entry_map[source_id][1]

                                # check if the qid is available
                                if bin["iter_idx"] <= bin["question_ids"].index(qid):
                                    neg_entry_idx = bin["entry_inds"][bin["question_ids"].index(qid)]
                                    bin["iter_idx"] += 1
                                    # insert to front
                                    bin["question_ids"].remove(qid)
                                    bin["question_ids"].append(qid)
                                    # iterator exhausted
                                    if bin["iter_idx"] == len(bin["entry_inds"]):
                                        bin["iter_idx"] = None

                                    passed = False
                                    break
                            else:
                                passed = True

                        if neg_choice == "random" or passed:
                            while True:
                                if use_gt_answer:
                                    import pdb
                                    pdb.set_trace()

                                idx = next(cycle_bins)
                                item = self.re_bins[idx]

                                # skip exhausted bins
                                if item[1]["iter_idx"] is None:
                                    continue

                                neg_entry_idx = item[1]["entry_inds"][item[1]["iter_idx"]]
                                item[1]["iter_idx"] += 1

                                # iterator exhausted
                                if item[1]["iter_idx"] == len(item[1]["entry_inds"]):
                                    item[1]["iter_idx"] = None
                                passed = False
                                break

                        assert neg_entry_idx != -1
                        batches_extension[batch_idx].append(neg_entry_idx)


        for batch, batch_ext in zip(batches, batches_extension):
            batch.extend(batch_ext)


        epoch_indices = []
        for _batch in batches:
            epoch_indices.extend(_batch)




    def build_batches(self):
        self.build_map()
        batches = []
        np.random.shuffle(self.re_bins)
        num_batches = int(np.ceil(len(self.data_source)/self.batch_size))
        exhausted_bins = []
        cycle_bins = cycle(list(range(len(self.re_bins))))


        for batch_idx in (range(num_batches)):
            batch_inds = []
            while True:
                # exit with incomplete last batch
                if len(exhausted_bins) == len(self.re_bins):
                    assert batch_idx == num_batches - 1
                    break

                idx = next(cycle_bins)
                # skip exhausted bins
                if idx in exhausted_bins:
                    continue

                item = self.re_bins[idx]
                entry_idx = item[1]["entry_inds"][item[1]["iter_idx"]]

                item[1]["iter_idx"] += 1
                batch_inds.append(entry_idx)

                # exit with complete batch
                if len(batch_inds) == self.batch_size:
                    # shuffle left and right parts
                    np.random.shuffle(self.re_bins[:idx+1])
                    np.random.shuffle(self.re_bins[idx+1:])

                    # iterator exhausted
                    if item[1]["iter_idx"] == len(item[1]["entry_inds"]):
                        item[1]["iter_idx"] = None
                        exhausted_bins.append(idx)
                    break

                # iterator exhausted
                if item[1]["iter_idx"] == len(item[1]["entry_inds"]):
                    item[1]["iter_idx"] = None
                    exhausted_bins.append(idx)

            batches.append(batch_inds)

        epoch_indices = []
        for _batch in batches:
            epoch_indices.extend(_batch)

        return epoch_indices

    def __iter__(self):
        # Todo: What's happening with the last batch? (drop_last is True)
        #  drop the last batch for now and move on
        #  Dataloader expects indices_len == len(dataset)
        epoch_indices = self.build_batches()
        return iter(epoch_indices)

    def __len__(self):
        return len(self.data_source)
