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

    def __init__(self, data_source, batch_size, task_cfg, args, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.batch_size = batch_size

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
                "data_inds": [self.question_map[x] if not registry.debug else 0 for x in question_ids ]
            }

        self.map_items = sorted(self.entry_map.items(), key=lambda x: x[0])

    def build_batches(self):
        self.build_map()
        batches = []
        np.random.shuffle(self.map_items)
        num_batches = int(np.ceil(len(self.data_source)/self.batch_size))
        exhausted_items = []
        cycle_inds = cycle(list(range(len(self.map_items))))


        for batch_idx in (range(num_batches)):
            batch_inds = []
            while True:
                idx = next(cycle_inds)
                item = self.map_items[idx]

                # exit with incomplete last batch
                if len(exhausted_items) == len(self.map_items):
                    assert batch_idx == num_batches - 1
                    break


                # skip exhausted bins
                if idx in exhausted_items:
                    continue

                data_idx = item[1]["data_inds"][item[1]["iter_idx"]]
                item[1]["iter_idx"] += 1
                batch_inds.append(data_idx)

                # exit with complete batch
                if len(batch_inds) == self.batch_size:
                    # shuffle left and right parts
                    np.random.shuffle(self.map_items[:idx+1])
                    np.random.shuffle(self.map_items[idx+1:])
                    # batch_inds = []
                    # iterator exhausted
                    if item[1]["iter_idx"] == len(item[1]["data_inds"]):
                        item[1]["iter_idx"] = None
                        exhausted_items.append(idx)
                    break

                # iterator exhausted
                if item[1]["iter_idx"] == len(item[1]["data_inds"]):
                    item[1]["iter_idx"] = None
                    exhausted_items.append(idx)

            batches.append(batch_inds)

        epoch_indices = []
        for _batch in batches:
            epoch_indices.extend(_batch)

        return epoch_indices

    def __iter__(self):
        epoch_indices = self.build_batches()
        return iter(epoch_indices)

    def __len__(self):
        return len(self.data_source)
