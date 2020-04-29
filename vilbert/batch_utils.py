from collections import defaultdict, Counter

import numpy as np
import torch


def build_scl_mask(batch_dict):
    assert len(batch_dict) == 2
    targets = torch.cat([batch_dict[0]["target"], batch_dict[1]["target"]])
    non_zero_targets = targets.nonzero().tolist()
    ans_inds_map = defaultdict(list)
    for idx, label in non_zero_targets:
        ans_inds_map[label].append(idx)

    # initialize the scl-mask
    batch_size = len(batch_dict[0]["target"])
    diag = np.eye(2 * batch_size)
    l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
    l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
    scl_mask = torch.from_numpy((diag + l1 + l2))
    scl_mask = (1 - scl_mask).type(torch.bool)
    for idx, label in non_zero_targets:
        for label_ind in ans_inds_map[label]:
            scl_mask[idx][label_ind] = 0

    batch_dict[0]["scl_mask"] = scl_mask
    batch_dict[1]["scl_mask"] = scl_mask

    # batch composition
    # cnt = Counter()
    # cnt.update(non_zero_targets)
    # cnt.most_common()

