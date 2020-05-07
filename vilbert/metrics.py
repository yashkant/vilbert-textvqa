from itertools import combinations
from tools.registry import registry
import numpy as np


def get_consistency_score():
    # bin the vqa-scores
    revqa_bins_scores = {}

    # vqa-score of all the samples
    total_vqa_scores = []

    for key, value in registry.revqa_bins.items():
        k_values = range(1, 1 + len(value))
        revqa_bins_scores[key] = {
            "vqa_scores": value,
        }

        total_vqa_scores.extend(value)

        # for subsets of size = k, check VQA accuracy
        for k_value in k_values:
            value_subsets = list(combinations(value, k_value))
            value_subset_scores = []

            # this loop is causing problems!
            for subset in value_subsets:
                if 0.0 not in subset:
                    value_subset_scores.append(1.0)
                else:
                    value_subset_scores.append(0.0)
            revqa_bins_scores[key][k_value] = sum(value_subset_scores) / len(value_subsets)

    result_dict = {}

    # Consistency Score Calculation
    max_k = 4
    for k_value in range(1, max_k+1):
        scores = []
        for key, value in revqa_bins_scores.items():
            # only consider questions that have all the rephrasings available
            if max_k in value:
                scores.append(value[k_value])
        # print(f"Consensus Score with K={k_value} is {sum(scores) / len(scores)}")
        result_dict[int(k_value)] = sum(scores) / len(scores)
        result_dict[f"len_{k_value}"] = len(scores)


    return result_dict

