import argparse
import bisect
import json
import logging
import os
import random
from collections import defaultdict
from copy import deepcopy
from io import open
from itertools import combinations

import numpy as np
import pprint
from tensorboardX import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict

import pdb
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from tools.registry import registry

import vilbert.utils as utils
import torch.distributed as dist

from vilbert.metrics import get_consistency_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_model(model, resume_file):
    logger.info(f"Resuming from Checkpoint: {resume_file}")
    checkpoint = torch.load(resume_file, map_location="cpu")
    new_dict = {}

    for attr in checkpoint["model_state_dict"]:
        if not attr.startswith("module."):
            new_dict["module." + attr] = checkpoint["model_state_dict"][attr]
        else:
            new_dict[attr] = checkpoint["model_state_dict"][attr]

    model.load_state_dict(new_dict)

    # Add checkpoint string
    log_keys = ["cs_rank", "vqa_rank", "vqa_acc", "cs_score", "global_step", "cs_bt_rank", "cs_score", "cs_bt_score"]
    ckpt_string = f"-------------- \n Checkpoint information: \n"

    for key in log_keys:
        if key in checkpoint:
            ckpt_string += f"{key}: {checkpoint[key]} \n"

    ckpt_string += "---------------"
    logger.info(ckpt_string)
    print("Not loading optimizer and scheduler states")
    del checkpoint

    return model




def final_evaluate(
    args,
    task_cfg,
    device,
    task_id,
    model,
    task_losses,
    resume_dir,
    val_split
):
    from vilbert.task_utils import LoadDatasets

    # LOAD DATASETS
    task_cfg[task_id]["val_split"] = val_split
    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, \
    task_dataloader_val = LoadDatasets(args, task_cfg, args.tasks.split("-"), split="val")

    resume_file = resume_dir + "/cs_bt_rank_3.tar"

    if not os.path.exists(resume_file):
        import pdb
        pdb.set_trace()

    model = set_model(model, resume_file)

    from vilbert.task_utils import ForwardModelsVal
    registry.eval_only = True

    # reset revqa_bins for each evaluation!
    if registry.revqa_eval or registry.revqa_eval_on_val:
        from easydict import EasyDict
        dd = defaultdict(list)
        dd_bt = defaultdict(list)

        super(EasyDict, registry).__setattr__("revqa_bins", dd)
        super(EasyDict, registry).__setitem__("revqa_bins", dd)

        super(EasyDict, registry).__setattr__("revqa_bt_bins", dd_bt)
        super(EasyDict, registry).__setitem__("revqa_bt_bins", dd_bt)

    model.to(torch.device("cuda"))
    model.eval()
    results = {}

    for batch in tqdm(task_dataloader_val[task_id]):
        with torch.no_grad():  # turn off autograd engine
            batch_dict = ForwardModelsVal(
                None, None, device, task_id, batch, model, task_losses, revqa_eval=registry.revqa_eval_on_val
            )
            # build the json file here!
            logits = torch.max(batch_dict["vil_prediction"], 1)[1].data  # argmax
            for idx in range(logits.size(0)):
                results[batch_dict["question_id"][idx].item()] = \
                    {
                        "question_id": batch_dict["question_id"][idx].item(),
                        "answer": task_dataloader_val[task_id].dataset.label2ans[
                            logits[idx].item()
                        ],
                        "vqa_score": np.round(batch_dict["vqa_scores"][idx], 1) if "vqa_scores" in batch_dict else None
                    }

    # make all none
    c_scores, revqa_bins_scores = None, None
    c_scores_bt, revqa_bins_scores_bt = None, None

    # if registry.revqa_eval and val_split == "val":
    #     cs_results = {}
    #     for batch in tqdm(task_dataloader_val["revqa"], desc="Revqa Eval"):
    #         with torch.no_grad():  # turn off autograd engine
    #             batch_dict = ForwardModelsVal(
    #                 None, None, device, task_id, batch, model, task_losses, revqa_eval=True, revqa_split="re_total"
    #             )
    #             # build the json file here!
    #             logits = torch.max(batch_dict["vil_prediction"], 1)[1].data  # argmax
    #             for idx in range(logits.size(0)):
    #                 cs_results[batch_dict["question_id"][idx].item()] = \
    #                     {
    #                         "question_id": batch_dict["question_id"][idx].item(),
    #                         "answer": task_dataloader_val[task_id].dataset.label2ans[
    #                             logits[idx].item()
    #                         ],
    #                         "vqa_score": np.round(batch_dict["vqa_scores"][idx], 1) if "vqa_scores" in batch_dict else None
    #                     }
    #     c_scores, revqa_bins_scores = get_consistency_score(results=cs_results, bins_scores=True)
    #
    #     cs_results_bt = {}
    #     for batch in tqdm(task_dataloader_val["revqa_bt"], desc="Revqa BT Eval"):
    #         with torch.no_grad():  # turn off autograd engine
    #             batch_dict = ForwardModelsVal(
    #                 None, None, device, task_id, batch, model, task_losses, revqa_eval=True, revqa_split="re_total_bt"
    #             )
    #             # build the json file here!
    #             logits = torch.max(batch_dict["vil_prediction"], 1)[1].data  # argmax
    #             for idx in range(logits.size(0)):
    #                 cs_results_bt[batch_dict["question_id"][idx].item()] = \
    #                     {
    #                         "question_id": batch_dict["question_id"][idx].item(),
    #                         "answer": task_dataloader_val[task_id].dataset.label2ans[
    #                             logits[idx].item()
    #                         ],
    #                         "vqa_score": np.round(batch_dict["vqa_scores"][idx], 1) if "vqa_scores" in batch_dict else None
    #                     }
    #     c_scores_bt, revqa_bins_scores_bt = get_consistency_score(results=cs_results_bt, bins_scores=True, bins_key="revqa_bt_bins")
    #
    # elif registry.revqa_eval_on_val and val_split == "val":
    #     logger.info("Ran re-vqa eval on validation!")
    #     c_scores, revqa_bins_scores = get_consistency_score(results=results, bins_scores=True)

    import pdb
    pdb.set_trace()

    final_results = {}
    final_results["results"] = results
    final_results["revqa_bins_scores"] = revqa_bins_scores
    final_results["c_scores"] = c_scores
    final_results["revqa_bins_scores_bt"] = revqa_bins_scores_bt
    final_results["c_scores_bt"] = c_scores_bt

    for key in registry:
        if "question_rephrase_dict" in key:
            final_results[key] = registry[key]

    evalai_results = deepcopy(list(results.values()))
    for item in evalai_results:
        del item["vqa_score"]

    save_dir = os.path.split(resume_file)[0]
    evalai_path = f"{save_dir}/evalai_{val_split}.json"
    preds_path = f"{save_dir}/preds_revqa_{val_split}.json"

    json.dump(evalai_results, open(evalai_path, "w"))
    json.dump(final_results, open(preds_path, "w"))

    print(f"Dumped: {evalai_path}")
    print(f"Dumped: {preds_path}")

    model.train()
