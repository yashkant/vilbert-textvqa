import logging
import time
from bisect import bisect
from collections import defaultdict

import torch
import torch.nn as nn
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    LambdaLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.registry import registry
from vilbert.datasets.vqa_dataset import VQAClassificationDataset
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
from vilbert.losses import ScaledSupConLoss
from vilbert.metrics import get_consistency_score

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "SCLLoss": ScaledSupConLoss(temperature=registry.temperature,
                                formulation=registry.scl_formulation,
                                base_temperature=registry.base_temperature),  # using the default parameter setting
}


def clip_gradients(model, max_grad_l2_norm, clip_norm_mode):
    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2_norm)
        elif clip_norm_mode == "question":
            question_embedding = model.module.question_embedding_module
            norm = nn.utils.clip_grad_norm(
                question_embedding.parameters(), max_grad_l2_norm
            )
        else:
            raise NotImplementedError("Clip norm mode %s not implemented" % clip_norm_mode)


def get_optim_scheduler(task_cfg,
                        optimizer_grouped_parameters,
                        base_lr,
                        ):
    optimizer = Adam(optimizer_grouped_parameters, lr=base_lr)
    warmup_iters = task_cfg["warmup_iters"]
    warmup_factor = task_cfg["warmup_factor"]
    lr_decay_iters = task_cfg["lr_decay_iters"]
    lr_decay = task_cfg["lr_decay"]

    def lr_update(_iter):
        if _iter <= warmup_iters:
            alpha = float(_iter) / float(warmup_iters)
            return warmup_factor * (1.0 - alpha) + alpha
        else:
            idx = bisect(lr_decay_iters, _iter)
            return pow(lr_decay, idx)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_update)
    return optimizer, warmup_scheduler


def to_device(batch_dict, device):
    if device.type == "cpu":
        return
    for batch in batch_dict:
        for key, value in batch.items():
            if key in ["image_id", "question_id"]:
                continue
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda(device=device, non_blocking=True)


def forward_eval(device,
                 batch_dict,
                 model,
                 revqa_eval=False,
                 revqa_split="revqa",
                 return_batch=False):
    if not isinstance(batch_dict, tuple) and not isinstance(batch_dict, list):
        batch_dict = [batch_dict]

    batch_size = len(batch_dict[0]["question_id"])
    for batch in batch_dict:
        results_dict = run_model(batch, model, device)
        batch.update(results_dict)

    loss, batch_score = add_ce_loss(batch_dict[0], device, val_run=True, revqa_eval=revqa_eval, split=revqa_split)

    # testing
    if registry.get("eval_only", False):
        return batch_dict[0]

    del results_dict
    del batch_dict

    return float(loss), float(batch_score), batch_size


def get_batch(dataloaders, dkey):
    ikey = dkey + "_iter"
    load_epoch = ikey not in dataloaders

    if not load_epoch:
        batch_dicts = next(dataloaders[ikey], None)
        if batch_dicts is None:
            load_epoch = True

    if load_epoch:
        dataloaders[ikey] = iter(dataloaders[dkey])
        batch_dicts = next(dataloaders[ikey], None)
        assert batch_dicts is not None

    return batch_dicts


def run_model(batch, model, device):
    # send to gpu
    input_keys = list(batch.keys())
    for key in input_keys:
        if key in ["image_id", "question_id"]:
            continue
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda(device=device, non_blocking=True)

    results_dict = model(batch)
    # delete batch-inputs (only keep results)
    for key in input_keys:
        if key in ["image_id", "question_id", "target"]:
            continue
        else:
            del batch[key]
    return results_dict


def forward_train(device, dataloaders, model, train_type):
    if registry.debug:
        train_type = "ce"

    if train_type == "ce":
        batch_dicts = get_batch(dataloaders, "train_ce")
        # throw away rephrasings batch
        batch_dicts = batch_dicts[:1]
    else:
        batch_dicts = get_batch(dataloaders, "train_scl")

    for batch in batch_dicts:
        results_dict = run_model(batch, model, device)
        batch.update(results_dict)

    if train_type == "scl":
        loss, batch_score = LossMap["SCLLoss"](batch_dicts)
    else:
        loss, batch_score = add_ce_loss(batch_dicts, device)

    del batch_dicts
    return loss, float(batch_score)


def add_ce_loss(batch_dict, device, val_run=False, revqa_eval=False, split="revqa"):
    if len(batch_dict) == 2 and not val_run:
        # train time
        vil_preds = torch.cat([batch_dict[0]["vil_prediction"], batch_dict[1]["vil_prediction"]], dim=0)
        vil_targets = torch.cat([batch_dict[0]["target"], batch_dict[1]["target"]], dim=0)
    else:
        if len(batch_dict) == 1:
            batch_dict = batch_dict[0]
        # validation time
        vil_preds = batch_dict["vil_prediction"]
        vil_targets = batch_dict["target"]

    vl_loss = LossMap["BCEWithLogitLoss"](vil_preds, vil_targets)
    vl_loss = vl_loss.mean() * vil_targets.size(1)
    batch_scores = compute_score_with_logits(vil_preds, vil_targets, device)
    batch_score = batch_scores.sum() / len(vil_preds)

    if isinstance(batch_dict, dict):
        # fill the scores for each question into the batch-dict
        batch_dict["vqa_scores"] = batch_scores.sum(dim=-1).tolist()

    # calculate consistency scores during validation run!
    if revqa_eval:
        # add vqa-scores to defaultdict(list) for each bin
        for idx, qid in enumerate(batch_dict["question_id"].tolist()):
            min_qid = registry[f"question_rephrase_dict_{split}"][qid]
            vqa_score = batch_dict["vqa_scores"][idx]
            bins_key = "revqa_bins" if split in ["revqa", "val"] else "revqa_bt_bins"
            registry[bins_key][min_qid].append((qid, vqa_score))

    return vl_loss, batch_score


def load_dataset(task_cfg):
    from vilbert.samplers import RandomSampler, ContrastiveSampler
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # image-features
    trainval_features = ImageFeaturesH5Reader(task_cfg["trainval_features_path"])
    test_features = ImageFeaturesH5Reader(task_cfg["test_features_path"])

    dataloaders = {}

    # one train split and multiple evaluation splits
    # load_splits = [task_cfg["train_split"]] + task_cfg["val_split"]
    load_splits = ["minval", "revqa_bt", "revqa", "val", "test", "train"]

    for split in load_splits:
        dataset = VQAClassificationDataset(
            split=split,
            image_features_reader=trainval_features if "test" not in split else test_features,
            tokenizer=tokenizer,
            extra_args=task_cfg
        )
        # specify the type of samplers
        if "train" in split:
            if registry.alt_train:
                samplers = ["scl", "ce"]
            else:
                samplers = ["ce"]
        else:
            samplers = ["none"]

        # build loaders for each sampler type
        for _sampler in samplers:
            sampler_tag = f"_{_sampler}" if _sampler != "none" else ""
            batch_size = task_cfg["batch_size"] if _sampler == "scl" else task_cfg["batch_size"] * 2

            # build the sampler
            if _sampler == "ce":
                sampler = RandomSampler(dataset)
            elif _sampler == "scl":
                sampler = ContrastiveSampler(
                    dataset,
                    task_cfg,
                    split=split
                )
            else:
                sampler = None

            split_tag = "train" if "train" in split else split
            dataloaders[f"{split_tag}" + sampler_tag] = DataLoader(
                dataset,
                sampler=sampler,
                batch_size= batch_size,
                num_workers=registry.workers,
                pin_memory=True,
                drop_last=True if split == "train" else False
            )

    return dataloaders


def compute_score_with_logits(logits, labels, device):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size())

    if device.type != "cpu":
        one_hots = one_hots.cuda()

    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores
