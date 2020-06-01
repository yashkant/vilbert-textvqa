# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from bisect import bisect
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, RandomSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from vilbert.datasets import DatasetMapTrain
from vilbert.datasets.textvqa_metrics import TextVQAAccuracy, STVQAAccuracy
from torch.optim.lr_scheduler import (
    LambdaLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from pytorch_transformers.optimization import (
    AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)
from tools.registry import registry

logger = logging.getLogger(__name__)


class M4CDecodingBCEWithMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])
        self.debug_count = 0

    def forward(self, scores, targets, loss_mask):
        # self.debug_count += 1
        assert scores.dim() == 3 and loss_mask.dim() == 2
        losses = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
        losses *= loss_mask.unsqueeze(-1)
        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        return loss


def clip_gradients(model, max_grad_l2_norm, clip_norm_mode):
    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2_norm)
        elif clip_norm_mode == "question":
            question_embedding = model.module.question_embedding_module
            norm = nn.utils.clip_grad_norm(
                question_embedding.parameters(), max_grad_l2_norm
            )

            # writer.add_scalars({"question_grad_norm": norm}, i_iter)
        else:
            raise NotImplementedError(
                "Clip norm mode %s not implemented" % clip_norm_mode
            )


def get_optim_scheduler(task_cfg, optimizer_grouped_parameters, base_lr):
    optimizer = Adam(optimizer_grouped_parameters, lr=base_lr)
    warmup_iters = task_cfg.get("warmup_iters", 1000)
    lr_decay_iters = task_cfg.get("lr_decay_iters", [14000, 19000])
    warmup_factor = task_cfg.get("warmup_factor", 0.1)
    def pythia_lr_update(_iter):
        if _iter <= warmup_iters:
            alpha = float(_iter) / float(warmup_iters)
            return warmup_factor * (1.0 - alpha) + alpha
        else:
            idx = bisect(lr_decay_iters, _iter)
            return pow(task_cfg.get("lr_decay", 0.2), idx)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=pythia_lr_update)
    return optimizer, warmup_scheduler


LossMap = {
    "TextVQA": M4CDecodingBCEWithMaskLoss(),
}

MetricsMap = {
    "TextVQA": TextVQAAccuracy(),
    "STVQA": STVQAAccuracy(),
}


def ForwardModelsVal(task_cfg,
                     batch_dict,
                     model,
                     return_batch=False):
    
    results_dict = model(batch_dict)
    batch_dict.update(results_dict)

    # TODO: Fix this ugly hack!
    if registry.get("is_running_validation", False):
        return None, None, None

    loss = LossMap[task_cfg["loss"]]
    textvqa_metric = MetricsMap[task_cfg["metric"]]
    loss = loss(batch_dict["textvqa_scores"], batch_dict["targets"], batch_dict["train_loss_mask"])
    batch_acc, batch_scores = textvqa_metric.calculate(batch_dict, batch_dict["textvqa_scores"])
    if return_batch:
        return float(loss), float(batch_acc), len(batch_dict["question_id"]), batch_dict

    return float(loss), float(batch_acc), len(batch_dict["question_id"])


def ForwardModelsTrain(task_cfg, model, batch_dict):
    loss = LossMap[task_cfg["loss"]]
    textvqa_metric = MetricsMap[task_cfg["metric"]]
    results_dict = model(batch_dict)
    batch_dict.update(results_dict)
    loss = loss(batch_dict["textvqa_scores"], batch_dict["targets"], batch_dict["train_loss_mask"])
    batch_acc, batch_scores = textvqa_metric.calculate(batch_dict, batch_dict["textvqa_scores"])
    return loss, batch_acc


def LoadLosses(args, task_cfg, task_ids):
    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id
        model_type = task_cfg["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg["loss"]]

    return losses


def get_loader(args, task_cfg, tokenizer, split):

    dataset_names = task_cfg[f"{split}_on"]
    assert isinstance(dataset_names, list)

    datasets = []
    for dset in dataset_names:
        _dataset = DatasetMapTrain[dset](
            split=split,
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            padding_index=0,
            task_cfg=task_cfg
        )
        datasets.append(_dataset)

    if len(datasets) > 1:
        dataset_instance = ConcatDataset(datasets)
    else:
        dataset_instance = datasets[0]

    random_sampler = RandomSampler(dataset_instance)
    loader = DataLoader(
            dataset_instance,
            sampler=random_sampler if split == "train" else None,
            batch_size=task_cfg["batch_size"],
            num_workers=task_cfg["num_workers"],
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
    return loader


def load_datasets(args, task_cfg, splits):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    loaders = {}
    for split in splits:
        loaders[split] = get_loader(args, task_cfg, tokenizer, split)
    return loaders


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores
