import logging
from bisect import bisect

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    LambdaLR,
)
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

from tools.registry import registry
from vilbert.datasets import DatasetMapTrain
from vilbert.datasets.metrics import TextVQAAccuracy, STVQAAccuracy

logger = logging.getLogger(__name__)


class M4CDecodingBCEWithMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])

    def forward(self, scores, targets, loss_mask):
        assert scores.dim() == 3 and loss_mask.dim() == 2
        losses = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
        losses *= loss_mask.unsqueeze(-1)
        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        return loss


def clip_gradients(model, max_grad_l2_norm):
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2_norm)


def get_optim_scheduler(
    task_cfg,
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


LossMap = {
    "TextVQALoss": M4CDecodingBCEWithMaskLoss(),
}

MetricsMap = {
    "TextVQA": TextVQAAccuracy(),
    "STVQA": STVQAAccuracy(),
}


def get_batch(dataloaders, key):
    ikey = f"{key}_iter"
    load_epoch = ikey not in dataloaders

    # add iterator
    if not load_epoch:
        batch_dict = next(dataloaders[ikey], None)

        # iterator exhausted
        if batch_dict is None:
            load_epoch = True

    # reload iterator
    if load_epoch:
        dataloaders[ikey] = iter(dataloaders[key])
        batch_dict = next(dataloaders[ikey], None)
        assert batch_dict is not None

    return batch_dict


def forward_val(args,
                task_cfg,
                device,
                task_id,
                batch_dict,
                model,
                task_losses,
                return_batch=False):
    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor):
            batch_dict[key] = value.cuda(device=device, non_blocking=True)

    question = batch_dict["question_indices"]
    batch_size = len(batch_dict["question_id"])
    batch_dict["task_tokens"] = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    results_dict = model(batch_dict)
    batch_dict.update(results_dict)

    # TODO: Fix this ugly hack!
    if registry.get("is_running_validation", False):
        return None, None, None

    if task_cfg["loss"] == "TextVQAandSpatialLoss":
        loss = task_losses[task_id](batch_dict)
    else:
        loss = task_losses[task_id](batch_dict["textvqa_scores"], batch_dict["targets"], batch_dict["train_loss_mask"])

    if "metric" in task_cfg:
        textvqa_metric = MetricsMap[task_cfg["metric"]]
    else:
        textvqa_metric = MetricsMap["TextVQA"]

    batch_acc, batch_scores = textvqa_metric.calculate(batch_dict, batch_dict["textvqa_scores"])

    if return_batch:
        return float(loss), float(batch_acc), batch_size, batch_dict

    return float(loss), float(batch_acc), batch_size


def forward_train(
        dataloaders,
        task_cfg,
        device,
        task_id,
        model,
):
    batch_dict = get_batch(dataloaders, "train")
    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor):
            batch_dict[key] = value.cuda(device=device, non_blocking=True)

    question = batch_dict["question_indices"]
    batch_dict["task_tokens"] = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    results_dict = model(batch_dict)
    batch_dict.update(results_dict)
    loss = LossMap["TextVQALoss"](batch_dict["textvqa_scores"], batch_dict["targets"], batch_dict["train_loss_mask"])
    textvqa_metric = MetricsMap[task_cfg["metric"]]
    batch_acc, batch_scores = textvqa_metric.calculate(batch_dict, batch_dict["textvqa_scores"])

    return loss, batch_acc


def load_losses(task_cfg, task_ids):
    losses = {}
    task_types = []
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id
        model_type = task_cfg["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg["loss"]]

    return losses


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def get_loader(task_cfg, tokenizer, split):

    dataset_names = task_cfg[f"{split}_on"]
    assert isinstance(dataset_names, list)

    datasets = []
    for dset in dataset_names:
        _dataset = DatasetMapTrain[dset](
            split=split,
            tokenizer=tokenizer,
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


def load_datasets(task_cfg, splits):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    loaders = {}
    for split in splits:
        loaders[split] = get_loader(task_cfg, tokenizer, split)
    return loaders
