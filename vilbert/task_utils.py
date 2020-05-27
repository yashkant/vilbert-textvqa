# Copyright (c) Facebook, Inc. and its affiliates.

import logging
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from bisect import bisect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers.optimization import (
    AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    LambdaLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from tools.registry import registry
from vilbert.datasets import DatasetMapTrain
from vilbert.datasets.textvqa_metrics import TextVQAAccuracy, STVQAAccuracy
from vilbert.optimization import RAdam

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


class M4CandSpatialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])
        self.print_count = 0

    def forward(self, batch_dict):
        self.print_count += 1

        # textvqa loss
        scores = batch_dict["textvqa_scores"]
        loss_mask = batch_dict["train_loss_mask"]
        targets = batch_dict["targets"]
        assert scores.dim() == 3 and loss_mask.dim() == 2


        losses = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
        losses *= loss_mask.unsqueeze(-1)
        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count

        # spatial loss
        assert "spatial_scores" in batch_dict
        assert "spatial_loss_weight" in registry
        assert "textvqa_loss_weight" in registry
        spa_loss_mask = batch_dict["spatial_loss_mask"].float()
        # Todo: When all the relations are 0, argmax will return an arbitrary number between 0-11 (check it is masked)
        spa_targets = batch_dict["spatial_adj_matrix"].argmax(dim=-1).view(-1)
        spa_scores = batch_dict["spatial_scores"].view(-1, 12)
        spa_losses = F.cross_entropy(spa_scores, spa_targets, reduction="none")
        spa_losses = spa_losses.view_as(spa_loss_mask)
        spa_losses *= spa_loss_mask
        spa_count = torch.max(torch.sum(spa_loss_mask), self.one.to(spa_losses.device))
        spa_loss = torch.sum(spa_losses) / spa_count
        total_loss = loss*(registry["textvqa_loss_weight"]) + spa_loss*(registry["spatial_loss_weight"])

        if self.print_count % 20 == 0:
            round_print = lambda x: round(float(x), 4)
            logger.info(f"Spatial Loss: {round_print(spa_loss)}, "
                        f"TextVQA Loss: {round_print(loss)}, "
                        f"Spatial Loss weight: {round_print(registry['spatial_loss_weight'])}, "
                        f"TextVQA Loss weight: {round_print(registry['textvqa_loss_weight'])}, "
                        f"Total Loss: {round_print(total_loss)}")

        return total_loss


def clip_gradients(model, max_grad_l2_norm, clip_norm_mode):
    # TODO: Fix question model retrieval
    # Todo: Add tensorboard logger

    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2_norm)
            # import pdb
            # pdb.set_trace()
            # print("Grad norm:", norm)
            # writer.add_scalars({"grad_norm": norm}, i_iter)

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


def get_optim_scheduler(args,
                        config,
                        optimizer_grouped_parameters,
                        num_train_optimization_steps,
                        base_lr,
                        median_num_iter,
                        no_warmup=False
                        ):
    optim_config = config["TASK19"]["optim"] if args.optim is None else args.optim
    scheduler_config = config["TASK19"]["lr_scheduler"] if args.lr_scheduler is None else args.lr_scheduler

    if optim_config == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr, correct_bias=False)
    elif optim_config == "RAdam":
        optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)
    elif optim_config == "Adam":
        optimizer = Adam(optimizer_grouped_parameters, lr=base_lr)
    else:
        raise ValueError

    warmpu_steps = args.warmup_proportion * num_train_optimization_steps
    lr_reduce_list = np.array(config["TASK19"].get("lr_decay_steps", [-1]))

    if not no_warmup:
        if scheduler_config == "warmup_linear":
            warmup_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmpu_steps,
                                                    t_total=num_train_optimization_steps)
        elif scheduler_config == "pythia_warmup_decay":
            warmup_iters = config["TASK19"].get("warmup_iters", 1000)
            lr_decay_iters = config["TASK19"].get("lr_decay_iters", [14000, 19000])
            warmup_factor = config["TASK19"].get("warmup_factor", 0.1)
            # total_iters = int(np.ceil(34602/config["TASK19"]["batch_size"])*num_train_optimization_steps)

            def pythia_lr_update(_iter):
                if _iter <= warmup_iters:
                    alpha = float(_iter) / float(warmup_iters)
                    return warmup_factor * (1.0 - alpha) + alpha
                else:
                    idx = bisect(lr_decay_iters, _iter)
                    return pow(config["TASK19"].get("lr_decay", 0.2), idx)

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=pythia_lr_update)
            warmpu_steps = -1
        else:
            warmup_scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmpu_steps)
        logger.info(f"Warmup Scheduler: {str(warmup_scheduler)}")
    else:
        warmup_scheduler = None
        logger.info(f"Not using Warmup Scheduler")

    if scheduler_config == "automatic":
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=1, cooldown=1, threshold=0.001
        )
    elif scheduler_config == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=median_num_iter * args.num_train_epochs
        )
    elif scheduler_config == "cosine_warm":
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=median_num_iter * args.num_train_epochs
        )
    elif scheduler_config == "mannul":
        def lr_lambda_fun(epoch):
            return pow(config["TASK19"].get("lr_decay", 0.2), np.sum(lr_reduce_list <= epoch))

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fun)
    else:
        logger.info(f"Didn't recognize lr_scheduler: {scheduler_config}")
        lr_scheduler = None

    logger.info(f"LR Scheduler: {str(lr_scheduler)}")
    return optimizer, warmup_scheduler, lr_scheduler, scheduler_config, warmpu_steps


LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
    "TextVQALoss": M4CDecodingBCEWithMaskLoss(),
    "TextVQAandSpatialLoss": M4CandSpatialLoss(),
}

MetricsMap = {
    "TextVQA": TextVQAAccuracy(),
    "STVQA": STVQAAccuracy(),
}


def ForwardModelsVal(args,
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
        if isinstance(value, dict):
            for k,v in value.items():
                batch_dict[key][k] = v.cuda(device=device, non_blocking=True)

    question = batch_dict["question_indices"]
    batch_size = len(batch_dict["question_id"])
    batch_dict["task_tokens"] = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    results_dict = model(batch_dict)
    batch_dict.update(results_dict)

    # TODO: Fix this ugly hack!
    if registry.get("is_running_validation", False):
        return None, None, None

    if task_cfg[task_id]["loss"] == "TextVQAandSpatialLoss":
        loss = task_losses[task_id](batch_dict)
    else:
        loss = task_losses[task_id](batch_dict["textvqa_scores"], batch_dict["targets"], batch_dict["train_loss_mask"])

    if "metric" in task_cfg[task_id]:
        textvqa_metric = MetricsMap[task_cfg[task_id]["metric"]]
    else:
        textvqa_metric = MetricsMap["TextVQA"]

    batch_acc, batch_scores = textvqa_metric.calculate(batch_dict, batch_dict["textvqa_scores"])

    if return_batch:
        return float(loss), float(batch_acc), batch_size, batch_dict

    return float(loss), float(batch_acc), batch_size


def ForwardModelsTrain(
        args,
        task_cfg,
        device,
        task_id,
        task_count,
        task_iter_train,
        task_dataloader_train,
        model,
        task_losses,
):

    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    batch_dict = task_iter_train[task_id].next()

    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor):
            batch_dict[key] = value.cuda(device=device, non_blocking=True)
        if isinstance(value, dict):
            for k,v in value.items():
                batch_dict[key][k] = v.cuda(device=device, non_blocking=True)

    question = batch_dict["question_indices"]
    batch_dict["task_tokens"] = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    results_dict = model(batch_dict)
    batch_dict.update(results_dict)
    if task_cfg[task_id]["loss"] == "TextVQAandSpatialLoss":
        loss = task_losses[task_id](batch_dict)
    else:
        loss = task_losses[task_id](batch_dict["textvqa_scores"], batch_dict["targets"], batch_dict["train_loss_mask"])

    if "metric" in task_cfg[task_id]:
        textvqa_metric = MetricsMap[task_cfg[task_id]["metric"]]
    else:
        textvqa_metric = MetricsMap["TextVQA"]
    batch_acc, batch_scores = textvqa_metric.calculate(batch_dict, batch_dict["textvqa_scores"])

    return loss, batch_acc


def LoadLosses(args, task_cfg, task_ids):
    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id
        model_type = task_cfg[task]["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]["loss"]]

    return losses


def LoadDatasets(args, task_cfg, ids, split="trainval", only_val=False, test_val_bs=32, test_val_workers=8):
    if "roberta" in args.bert_model:
        tokenizer = RobertaTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )


    task_datasets_train = {}
    task_datasets_val = {}
    task_datasets_test = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_dataloader_test = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    task_id = "19"
    task = "TASK" + task_id
    task_ids.append(task)
    batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps
    num_workers = task_cfg[task].get("num_workers", 0)
    # if args.local_rank != -1:
    #     batch_size = int(batch_size / dist.get_world_size())
    #     num_workers = int(num_workers / dist.get_world_size())

    if "use_datasets" not in task_cfg[task]:
        task_cfg[task]['use_datasets'] = ["textvqa"]
        logger.info("Did not find `use_datasets` key in task configuration, generating it!")


    key_map = {
        "textvqa": "TextVQA",
        "rev_textvqa": "RevTextVQA",
        "stvqa": "STVQA",
        "ocrvqa": "OCRVQA"
    }


    if not only_val:

        logger.info(
            f"Loading Train Dataset(s) {task_cfg[task]['use_datasets']}  with batch size {batch_size}"
        )

        train_datasets = []
        for entry in task_cfg[task]["use_datasets"]:
            dataset = DatasetMapTrain[key_map[entry]](
                split=task_cfg[task]["train_split"],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
                extra_args=task_cfg[task]
            )
            train_datasets.append(dataset)

        task_datasets_train["separate_datasets"] = train_datasets
        task_datasets_train[task] = ConcatDataset(train_datasets)


    if "val_on" not in task_cfg[task]:
        task_cfg[task]['val_on'] = ["textvqa"]
        logger.info("Did not find `val_on` key in task configuration, generating it!")

    assert len(task_cfg[task]['val_on']) == 1
    logger.info(
        f"Loading Val Dataset {task_cfg[task]['val_on']}  with batch size {batch_size}"
    )
    val_task_name = key_map[task_cfg[task]['val_on'][0]]
    task_datasets_val[task] = DatasetMapTrain[val_task_name](
        split=task_cfg[task]["val_split"],
        tokenizer=tokenizer,
        bert_model=args.bert_model,
        padding_index=0,
        max_seq_length=task_cfg[task]["max_seq_length"],
        max_region_num=task_cfg[task]["max_region_num"],
        extra_args=task_cfg[task]
    )

    if "test" in split:
        logger.info(
            f"Loading Test Dataset(s) {task_cfg[task]['val_on']}  with batch size {batch_size}"
        )
        task_datasets_test[task] = DatasetMapTrain[val_task_name](
            split="test",
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            extra_args=task_cfg[task]
        )

    # Make sure we are using correct-vocabs
    if val_task_name == "TextVQA":
        assert task_cfg[task]["vocab_type"] != "5k_stvqa"
    elif val_task_name == "RevTextVQA":
        assert task_cfg[task]["vocab_type"] != "5k_stvqa"
    elif val_task_name == "STVQA":
        assert task_cfg[task]["vocab_type"] == "5k_stvqa"
    elif val_task_name == "OCRVQA":
        assert task_cfg[task]["vocab_type"] == "ocrvqa"
    else:
        raise ValueError

    task_num_iters[task] = 0
    task_batch_size[task] = 0
    if "train" in split and not only_val:
        if args.local_rank == -1:
            train_sampler = RandomSampler(task_datasets_train[task])
        else:
            # TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(task_datasets_train[task])

        task_dataloader_train[task] = DataLoader(
            task_datasets_train[task],
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_train[task])
        task_batch_size[task] = batch_size

    if "val" in split:
        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=test_val_bs,
            num_workers=test_val_workers,
            pin_memory=True,
        )

    if "test" in split:
        task_dataloader_test[task] = DataLoader(
            task_datasets_test[task],
            shuffle=False,
            batch_size=test_val_bs,
            num_workers=test_val_workers,
            pin_memory=True,
        )
        return (
            task_batch_size,
            task_num_iters,
            task_ids,
            task_datasets_train,
            task_datasets_val,
            task_datasets_test,
            task_dataloader_train,
            task_dataloader_val,
            task_dataloader_test,
        )

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )




def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


