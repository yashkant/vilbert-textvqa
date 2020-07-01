# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from bisect import bisect
from collections import defaultdict
from io import open
import json
import os
import sys
from itertools import combinations
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from vilbert.samplers import RandomSampler, NegativeSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from vilbert.datasets import DatasetMapTrain, DatasetMapEval
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import pdb
from vilbert.batch_utils import build_scl_mask
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

from vilbert.losses import NTXentLoss, SCLLoss, SupConLoss
from vilbert.optimization import RAdam
from tools.registry import registry
from vilbert.utils import debug_sampler

import logging
logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
    # "NTXentLoss": NTXentLoss(registry.batch_size),
    "SCLLoss": SupConLoss(temperature=0.5, formulation=registry.scl_formulation), # using the default parameter setting
}

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
    # moved to model-file
    # weight_decay = config["TASK19"].get("weight_decay", 0.0)

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

def to_device(batch_dict, device):

    if device.type == "cpu":
        return

    for batch in batch_dict:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda(device=device, non_blocking=True)



def ForwardModelsVal(args,
                     task_cfg,
                     device,
                     task_id,
                     batch_dict,
                     model,
                     task_losses,
                     revqa_eval=False,
                     return_batch=False):

    if not isinstance(batch_dict, tuple) and not isinstance(batch_dict, list):
        batch_dict = [batch_dict]
    # else:
    #     build_scl_mask(batch_dict)

    # Sanity check negatives w/ negative sampler
    # if task_cfg["TASK19"].get("contrastive", None) in ["simclr", "better"] and not task_cfg["TASK19"]["debug"]\
    #         and task_cfg["TASK19"].get("val_neg_sampler", True):
    #     for batch in batch_dict:
    #         rephrasing_of = []
    #         # iterate over question-ids
    #         for question_id in batch["question_id"].tolist():
    #             try:
    #                 assert registry.question_rephrase_dict_val[question_id] not in rephrasing_of
    #                 rephrasing_of.append(registry.question_rephrase_dict_val[question_id])
    #             except:
    #                 import pdb
    #                 pdb.set_trace()

    # send to device
    to_device(batch_dict, device)

    for batch in batch_dict:
        question = batch["question_indices"]
        # batch["task_tokens"] = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
        batch_size = len(question)
        results_dict = model(batch)
        batch.update(results_dict)

    # if task_cfg[task_id]["type"] == "ContrastiveProjection":
    #     loss, batch_score = task_losses[task_id](batch_dict)
    #
    # if len(batch_dict) == 1:
    #     batch_dict = batch_dict[0]
    #
    # # for different task, we use different output to calculate the loss.
    # if task_cfg[task_id]["type"] == "VL-classifier":
    #     loss = task_losses[task_id](batch_dict["vil_prediction"], batch_dict["target"])
    #     loss = loss.mean() * batch_dict["target"].size(1)
    #     batch_scores = compute_score_with_logits(batch_dict["vil_prediction"], batch_dict["target"], device)
    #     batch_score = batch_scores.sum() / float(batch_size)

        # # calculate consistency scores
        # if registry.get("revqa_eval", False):
        #     # fill the scores for each question into the batch-dict
        #     batch_dict["vqa_scores"] = batch_scores.sum(dim=-1).tolist()
        #
        #     # add vqa-scores to defaultdict(list) for each bin
        #     for idx, qid in enumerate(batch_dict["question_id"].tolist()):
        #         registry.revqa_bins[registry["question_rephrase_dict_val"][qid]].append(batch_dict["vqa_scores"][idx])


    # elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
    #     loss = task_losses[task_id](batch_dict["vil_prediction_gqa"], batch_dict["target"])
    #     loss = loss.mean() * batch_dict["target"].size(1)
    #     batch_score = compute_score_with_logits(
    #         batch_dict["vil_prediction_gqa"], batch_dict["target"], device
    #     ).sum() / float(batch_size)
    #
    # elif task_cfg[task_id]["type"] == "VL-classifier-only-ce":
    #     loss, batch_score = add_ce_loss(batch_dict[0], device, val_run=True)

    # if registry.use_ce_loss:
    #     vl_loss, batch_score = add_ce_loss(batch_dict[0], device, val_run=True)
    #     # don't care about the scl-loss
    #     loss = vl_loss


    # only report CE loss on validation-set
    loss, batch_score = add_ce_loss(batch_dict[0], device, val_run=True, revqa_eval=revqa_eval)


    if registry.get("eval_only", False):
        return batch_dict

    del results_dict
    del batch_dict

    return float(loss), float(batch_score), batch_size


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
    # given the current task, decided whether to forward the model and forward with specific loss.
    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    start_time = time.time()
    task_count[task_id] += 1
    batch_dict = task_iter_train[task_id].next()
    batch_time = time.time()
    # print(f"Build Batch Time: {batch_time-start_time}")

    if not isinstance(batch_dict, tuple) and not isinstance(batch_dict, list):
        batch_dict = [batch_dict]
    # else:
    #     build_scl_mask(batch_dict)

    # Sanity check negatives
    if task_cfg["TASK19"].get("contrastive", None) in ["simclr", "better"] and not task_cfg["TASK19"]["debug"]:
        for batch in batch_dict:
            rephrasing_of = []
            # iterate over question-ids
            for question_id in batch["question_id"].tolist():
                try:
                    assert registry.question_rephrase_dict_train[question_id] not in rephrasing_of
                    rephrasing_of.append(registry.question_rephrase_dict_train[question_id])
                except:
                    import pdb
                    pdb.set_trace()

    # send to device
    to_device(batch_dict, device)
    question = batch_dict[0]["question_indices"]
    # batch["task_tokens"] = question.new().resizeA_(question.size(0), 1).fill_(int(task_id[4:]))
    batch_size = len(question)

    # if not registry.squint_loss:
    for batch in batch_dict:
        results_dict = model(batch)
        batch.update(results_dict)
    # else:
    #     squint_batches = [{}, {}]
    #     assert len(batch_dict) == 2
    #     hbs = int(len(batch_dict[0]["input_imgs"])/2)
    #     for key in batch_dict[0].keys():
    #         squint_batches[0][key] = torch.cat([batch_dict[0][key][:hbs], batch_dict[1][key][:hbs]])
    #         squint_batches[1][key] = torch.cat([batch_dict[0][key][hbs:], batch_dict[1][key][hbs:]])
    #
    #     for batch in squint_batches:
    #         results_dict = model(batch)
    #         batch.update(results_dict)

    forward_time = time.time()
    # print(f"Forward Batch Time: {forward_time-batch_time}")

    if task_cfg[task_id]["type"] == "ContrastiveProjection":
        loss, batch_score = task_losses[task_id](batch_dict)

    if len(batch_dict) == 1:
        batch_dict = batch_dict[0]

    # for different task, we use different output to calculate the loss.
    if task_cfg[task_id]["type"] == "VL-classifier":
        # loss = task_losses[task_id](batch_dict["vil_prediction"], batch_dict["target"])
        # loss = loss.mean() * batch_dict["target"].size(1)
        # batch_score = compute_score_with_logits(batch_dict["vil_prediction"], batch_dict["target"]).sum() / float(
        #     batch_size
        # )
        loss, batch_score = add_ce_loss(batch_dict, device)

    elif task_cfg[task_id]["type"] == "VL-classifier-only-ce":
        loss, batch_score = add_ce_loss(batch_dict, device)

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = task_losses[task_id](batch_dict["vil_prediction_gqa"], batch_dict["target"])
        loss = loss.mean() * batch_dict["target"].size(1)
        batch_score = compute_score_with_logits(
            batch_dict["vil_prediction_gqa"], batch_dict["target"], device
        ).sum() / float(batch_size)

    losses = []
    if registry.use_ce_loss:
        vl_loss, batch_score = add_ce_loss(batch_dict, device)
        losses.append(loss)
        losses.append(vl_loss)
        assert registry.scl_coeff > 0
        loss = loss*registry.scl_coeff + vl_loss

    if registry.squint_loss:
        from vilbert.losses import MSELoss
        squint_loss = MSELoss(batch_dict)
        loss += squint_loss

    loss_time = time.time()
    # print(f"Loss Time: {loss_time - forward_time}")


    del results_dict
    del batch_dict
    # # we are storing tensors in batch-dict, refs to which cause OOM (I think)


    return loss, float(batch_score), losses


# todo: replace this in vl-classifier if-else
def add_ce_loss(batch_dict, device, val_run=False, revqa_eval=False, split="re_total"):
    if len(batch_dict) == 2 and not val_run:
        # train time
        if not registry.ce_half:
            vil_preds = torch.cat([batch_dict[0]["vil_prediction"], batch_dict[1]["vil_prediction"]], dim=0)
            vil_targets = torch.cat([batch_dict[0]["target"], batch_dict[1]["target"]], dim=0)
        else:
            # randomly pick the half batch from SCL
            idx = np.random.randint(2)
            vil_preds = batch_dict[idx]["vil_prediction"]
            vil_targets = batch_dict[idx]["target"]

    else:
        # validation time
        vil_preds = batch_dict["vil_prediction"]
        vil_targets = batch_dict["target"]


    vl_loss = LossMap["BCEWithLogitLoss"](vil_preds, vil_targets)
    vl_loss = vl_loss.mean() * vil_targets.size(1)
    batch_scores = compute_score_with_logits(vil_preds, vil_targets, device)
    batch_score = batch_scores.sum() / len(vil_preds)


    # calculate consistency scores during validation run!
    if revqa_eval:
        # fill the scores for each question into the batch-dict
        batch_dict["vqa_scores"] = batch_scores.sum(dim=-1).tolist()

        # add vqa-scores to defaultdict(list) for each bin
        for idx, qid in enumerate(batch_dict["question_id"].tolist()):
            registry.revqa_bins[registry[f"question_rephrase_dict_{split}"][qid]].append(batch_dict["vqa_scores"][idx])

    return vl_loss, batch_score

def LoadLosses(args, task_cfg, task_ids):

    losses = {}
    task_types = []
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id
        model_type = task_cfg[task]["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]["loss"]]

    return losses


def LoadDatasets(args, task_cfg, ids, split="trainval"):

    if "roberta" in args.bert_model:
        tokenizer = RobertaTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
        if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )
    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_name = task_cfg[task]["name"]
        task_ids.append(task)
        batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps

        assert args.local_rank == -1
        # if args.local_rank != -1:
        #     batch_size = int(batch_size / dist.get_world_size())
        #     num_workers = int(num_workers / dist.get_world_size())

        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        task_datasets_train[task] = None
        if "train" in split:
            task_datasets_train[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
                split=task_cfg[task]["train_split"],
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ],
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
                extra_args=task_cfg["TASK19"]
            )

        task_datasets_val[task] = None
        if "val" in split:
            task_datasets_val[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
                split=task_cfg[task]["val_split"],
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ],
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
                extra_args=task_cfg["TASK19"]
            )

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        if "train" in split:
            if task_cfg["TASK19"].get("contrastive", None) in ["simclr", "better"]:
                logger.info("Using Negative Train Sampler")
                train_sampler = NegativeSampler(
                    task_datasets_train[task],
                    batch_size,
                    task_cfg,
                    args,
                    split=task_cfg[task]["train_split"]
                )
            elif args.local_rank == -1:
                train_sampler = RandomSampler(task_datasets_train[task])
            else:
                # TODO: check if this works with current data generator from disk that relies on next(file)
                # (it doesn't return item back by index)
                train_sampler = DistributedSampler(task_datasets_train[task])

            task_dataloader_train[task] = DataLoader(
                task_datasets_train[task],
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=registry.train_workers,
                pin_memory=True,
                drop_last=True
            )

            task_num_iters[task] = len(task_dataloader_train[task])
            task_batch_size[task] = batch_size

        if "val" in split:
            if task_cfg["TASK19"].get("contrastive", None) in ["simclr", "better"] and task_cfg["TASK19"].get("val_neg_sampler", True):
                logger.info("Using Negative Validation Sampler")
                val_sampler = NegativeSampler(
                    task_datasets_val[task],
                    batch_size,
                    task_cfg,
                    args,
                    split=task_cfg[task]["val_split"]
                )
            else:
                logger.info("Using Simple Validation Sampler")
                val_sampler=None

            task_dataloader_val[task] = DataLoader(
                task_datasets_val[task],
                shuffle=False,
                sampler=val_sampler,
                batch_size=registry.val_batch_size,
                num_workers=registry.val_workers,
                pin_memory=True,
                drop_last=registry.val_drop_last
            )

        # load the ReVQA val-split!
        if registry.revqa_eval:
            logger.info("Loding ReVQA Dataset!")
            task_datasets_val["revqa"] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
                split="re_total",
                image_features_reader=task_feature_reader1[
                    task_cfg[task]["features_h5path1"]
                ],
                gt_image_features_reader=task_feature_reader2[
                    task_cfg[task]["features_h5path2"]
                ],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
                extra_args=task_cfg["TASK19"]
            )

            task_dataloader_val["revqa"] = DataLoader(
                task_datasets_val["revqa"],
                shuffle=False,
                sampler=None,
                batch_size=registry.val_batch_size,
                num_workers=registry.val_workers,
                pin_memory=True,
                drop_last=False
            )

    # debug_sampler(train_sampler)
    # debug_sampler(val_sampler)

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )


def compute_score_with_logits(logits, labels, device):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size())

    if device.type != "cpu":
        one_hots = one_hots.cuda()

    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


