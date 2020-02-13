# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from bisect import bisect
from io import open
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from vilbert.datasets import DatasetMapTrain, DatasetMapEval
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
from vilbert.datasets.textvqa_metrics import TextVQAAccuracy
import pdb
from tools.registry import registry
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
        spa_loss_mask = batch_dict["spatial_loss_mask"].float()
        # Todo: When all the relations are 0, argmax will return an arbitrary number between 0-11 (check it is masked)
        spa_targets = batch_dict["spatial_adj_matrix"].argmax(dim=-1).view(-1)
        spa_scores = batch_dict["spatial_scores"].view(-1, 12)
        spa_losses = F.cross_entropy(spa_scores, spa_targets, reduction="none")
        spa_losses = spa_losses.view_as(spa_loss_mask)
        spa_losses *= spa_loss_mask
        spa_count = torch.max(torch.sum(spa_loss_mask), self.one.to(spa_losses.device))
        spa_loss = torch.sum(spa_losses) / spa_count
        total_loss = loss + spa_loss

        if self.print_count % 20 == 0:
            round_print = lambda x: round(float(x), 4)
            logger.info(f"Spatial Loss: {round_print(spa_loss)}, "
                        f"TextVQA Loss: {round_print(loss)}, "
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
}


def ForwardModelsVal(args, task_cfg, device, task_id, batch_dict, model, task_losses):
    # batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    # import pdb
    # pdb.set_trace()

    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor):
            batch_dict[key] = value.cuda(device=device, non_blocking=True)
    # if task_id == "TASK4" or task_id == "TASK17":
    #     features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (
    #         batch
    #     )
    # else:
    #     features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, image_id, question_id = (
    #         batch
    #     )
    #
    # batch_size = features.size(0)
    # if task_cfg[task_id]["process"] in ["expand"]:
    #     max_num_bbox = features.size(1)
    #     num_options = question.size(1)
    #     features = (
    #         features.unsqueeze(1)
    #         .expand(batch_size, num_options, max_num_bbox, 2048)
    #         .contiguous()
    #         .view(-1, max_num_bbox, 2048)
    #     )
    #     spatials = (
    #         spatials.unsqueeze(1)
    #         .expand(batch_size, num_options, max_num_bbox, 5)
    #         .contiguous()
    #         .view(-1, max_num_bbox, 5)
    #     )
    #     image_mask = (
    #         image_mask.unsqueeze(1)
    #         .expand(batch_size, num_options, max_num_bbox)
    #         .contiguous()
    #         .view(-1, max_num_bbox)
    #     )
    #     question = question.view(-1, question.size(2))
    #     input_mask = input_mask.view(-1, input_mask.size(2))
    #     segment_ids = segment_ids.view(-1, segment_ids.size(2))
    #     co_attention_mask = co_attention_mask.view(
    #         -1, co_attention_mask.size(2), co_attention_mask.size(3)
    #     )
    #
    # elif task_cfg[task_id]["process"] in ["retrieval"]:
    #     max_num_bbox = features.size(1)
    #     num_options = question.size(1)
    #     features = features.view(-1, features.size(2), features.size(3))
    #     spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
    #     image_mask = image_mask.view(-1, image_mask.size(2))
    #     question = question.view(-1, question.size(2))
    #     input_mask = input_mask.view(-1, input_mask.size(2))
    #     segment_ids = segment_ids.view(-1, segment_ids.size(2))
    #     co_attention_mask = co_attention_mask.view(
    #         -1, co_attention_mask.size(2), co_attention_mask.size(3)
    #     )
    #
    # elif task_cfg[task_id]["process"] in ["nlvr"]:
    #     batch_size = features.size(0)
    #     max_num_bbox = features.size(1)
    #     num_options = question.size(1)
    #     features = features.view(
    #         batch_size * 2, int(features.size(1) / 2), features.size(2)
    #     )
    #     spatials = spatials.view(
    #         batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
    #     )
    #     image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
    #     question = question.repeat(1, 2)
    #     question = question.view(batch_size * 2, int(question.size(1) / 2))
    #     input_mask = input_mask.repeat(1, 2)
    #     input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
    #     segment_ids = segment_ids.repeat(1, 2)
    #     segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
    #     co_attention_mask = co_attention_mask.view(
    #         batch_size * 2,
    #         int(co_attention_mask.size(1) / 2),
    #         co_attention_mask.size(2),
    #     )

    # task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
    #
    # vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
    #     question,
    #     features,
    #     spatials,
    #     segment_ids,
    #     input_mask,
    #     image_mask,
    #     co_attention_mask,
    #     task_tokens,
    # )

    question = batch_dict["question_indices"]
    batch_size = len(batch_dict["question_id"])
    batch_dict["task_tokens"] = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    results_dict = model(batch_dict)
    batch_dict.update(results_dict)
    if task_cfg[task_id]["loss"] == "TextVQAandSpatialLoss":
        loss = task_losses[task_id](batch_dict)
    else:
        loss = task_losses[task_id](batch_dict["textvqa_scores"], batch_dict["targets"], batch_dict["train_loss_mask"])

    textvqa_metric = MetricsMap["TextVQA"]
    batch_acc, batch_scores = textvqa_metric.calculate(batch_dict, batch_dict["textvqa_scores"])

    # if task_cfg[task_id]["type"] == "VL-classifier":
    #     loss = task_losses[task_id](vil_prediction, target)
    #     loss = loss.mean() * target.size(1)
    #     batch_score = compute_score_with_logits(vil_prediction, target).sum()
    #
    # if task_cfg[task_id]["type"] == "VL-classifier-GQA":
    #     loss = task_losses[task_id](vil_prediction_gqa, target)
    #     loss = loss.mean() * target.size(1)
    #     batch_score = compute_score_with_logits(vil_prediction_gqa, target).sum()
    #
    # elif task_cfg[task_id]["type"] == "VL-logit":
    #     vil_logit = vil_logit.view(batch_size, num_options)
    #     loss = task_losses[task_id](vil_logit, target)
    #     _, preds = torch.max(vil_logit, 1)
    #     batch_score = (preds == target).sum()
    #
    # elif task_cfg[task_id]["type"] == "V-logit":
    #     loss = task_losses[task_id](vision_logit, target)
    #     loss = loss.mean() * target.size(1)
    #     _, select_idx = torch.max(vision_logit, dim=1)
    #     select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
    #     batch_score = torch.sum(select_target > 0.5).item()
    #
    # elif task_cfg[task_id]["type"] == "V-logit-mc":
    #     vision_logit = vision_logit[:, 101:]
    #     vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
    #     vision_logit = vision_logit.unsqueeze(2)
    #     loss = task_losses[task_id](vision_logit, target)
    #     loss = loss.mean() * target.size(1)
    #     _, preds = torch.max(vision_logit, dim=1)
    #     _, target = torch.max(target, dim=1)
    #     batch_score = (preds == target).sum()
    #
    # elif task_cfg[task_id]["type"] == "VL-binary-classifier":
    #     loss = task_losses[task_id](vil_binary_prediction, target)
    #     loss = loss.mean()
    #     batch_score = compute_score_with_logits(vil_binary_prediction, target).sum()
    #
    # elif task_cfg[task_id]["type"] == "VL-tri-classifier":
    #     loss = task_losses[task_id](vil_tri_prediction, target)
    #     loss = loss.mean()
    #     batch_score = compute_score_with_logits(vil_tri_prediction, target).sum()

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
    # given the current task, decided whether to forward the model and forward with specific loss.

    # import pdb
    # pdb.set_trace()

    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    batch_dict = task_iter_train[task_id].next()

    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor):
            batch_dict[key] = value.cuda(device=device, non_blocking=True)
    # batch_dict = tuple(t.cuda(device=device, non_blocking=True) for t in batch if isinstance(t, torch.Tensor))
    # if task_id == "TASK4" or task_id == "TASK17":
    #     features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (
    #         batch
    #     )
    # else:
    #     features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, image_id, question_id = (
    #         batch
    #     )

    # batch_size = batch_dict["pad_obj_features"].size(0)
    # target = batch_dict["targets"]
    # if task_cfg[task_id]["process"] in ["dialog"]:
    #     max_num_bbox = features.size(1)
    #     nround = question.size(1)
    #     num_options = question.size(2)
    #     rbatch_size = batch_size * nround
    #     question = question.view(rbatch_size, question.size(2), question.size(3))
    #     target = target.view(-1)
    #     input_mask = input_mask.view(
    #         rbatch_size, input_mask.size(2), input_mask.size(3)
    #     )
    #     segment_ids = segment_ids.view(
    #         rbatch_size, segment_ids.size(2), segment_ids.size(3)
    #     )
    #     co_attention_mask = co_attention_mask.view(
    #         rbatch_size,
    #         co_attention_mask.size(2),
    #         co_attention_mask.size(3),
    #         co_attention_mask.size(4),
    #     )
    #
    #     features = (
    #         features.unsqueeze(1)
    #         .unsqueeze(1)
    #         .expand(batch_size, nround, num_options, max_num_bbox, 2048)
    #         .contiguous()
    #         .view(-1, max_num_bbox, 2048)
    #     )
    #     spatials = (
    #         spatials.unsqueeze(1)
    #         .unsqueeze(1)
    #         .expand(batch_size, nround, num_options, max_num_bbox, 5)
    #         .contiguous()
    #         .view(-1, max_num_bbox, 5)
    #     )
    #     image_mask = (
    #         image_mask.unsqueeze(1)
    #         .expand(batch_size, nround, num_options, max_num_bbox)
    #         .contiguous()
    #         .view(-1, max_num_bbox)
    #     )
    #
    #     question = question.view(-1, question.size(2))
    #     input_mask = input_mask.view(-1, input_mask.size(2))
    #     segment_ids = segment_ids.view(-1, segment_ids.size(2))
    #     co_attention_mask = co_attention_mask.view(
    #         -1, co_attention_mask.size(2), co_attention_mask.size(3)
    #     )
    #     batch_size = rbatch_size
    #
    # elif task_cfg[task_id]["process"] in ["expand"]:
    #     max_num_bbox = features.size(1)
    #     num_options = question.size(1)
    #     features = (
    #         features.unsqueeze(1)
    #         .expand(batch_size, num_options, max_num_bbox, 2048)
    #         .contiguous()
    #         .view(-1, max_num_bbox, 2048)
    #     )
    #     spatials = (
    #         spatials.unsqueeze(1)
    #         .expand(batch_size, num_options, max_num_bbox, 5)
    #         .contiguous()
    #         .view(-1, max_num_bbox, 5)
    #     )
    #     image_mask = (
    #         image_mask.unsqueeze(1)
    #         .expand(batch_size, num_options, max_num_bbox)
    #         .contiguous()
    #         .view(-1, max_num_bbox)
    #     )
    #     question = question.view(-1, question.size(2))
    #     input_mask = input_mask.view(-1, input_mask.size(2))
    #     segment_ids = segment_ids.view(-1, segment_ids.size(2))
    #     co_attention_mask = co_attention_mask.view(
    #         -1, co_attention_mask.size(2), co_attention_mask.size(3)
    #     )
    #
    # elif task_cfg[task_id]["process"] in ["retrieval"]:
    #     max_num_bbox = features.size(1)
    #     num_options = question.size(1)
    #     features = features.view(-1, features.size(2), features.size(3))
    #     spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
    #     image_mask = image_mask.view(-1, image_mask.size(2))
    #     question = question.view(-1, question.size(2))
    #     input_mask = input_mask.view(-1, input_mask.size(2))
    #     segment_ids = segment_ids.view(-1, segment_ids.size(2))
    #     co_attention_mask = co_attention_mask.view(
    #         -1, co_attention_mask.size(2), co_attention_mask.size(3)
    #     )
    #
    # elif task_cfg[task_id]["process"] in ["nlvr"]:
    #     batch_size = features.size(0)
    #     max_num_bbox = features.size(1)
    #     num_options = question.size(1)
    #     features = features.view(
    #         batch_size * 2, int(features.size(1) / 2), features.size(2)
    #     )
    #     spatials = spatials.view(
    #         batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
    #     )
    #     image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
    #     question = question.repeat(1, 2)
    #     question = question.view(batch_size * 2, int(question.size(1) / 2))
    #     input_mask = input_mask.repeat(1, 2)
    #     input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
    #     segment_ids = segment_ids.repeat(1, 2)
    #     segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
    #     co_attention_mask = co_attention_mask.view(
    #         batch_size * 2,
    #         int(co_attention_mask.size(1) / 2),
    #         co_attention_mask.size(2),
    #     )
    question = batch_dict["question_indices"]
    batch_dict["task_tokens"] = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    results_dict = model(batch_dict)
    batch_dict.update(results_dict)
    if task_cfg[task_id]["loss"] == "TextVQAandSpatialLoss":
        loss = task_losses[task_id](batch_dict)
    else:
        loss = task_losses[task_id](batch_dict["textvqa_scores"], batch_dict["targets"], batch_dict["train_loss_mask"])

    textvqa_metric = MetricsMap["TextVQA"]
    batch_acc, batch_scores = textvqa_metric.calculate(batch_dict, batch_dict["textvqa_scores"])

    # # for different task, we use different output to calculate the loss.
    # if task_cfg[task_id]["type"] == "VL-classifier":
    #     loss = task_losses[task_id](vil_prediction, target)
    #     loss = loss.mean() * target.size(1)
    #     batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(
    #         batch_size
    #     )
    #
    # elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
    #     loss = task_losses[task_id](vil_prediction_gqa, target)
    #     loss = loss.mean() * target.size(1)
    #     batch_score = compute_score_with_logits(
    #         vil_prediction_gqa, target
    #     ).sum() / float(batch_size)
    #
    # elif task_cfg[task_id]["type"] == "VL-logit":
    #     vil_logit = vil_logit.view(batch_size, num_options)
    #     loss = task_losses[task_id](vil_logit, target)
    #     _, preds = torch.max(vil_logit, 1)
    #     batch_score = float((preds == target).sum()) / float(batch_size)
    #
    # elif task_cfg[task_id]["type"] == "V-logit":
    #     loss = task_losses[task_id](vision_logit, target)
    #     loss = loss.mean() * target.size(1)
    #     _, select_idx = torch.max(vision_logit, dim=1)
    #     select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
    #     batch_score = float(torch.sum(select_target > 0.5)) / batch_size
    #
    # elif task_cfg[task_id]["type"] == "V-logit-mc":
    #     vision_logit = vision_logit[:, 101:]
    #     vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
    #     vision_logit = vision_logit.unsqueeze(2)
    #     loss = task_losses[task_id](vision_logit, target)
    #     loss = loss.mean() * target.size(1)
    #     _, preds = torch.max(vision_logit, dim=1)
    #     _, target = torch.max(target, dim=1)
    #     batch_score = float((preds == target).sum()) / float(batch_size)
    #
    # elif task_cfg[task_id]["type"] == "VL-binary-classifier":
    #     loss = task_losses[task_id](vil_binary_prediction, target)
    #     loss = loss.mean()
    #     batch_score = compute_score_with_logits(
    #         vil_binary_prediction, target
    #     ).sum() / float(batch_size)
    #
    # elif task_cfg[task_id]["type"] == "VL-tri-classifier":
    #     loss = task_losses[task_id](vil_tri_prediction, target)
    #     loss = loss.mean()
    #     batch_score = compute_score_with_logits(
    #         vil_tri_prediction, target
    #     ).sum() / float(batch_size)

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

    # (YK): Use with test split
    # import pdb
    # pdb.set_trace()
    #
    # for features_h5path in task_feature_reader1.keys():
    #     task_feature_reader1.pop(features_h5path)
    #     if features_h5path != "":
    #         task_feature_reader1[features_h5path.format("trainval")] = None
    #
    # for features_h5path in task_feature_reader2.keys():
    #     task_feature_reader2.pop(features_h5path)
    #     if features_h5path != "":
    #         task_feature_reader2[features_h5path.format("trainval")] = None

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
        num_workers = args.num_workers
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())

        # num_workers = int(num_workers / len(ids))
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
                extra_args=task_cfg[task]
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
                extra_args=task_cfg[task]
            )

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        if "train" in split:
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
                batch_size=32,
                num_workers=2,
                pin_memory=True,
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


def LoadDatasetEval(args, task_cfg, ids):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

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

    task_datasets_val = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_ids.append(task)
        task_name = task_cfg[task]["name"]
        batch_size = args.batch_size
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())

        num_workers = int(args.num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        if args.split:
            eval_split = args.split
        else:
            eval_split = task_cfg[task]["val_split"]

        task_datasets_val[task] = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
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
        )

        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_val[task])
        task_batch_size[task] = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_val,
        task_dataloader_val,
    )


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def EvaluatingModel(
        args,
        task_cfg,
        device,
        task_id,
        batch,
        model,
        task_dataloader,
        task_losses,
        results,
        others,
):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch if isinstance(t, torch.Tensor))

    if task_id == "TASK4" or task_id == "TASK17":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (
            batch
        )
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            batch
        )
    batch_size = features.size(0)

    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )
        co_attention_mask = co_attention_mask.view(
            rbatch_size,
            co_attention_mask.size(2),
            co_attention_mask.size(3),
            co_attention_mask.size(4),
        )

        features = (
            features.unsqueeze(1)
                .unsqueeze(1)
                .expand(batch_size, nround, num_options, max_num_bbox, 2048)
                .contiguous()
                .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
                .unsqueeze(1)
                .expand(batch_size, nround, num_options, max_num_bbox, 5)
                .contiguous()
                .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
                .expand(batch_size, nround, num_options, max_num_bbox)
                .contiguous()
                .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
                .expand(batch_size, num_options, max_num_bbox, 2048)
                .contiguous()
                .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
                .expand(batch_size, num_options, max_num_bbox, 5)
                .contiguous()
                .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
                .expand(batch_size, num_options, max_num_bbox)
                .contiguous()
                .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(
            batch_size * 2, int(features.size(1) / 2), features.size(2)
        )
        spatials = spatials.view(
            batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
        )
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
        co_attention_mask = co_attention_mask.view(
            batch_size * 2,
            int(co_attention_mask.size(1) / 2),
            co_attention_mask.size(2),
        )

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    with torch.no_grad():
        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            task_tokens,
        )

    if task_cfg[task_id]["type"] == "VL-classifier":
        logits = torch.max(vil_prediction, 1)[1].data  # argmax
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": task_dataloader[task_id].dataset.label2ans[
                        logits[i].item()
                    ],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        logits = torch.max(vil_prediction_gqa, 1)[1].data
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "questionId": str(question_id[i].item()),
                    "prediction": task_dataloader[task_id].dataset.label2ans[
                        logits[i].item()
                    ],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_logit.view(batch_size, num_options)
        loss = task_losses[task_id](vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

        probs = torch.softmax(vil_logit, dim=1)
        for i in range(vil_logit.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": [prob.item() for prob in probs[i]],
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vision_logit, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vision_logit[:, 101:]
        vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum())

        for i in range(preds.size(0)):
            results.append({"id": question_id[i].item(), "target": preds[i].item()})

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = task_losses[task_id](vil_binary_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_binary_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = task_losses[task_id](vil_tri_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_tri_prediction, target).sum()

    return float(loss), float(batch_score), batch_size, results, others
