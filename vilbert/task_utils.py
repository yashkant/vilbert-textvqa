import logging
import time
from bisect import bisect

import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers.optimization import (
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    LambdaLR,
)
from torch.utils.data import DataLoader

from tools.registry import registry
from vilbert.datasets import DatasetMapTrain, VQAClassificationDataset
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
from vilbert.losses import SupConLoss

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
    "SCLLoss": SupConLoss(temperature=registry.temperature,
                          formulation=registry.scl_formulation,
                          base_temperature=registry.base_temperature), # using the default parameter setting
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


def get_optim_scheduler(config,
                        optimizer_grouped_parameters,
                        num_train_optimization_steps,
                        base_lr,
                        no_warmup=False
                        ):
    optim_config = config["optim"]
    scheduler_config = config["lr_scheduler"]

    if optim_config == "Adam":
        optimizer = Adam(optimizer_grouped_parameters, lr=base_lr)
    else:
        raise ValueError

    warmpu_steps = config["warmup_proportion"] * num_train_optimization_steps

    if not no_warmup:
        if scheduler_config == "warmup_linear":
            warmup_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmpu_steps,
                                                    t_total=num_train_optimization_steps)
        elif scheduler_config == "pythia_warmup_decay":
            warmup_iters = config.get("warmup_iters", 1000)
            lr_decay_iters = config.get("lr_decay_iters", [14000, 19000])
            warmup_factor = config.get("warmup_factor", 0.1)
            # total_iters = int(np.ceil(34602/config["batch_size"])*num_train_optimization_steps)

            def pythia_lr_update(_iter):
                if _iter <= warmup_iters:
                    alpha = float(_iter) / float(warmup_iters)
                    return warmup_factor * (1.0 - alpha) + alpha
                else:
                    idx = bisect(lr_decay_iters, _iter)
                    return pow(config.get("lr_decay", 0.2), idx)

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=pythia_lr_update)
            warmpu_steps = -1
        else:
            warmup_scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmpu_steps)
        logger.info(f"Warmup Scheduler: {str(warmup_scheduler)}")
    else:
        warmup_scheduler = None
        logger.info(f"Not using Warmup Scheduler")

    return optimizer, warmup_scheduler, None, scheduler_config, warmpu_steps


def to_device(batch_dict, device):

    if device.type == "cpu":
        return

    for batch in batch_dict:
        for key, value in batch.items():

            if key in ["image_id", "question_id"]:
                continue

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
                     revqa_split="re_total",
                     return_batch=False):

    if not isinstance(batch_dict, tuple) and not isinstance(batch_dict, list):
        batch_dict = [batch_dict]


    # send to device
    to_device(batch_dict, device)

    for batch in batch_dict:
        question = batch["question_indices"]
        # batch["task_tokens"] = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
        batch_size = len(question)
        results_dict = model(batch)
        batch.update(results_dict)

    revqa_split = "val" if registry.revqa_eval_on_val else revqa_split
    loss, batch_score = add_ce_loss(batch_dict[0], device, val_run=True, revqa_eval=revqa_eval, split=revqa_split)


    # When testing move this to above loss calculation
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


def ForwardModelsTrain(
    device,
    dataloaders,
    model,
    train_type="scl"
):

    if train_type == "ce" and registry.alt_train:
        batch_dicts = get_batch(dataloaders, "train_alt")
        # throw away rephrasings batch
        batch_dicts = batch_dicts[:1]
    else:
        batch_dicts = get_batch(dataloaders, "train")

    for batch in batch_dicts:
        results_dict = run_model(batch, model, device)
        batch.update(results_dict)

    if train_type == "scl":
        loss, batch_score = LossMap["SCLLoss"](batch_dicts)
    else:
        loss, batch_score = add_ce_loss(batch_dicts, device)

    del batch_dicts
    return loss, float(batch_score)


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
            bins_key = "revqa_bins" if split in ["re_total", "val"] else "revqa_bt_bins"
            registry[bins_key][min_qid].append((qid, vqa_score))

    return vl_loss, batch_score


def LoadLosses(task_cfg, task_ids):

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
    from vilbert.samplers_new import RandomSampler, NegativeSampler
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    task_feature_reader1[task_cfg["features_h5path1"]] = ImageFeaturesH5Reader(task_cfg["features_h5path1"])
    task_feature_reader2[task_cfg["features_h5path2"]] = ImageFeaturesH5Reader(task_cfg["features_h5path2"])

    dataloaders = {}
    splits = [
        ("train", "train", [("negative", ""), ("random", "_alt")]),
        # ("val", "val", [("none", "")]),
        # ("val", "revqa", [("none", "")]),
        # ("val", "revqa_bt", [("none", "")])
    ]

    for split, key, samplers in splits:
        dataset = VQAClassificationDataset(
                    task=task_cfg["name"],
                    dataroot=task_cfg["dataroot"],
                    annotations_jsonpath=task_cfg[f"{split}_annotations_jsonpath"],
                    split=task_cfg[f"{split}_split"],
                    image_features_reader=task_feature_reader1[
                        task_cfg["features_h5path1"]
                    ],
                    gt_image_features_reader=task_feature_reader2[
                        task_cfg["features_h5path2"]
                    ],
                    tokenizer=tokenizer,
                    padding_index=0,
                    max_seq_length=task_cfg["max_seq_length"],
                    max_region_num=task_cfg["max_region_num"],
                    extra_args=task_cfg
                )

        for sampler_name, tag in samplers:
            sampler = None
            if sampler_name == "random":
                sampler = RandomSampler(dataset)
            elif sampler_name == "negative":
                sampler = NegativeSampler(
                    dataset,
                    task_cfg,
                    split=split
                )

            dataloaders[f"{key}" + tag] = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=task_cfg["batch_size"] if not "alt" in tag else task_cfg["batch_size"]*2,
                num_workers=registry.workers,
                pin_memory=True,
                drop_last=True if split == "train" else False
            )

    # add iterators
    # keys = list(dataloaders.keys())
    # for key in keys:
    #     dataloaders[f"{key}_iter"] = iter(dataloaders[key])

    return dataloaders


def compute_score_with_logits(logits, labels, device):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size())

    if device.type != "cpu":
        one_hots = one_hots.cuda()

    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


