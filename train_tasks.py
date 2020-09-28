# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import random
from io import open
import numpy as np
import pprint
from tensorboardX import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict

import pdb
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from vilbert.task_utils import (
    LoadDatasets,
    LoadLosses,
    ForwardModelsTrain,
    ForwardModelsVal,
    clip_gradients,
    get_optim_scheduler)

import vilbert.utils as utils
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=100,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
             "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
             "0 (default value): dynamic loss scaling.\n"
             "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--save_name", default="", type=str, help="save name for training."
    )
    parser.add_argument(
        "--in_memory",
        default=True,
        type=bool,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--optim", default=None, type=str, help="what to use for the optimization."
    )
    parser.add_argument(
        "--tasks", default="", type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--freeze",
        default=-1,
        type=int,
        help="till which layer of textual stream of vilbert need to fixed.",
    )
    parser.add_argument(
        "--lr_scheduler",
        default=None,
        type=str,
        help="whether use learning rate scheduler.",
    )
    parser.add_argument(
        "--resume_file", default="", type=str, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--task_file", default="vilbert_tasks.yml", type=str, help="joint config file"
    )

    parser.add_argument(
        "--tag", default="debug", type=str, help="tag for the experiment", required=True
    )

    parser.add_argument(
        "--model_type", default=None, type=str, help="Type of model 22 or 31 or 22nf or 22lf"
    )

    parser.add_argument(
        "--from_scratch", action="store_true", help="Initialize ViLBERT weights from scratch/ does it make"
                                                    "what about question encodings!?")

    parser.add_argument(
        "--only_eval", action="store_true", help="Initialize ViLBERT weights from scratch/ does it make"
                                                 "what about question encodings!?")

    parser.add_argument(
        "--use_share2", action="store_true", help="Initialize ViLBERT weights from scratch/ does it make"
                                                 "what about question encodings!?")

    args = parser.parse_args()
    with open(args.task_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    logger.info("-"*20 + "Config Start" + "-"*20)
    print(vars(args))
    print(task_cfg["TASK19"])
    logger.info("-"*20 + "Config End" + "-"*20)

    seed = task_cfg["TASK19"].get("seed", args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Using seed: {seed}")

    model_type = task_cfg["TASK19"]["model_type"] if args.model_type is None else args.model_type

    task_names = []
    task_lr = []
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        name = task_cfg[task]["name"]
        task_names.append(name)
        task_lr.append(task_cfg[task]["lr"])

    if args.only_eval:
        if args.save_name:
            prefix = "-" + args.save_name
        else:
            prefix = ""
        timeStamp = (
                "-".join(task_names)
                + "_"
                + args.config_file.split("/")[1].split(".")[0]
                + prefix
                + f"-{args.tag}"
        )
        savePath = os.path.join(args.output_dir, timeStamp)
        output_checkpoint_path = os.path.join(savePath, "pytorch_ckpt_latest.tar")
        return args.task_file, output_checkpoint_path, args.use_share2

    if model_type == "m4c":
        logger.info("Using M4C model")
        from vilbert.m4c import BertConfig
        from vilbert.m4c import M4C
    elif model_type == "m4c_spatial":
        logger.info("Using M4C-Spatial model")
        from vilbert.m4c_spatial import BertConfig, M4C
    else:
        raise ValueError


    base_lr = min(task_lr)
    loss_scale = {}
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        loss_scale[task] = task_lr[i] / base_lr

    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""
    timeStamp = (
            "-".join(task_names)
            + "_"
            + args.config_file.split("/")[1].split(".")[0]
            + prefix
            + f"-{args.tag}"
    )
    savePath = os.path.join(args.output_dir, timeStamp)

    # bert_weight_name = json.load(
    #     open("config/" + args.bert_model + "_weight_name.json", "r")
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    if task_cfg["TASK19"]["grad_clip_mode"] == "all":
        logger.info(f"Using gradients clipping mode: {task_cfg['TASK19']['grad_clip_mode']}")

    # (YK): default_gpu decides whether to log outputs
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu:
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    config = BertConfig.from_json_file(args.config_file)
    if default_gpu:
        # save all the hidden parameters.
        with open(os.path.join(savePath, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    # LOAD DATASETS
    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, \
    task_dataloader_val = LoadDatasets(args, task_cfg, args.tasks.split("-"), test_val_workers=16, test_val_bs=64)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_ave_iter = {}
    # task_stop_controller = {}
    for task_id, num_iter in task_num_iters.items():
        task_ave_iter[task_id] = int(
            task_cfg[task]["num_epoch"]
            * num_iter
            / args.num_train_epochs
        )

    task_ave_iter_list = sorted(task_ave_iter.values())
    median_num_iter = task_ave_iter_list[-1]
    num_train_optimization_steps = (
            median_num_iter * args.num_train_epochs // args.gradient_accumulation_steps
    )

    # LOAD PRETRAINED VILBERT
    if args.from_scratch:
        logger.info("Not Using Pre-trained weights")
        if "m4c" not in model_type:
            model = VILBertForVLTasks(
                config=config,
                num_labels=None,
                default_gpu=default_gpu,
            )
        else:
            if model_type in ["m4c_spatial", "m4c_topk", "m4c_regat", "m4c_regat_spatial"]:
                if "m4c_spatial" in model_type:
                    assert "attention_mask_quadrants" in task_cfg["TASK19"]
                # assert "spatial_type" in task_cfg["TASK19"]
                # Transfer keys from config to BertConfig
                transfer_keys = ["attention_mask_quadrants",
                                 "hidden_size",
                                 "num_implicit_relations",
                                 "spatial_type",
                                 "num_hidden_layers",
                                 "num_spatial_layers",
                                 "layer_type_list",
                                 "cond_type",
                                 "use_bias",
                                 "no_drop",
                                 "mix_list"]
            elif model_type == "m4c" or model_type == "m4c_rd":
                # Transfer keys from config to BertConfig
                transfer_keys = ["num_hidden_layers"]
            else:
                raise ValueError

            # Common keys
            transfer_keys.extend(["aux_spatial_fusion", "use_aux_heads"])

            # Load config-file M4C
            with open(args.config_file, "r") as file:
                config_dict = json.load(file)

            # Adding blank keys that could be dynamically replaced later
            config_dict["layer_type_list"] = None
            config_dict["mix_list"] = None

            # Always use beam-size 1 for training
            config_dict["beam_size"] = 1

            # Replace keys
            for key in transfer_keys:
                if key in task_cfg["TASK19"]:
                    config_dict[key] = task_cfg["TASK19"][key]
                    logger.info(f"Transferring keys:  {key}, {config_dict[key]}")
            mmt_config = BertConfig.from_dict(config_dict)

            text_bert_config = BertConfig.from_json_file("config/m4c_textbert_textvqa.json")
            model = M4C(mmt_config, text_bert_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training Parameters: {trainable_params}")
    # LOAD LOSSES
    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))
    optimizer_grouped_parameters = model.get_optimizer_parameters(base_lr)

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    optimizer, warmup_scheduler, lr_scheduler, lr_scheduler_config, warmpu_steps = get_optim_scheduler(
        args, task_cfg, optimizer_grouped_parameters, num_train_optimization_steps, base_lr, median_num_iter
    )

    startIterID = 0
    global_step = 0
    start_epoch = 0

    if args.resume_file != "" and os.path.exists(args.resume_file):
        logger.info(f"Resuming from Checkpoint: {args.resume_file}")
        checkpoint = torch.load(args.resume_file, map_location="cpu")
        new_dict = {}
        for attr in checkpoint["model_state_dict"]:
            if attr.startswith("module."):
                new_dict[attr.replace("module.", "", 1)] = checkpoint[
                    "model_state_dict"
                ][attr]
            else:
                new_dict[attr] = checkpoint["model_state_dict"][attr]
        model.load_state_dict(new_dict)
        warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch_id"]) + 1
        del checkpoint

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)

    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if default_gpu:
        print("***** Running training *****")
        print("  Num Iters: ", task_num_iters)
        print("  Batch size: ", task_batch_size)
        print("  Num steps: %d" % num_train_optimization_steps)

    task_iter_train = {name: None for name in task_ids}
    task_count = {name: 0 for name in task_ids}

    if start_epoch >= args.num_train_epochs:
        logger.info("Resetting Train Epochs to 0")
        start_epoch = 0

    if task_cfg["TASK19"]["debug"]:
        median_num_iter = 1000

    # This validation score is used for model-saving.
    best_val_epoch, best_val_score = -1, 0

    global_step = 0
    loss_values, score_values = [], []

    # Train loop
    model.train()
    for epochId in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"):
        assert model.training
        for step in tqdm(range(median_num_iter), desc="Iters"):
            iterId = startIterID + step + (epochId * median_num_iter)

            for task_id in task_ids:
                loss, score = ForwardModelsTrain(
                    args,
                    task_cfg,
                    device,
                    task_id,
                    task_count,
                    task_iter_train,
                    task_dataloader_train,
                    model,
                    task_losses,
                )
                loss_values.append(loss)
                score_values.append(score)
                loss.backward()
                clip_gradients(model, task_cfg["TASK19"]["max_grad_norm"], task_cfg["TASK19"]["grad_clip_mode"])
                optimizer.step()
                warmup_scheduler.step()
                model.zero_grad()
                global_step += 1

            if (
                    step % 20 == 0
                    and step != 0
                    and default_gpu
            ):
                log_str = f"Batch: loss = {float(sum(loss_values)/len(loss_values))}; " \
                          f"accuracy  = {float(sum(score_values)/len(score_values))}; "
                loss_values, score_values = [], []
                if step % 100 == 0:
                    log_str += f"\n lr rates = {[float(grp['lr']) for grp in optimizer.param_groups]}"
                logger.info(log_str)

            # don't run validation during debug runs
            if not task_cfg["TASK19"]["debug"] and (iterId != 0 and iterId % task_num_iters[task_id] == 0) or (
                    epochId == args.num_train_epochs - 1 and step == median_num_iter - 1
            ):
                curr_val_score = evaluate(
                    args,
                    task_dataloader_val,
                    None,
                    task_cfg,
                    device,
                    task_id,
                    model,
                    task_losses,
                    epochId,
                    default_gpu,
                    None,
                )

                # Save a trained model
                logger.info("** ** * Saving fine - tuned model ** ** * ")
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )
                if curr_val_score > best_val_score:
                    output_checkpoint = os.path.join(savePath, "pytorch_ckpt_latest.tar")
                    logger.info(f"Saving Checkpoint: {output_checkpoint}")
                    logger.info(
                        f"Current Validation Score: {curr_val_score} | Previous Best Validation Score: {best_val_score}")
                    best_val_score = curr_val_score
                    torch.save(
                        {
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                            "global_step": global_step,
                            "epoch_id": epochId,
                        },
                        output_checkpoint,
                    )

    return output_checkpoint, args.task_file, args.use_share2


def evaluate(
    args,
    task_dataloader_val,
    task_stop_controller,
    task_cfg,
    device,
    task_id,
    model,
    task_losses,
    epochId,
    default_gpu,
    tbLogger,
):

    model.eval()
    for batch in tqdm(task_dataloader_val[task_id], desc="Validation"):
        loss, score, batch_size = ForwardModelsVal(
            args, task_cfg, device, task_id, batch, model, task_losses
        )
    model.train()
    return score


if __name__ == "__main__":
    try:
        task_file_path, output_checkpoint_path, use_share2 = main()
        for beam_size in [1, 5]:
            for split in ["val"]:
                eval_command = f"python evaluate_textvqa.py " \
                                f"--task_file {task_file_path} " \
                                f"--config_file config/spatial_m4c_mmt_textvqa.json " \
                                f"--batch_size 96 --split {split} " \
                                f"--model_ckpt {output_checkpoint_path} " \
                                f"--beam_size {beam_size} " + (f"--use_share2 {use_share2}" if use_share2 else "")
                print("-"*20)
                print(f"Eval Command: {eval_command}")
                print("-"*20)
                os.system(eval_command)
    finally:
        import os
        os.system("watch -n 1 session-quit-error")
