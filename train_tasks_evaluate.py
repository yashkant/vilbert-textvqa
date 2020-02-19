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

def get_arguments():
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
        "--train_iter_multiplier",
        default=1.0,
        type=float,
        help="multiplier for the multi-task training.",
    )
    parser.add_argument(
        "--train_iter_gap",
        default=4,
        type=int,
        help="forward every n iteration is the validation score is not improving over the last 3 epoch, -1 means will stop",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
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
        default=0,
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
        "--vision_scratch",
        action="store_true",
        help="whether pre-trained the image or not.",
    )
    parser.add_argument(
        "--evaluation_interval", default=1, type=int, help="evaluate very n epoch."
    )
    parser.add_argument(
        "--lr_scheduler",
        default=None,
        type=str,
        help="whether use learning rate scheduler.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument(
        "--resume_file", default="", type=str, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--dynamic_attention",
        action="store_true",
        help="whether use dynamic attention.",
    )
    parser.add_argument(
        "--clean_train_sets",
        default=False,
        type=bool,
        help="whether clean train sets for multitask data.",
    )
    parser.add_argument(
        "--visual_target",
        default=0,
        type=int,
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.",
    )
    parser.add_argument(
        "--task_specific_tokens",
        action="store_true",
        help="whether to use task specific tokens for the multi-task learning.",
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

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()

    with open(args.task_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    logger.info("-"*20 + "Config Start" + "-"*20)
    print(vars(args))
    print(task_cfg["TASK19"])
    logger.info("-"*20 + "Config End" + "-"*20)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_type = task_cfg["TASK19"]["model_type"] if args.model_type is None else args.model_type

    if args.baseline:
        from pytorch_transformers.modeling_bert import BertConfig
        from vilbert.basebert import BaseBertForVLTasks
    elif model_type == "22":
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import VILBertForVLTasks
    elif model_type == "31":
        from vilbert.vilbert_3_1 import BertConfig
        from vilbert.vilbert_3_1 import VILBertForVLTasks
    elif model_type == "22lf":
        from vilbert.vilbert_2_2_lf import BertConfig
        from vilbert.vilbert_2_2_lf import VILBertForVLTasks
    elif model_type == "22nf":
        logger.info("Using 2-2 no-fusion model")
        from vilbert.vilbert_2_2_nf import BertConfig
        from vilbert.vilbert_2_2_nf import VILBertForVLTasks
    elif model_type == "m4c":
        logger.info("Using M4C model")
        from vilbert.m4c import BertConfig
        from vilbert.m4c import M4C
    elif model_type == "m4c_rd":
        logger.info("Using M4C-RD model")
        from vilbert.m4c_decode_rd import BertConfig
        from vilbert.m4c_decode_rd import M4C
    elif model_type == "22ss":
        from vilbert.vilbert_ss import BertConfig, VILBertForVLTasks
    elif model_type == "m4c_spatial":
        logger.info("Using M4C-Spatial model")
        from vilbert.m4c_spatial import BertConfig, M4C
    else:
        raise ValueError

    task_names = []
    task_lr = []
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        name = task_cfg[task]["name"]
        task_names.append(name)
        task_lr.append(task_cfg[task]["lr"])

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

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")

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
    task_dataloader_val = LoadDatasets(args, task_cfg, args.tasks.split("-"))

    logdir = os.path.join(savePath, "logs")

    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target

    if args.task_specific_tokens:
        config.task_specific_tokens = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_ave_iter = {}
    # task_stop_controller = {}
    for task_id, num_iter in task_num_iters.items():
        task_ave_iter[task_id] = int(
            task_cfg[task]["num_epoch"]
            * num_iter
            * args.train_iter_multiplier
            / args.num_train_epochs
        )

    task_ave_iter_list = sorted(task_ave_iter.values())
    median_num_iter = task_ave_iter_list[-1]
    num_train_optimization_steps = (
        median_num_iter * args.num_train_epochs // args.gradient_accumulation_steps
    )


    # LOAD PRETRAINED VILBERT
    if args.baseline:
        # (YK): Single-stream baseline model of ViLBERT
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained,
            config=config,
            num_labels=None,
            default_gpu=default_gpu,
        )
    else:
        if args.from_scratch:
            logger.info("Not Using Pre-trained weights")
            if "m4c" not in model_type:
                model = VILBertForVLTasks(
                    config=config,
                    num_labels=None,
                    default_gpu=default_gpu,
                )
            else:
                if model_type == "m4c_spatial":
                    assert "attention_mask_quadrants" in task_cfg["TASK19"]
                    # assert "spatial_type" in task_cfg["TASK19"]
                    # Transfer keys from config to BertConfig
                    transfer_keys = ["attention_mask_quadrants", "hidden_size", "num_implicit_relations", "spatial_type", "num_hidden_layers", "num_spatial_layers", "layer_type_list"]
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

                # Replace keys
                for key in transfer_keys:
                    if key in task_cfg["TASK19"]:
                        config_dict[key] = task_cfg["TASK19"][key]
                        logger.info(f"Transferring keys:  {key}, {config_dict[key]}")
                mmt_config = BertConfig.from_dict(config_dict)

                text_bert_config = BertConfig.from_json_file("config/m4c_textbert_textvqa.json")
                model = M4C(mmt_config, text_bert_config)
        else:
            model = VILBertForVLTasks.from_pretrained(
                args.from_pretrained,
                config=config,
                num_labels=None,
                default_gpu=default_gpu,
            )

    # LOAD LOSSES
    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]


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
        del checkpoint

    model.to(device)

    
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

    
    # This validation score is used for model-saving.
    best_val_score = 0

    # Todo: EVALUATION LOOP (Currently we perform one forward-step and then evaluate)

    epochId = 0
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
        default_gpu
    )


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
    default_gpu
):

    predictions = []
    scores = 0.0
    data_size = 0
    model.eval()

    for i, batch in enumerate(task_dataloader_val[task_id]):
        # batch['spatial_adj_matrix'] = torch.zeros_like(batch['spatial_adj_matrix'])
        # batch['spatial_adj_matrix'] = torch.ones_like(batch['spatial_adj_matrix'])
        # batch['spatial_adj_matrix'] = torch.transpose(batch['spatial_adj_matrix'], 2, 1)

        loss, score, batch_size,  batch_dict = ForwardModelsVal(
            args, task_cfg, device, task_id, batch, model, task_losses
        )

        scores += score * batch_size
        data_size += batch_size
        
        save_keys = ['question_id', 'textvqa_scores', 'targets']

        batch_dict_keys = list(batch_dict.keys())
        for key in batch_dict_keys:
            if key not in save_keys:
                del batch_dict[key]
            else:
                batch_dict[key] = batch_dict[key].cpu().detach().numpy()

            predictions.append(batch_dict)

        if default_gpu:
            sys.stdout.write("%d/%d\r" % (i, len(task_dataloader_val[task_id])))
            sys.stdout.flush()

    print("Val Score: ", float(scores)/data_size)
    
    model.train()
    return score


if __name__ == "__main__":

    main()
