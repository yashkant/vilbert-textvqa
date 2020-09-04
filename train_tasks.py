# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bisect
import json
import logging
import os
import random
from collections import defaultdict
from io import open
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

from evaluator import final_evaluate
from tools.registry import registry

import vilbert.utils as utils
import torch.distributed as dist

from vilbert.metrics import get_consistency_score

def assert_add_registry(task_cfg, args):
    assert task_cfg["TASK19"]["num_epoch"] == args.num_train_epochs
    assert_keys = [
        "val_batch_size",
        "val_workers",
        "train_workers"
    ]

    for key in assert_keys:
        assert key in task_cfg["TASK19"], f"Key not found: {key}"

    add_keys = [
        "val_batch_size",
        "val_workers",
        "train_workers",
        "num_epoch",
        "val_split",
        "train_split",
        ("revqa_eval", False),
        ("use_ce_loss", False),
        ("scl_coeff", -1),
        "batch_size",
        ("mask_image", False),
        ("scl_formulation", "normal"),
        ("val_drop_last", False),
        ("squint_loss", False),
        ("squint_layers", None),
        ("squint_type", None),
        ("ce_half", False),
        ("use_rephrasings", True),
        ("aug_filter", None),
        ("use_old_sampler", False),
        ("sampler_type", None),
        ("weighted_sampling", False),
        ("remove_ambiguous", False),
        ("sdebug", False),
        ("scl_mask_thresh", -1),
        ("scl_mask_rescale_factor", -1),
        ("scl_random_sampler", False),
        ("alt_train", False),
        ("alt_re", True),
        ("ce_freq", 2),
        ("scl_freq", 2),
        ("use_freq", "ce"),
        ("save_top", 3),
        ("use_bt_re", False),
        ("use_no_re", False),
        ("revqa_eval_on_val", False),
        ("num_rep", 2),
        ("base_temperature", 0.07),
        ("temperature", 0.5),   # old-defaults
        ("bt_eval_key", "val_aug"),
        ("allowed_only", False),
        ("two_norm", False),
        ("freeze_textbert_and_mmt", False),
    ]

    for key in add_keys:
        if isinstance(key, tuple):
            registry[key[0]] = task_cfg["TASK19"].get(key[0], key[1])
        else:
            registry[key] = task_cfg["TASK19"][key]

    if registry.sdebug:
        registry.revqa_eval = False
        task_cfg["TASK19"]["revqa_eval"] = False
    else:
        if not registry.revqa_eval:
            print(f"CS scores evaluation is disabled, press c to continue")
            # import pdb
            # pdb.set_trace()
    print(json.dumps(vars(registry), indent=2))
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


def get_parser():
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
        default=5,
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
        default=False,
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
        "--hard_stop",
        type=int,
        required=True
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
        "--task_file", default="sweeps/vqa_task.yml", type=str, help="joint config file"
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

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.task_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    logger.info("-" * 20 + "Config Start" + "-" * 20)
    print(json.dumps(vars(args), indent=2))
    print(json.dumps(vars(task_cfg["TASK19"]), indent=2))
    seed = task_cfg["TASK19"].get("seed", args.seed)

    # Set deterministic mode on!
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Using seed: {seed}")

    assert_add_registry(task_cfg, args)
    logger.info("-" * 20 + "Config End" + "-" * 20)

    from vilbert.task_utils import (
        LoadDatasets,
        LoadLosses,
        ForwardModelsTrain,
        clip_gradients,
        get_optim_scheduler)

    model_type = task_cfg["TASK19"]["model_type"] if args.model_type is None else args.model_type

    assert task_cfg["TASK19"]["features_h5path2"] != ""

    if args.baseline:
        from pytorch_transformers.modeling_bert import BertConfig
        from vilbert.basebert import BaseBertForVLTasks
    elif model_type == "vilbert":
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import VILBertForVLTasks
    elif model_type == "m4c_spatial":
        logger.info("Using M4C-Spatial model")
        from vilbert.m4c_spatial import BertConfig, M4C
    elif model_type == "m4c":
        logger.info("Using M4C model")
        from vilbert.m4c import BertConfig, M4C
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

    # bert_weight_name = json.load(
    #     open("config/" + args.bert_model + "_weight_name.json", "r")
    # )

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        # torch.backends.cudnn.benchmark = True
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
            print("\n", file=f)
            print(task_cfg, file=f)

    # LOAD DATASETS
    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, \
    task_dataloader_val = LoadDatasets(args, task_cfg, args.tasks.split("-"))

    if registry.sdebug:
        batch_iter = iter(task_dataloader_train["TASK19"])
        batch_dict = batch_iter.next()
        epoch_inds = np.load(registry.sampler_cache_name, allow_pickle=True)[0]
        import pdb
        pdb.set_trace()



    logdir = os.path.join(savePath, "logs")
    tbLogger = utils.tbLogger(
        logdir,
        savePath,
        task_names,
        task_ids,
        task_num_iters,
        args.gradient_accumulation_steps,
    )

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
                transfer_keys.extend(["aux_spatial_fusion",
                                      "use_aux_heads",
                                      "contrastive",
                                      "weight_decay",
                                      "freeze_mmt_and_textbert",
                                      "lr_scale_text_bert",
                                      "contrast_out_dim",
                                      "lr_scale_mmt",
                                      "output_attentions",
                                      ])

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

                text_bert_config = "config/m4c_textbert_vqa.json"

                if task_cfg["TASK19"].get("freeze_text_bert", False):
                    logger.info("TextBert Frozen")
                    text_bert_config = "config/m4c_textbert_vqa_frozen.json"

                if not task_cfg["TASK19"].get("text_bert_init_from_bert_base", True):
                    text_bert_config = "config/m4c_textbert_vqa_scratch.json"

                text_bert_config = BertConfig.from_json_file(text_bert_config)
                model = M4C(mmt_config, text_bert_config)
        else:
            model = VILBertForVLTasks.from_pretrained(
                args.from_pretrained,
                config=config,
                num_labels=None,
                default_gpu=default_gpu,
            )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training Parameters: {trainable_params}")
    # LOAD LOSSES
    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))

    # Turned of weight-decay and fine-tune stuff in ViLBERT!
    if "m4c" not in model_type:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                lr = base_lr
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]
    else:
        optimizer_grouped_parameters = model.get_optimizer_parameters(base_lr)

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    optimizer, warmup_scheduler, lr_scheduler, lr_scheduler_config, warmpu_steps = get_optim_scheduler(
        args, task_cfg, optimizer_grouped_parameters, num_train_optimization_steps, base_lr, median_num_iter
    )

    startIterID = 0
    global_step = 0
    start_epoch = 0

    # for finetuning we need resume-file
    if task_cfg["TASK19"].get("contrastive", None) == "finetune":
        assert args.resume_file != ""

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

        if registry.freeze_textbert_and_mmt:
            model.load_state_dict(new_dict, strict=False)
            trainable_params = [p.numel() for p in model.parameters() if p.requires_grad]
            trainable_params = sum(trainable_params)
            logger.info(f"Training Parameters: {trainable_params}")
        else:
            model.load_state_dict(new_dict)

        if task_cfg["TASK19"].get("load_state", False):
            print("Loading Optimizer and Scheduler States")
            warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            global_step = checkpoint["global_step"]
            start_epoch = int(checkpoint["epoch_id"]) + 1
            # task_stop_controller = checkpoint["task_stop_controller"]
            tbLogger = checkpoint["tb_logger"]
        else:
            print("Not loading optimizer and scheduler states")
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

    if registry.alt_train:
        task_iter_train[task_ids[0] + "alt"] = None
        task_count[task_ids[0] + "alt"] = 0

    if start_epoch >= args.num_train_epochs:
        logger.info("Resetting Train Epochs to 0")
        start_epoch = 0

    # This val  idation score/loss is used for model-saving.
    best_val_score = 0
    best_val_loss = np.inf
    best_val_epoch = 0
    # grad_dots = []
    grad_diff = []
    import time
    eval_iter_factor = task_cfg["TASK19"].get("eval_iter_factor", 1500)

    # list of (iter, vqa_score, cs_score, cs_score_bt)
    best_checkpoints = [(-1, -1, -1, -1)]
    cs_checkpoints = [(-1, -1, -1, -1)]
    cs_bt_checkpoints = [(-1, -1, -1, -1)]

    ckpts_log_file = os.path.join(savePath, "ckpts.log")

    # TRAINING LOOP
    for epochId in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"):
        model.train()
        for step in tqdm(range(median_num_iter), desc="Iters"):

            if global_step > args.hard_stop:
                logger.info(f"Breaking w/ hard-stop at {args.hard_stop}")
                break

            # logger.info(f"LR rates: {[grp['lr'] for grp in optimizer.param_groups]}")
            start_time = time.time()
            iterId = startIterID + step + (epochId * median_num_iter)
            first_task = True
            for task_id in task_ids:
                is_forward = True
                if is_forward:

                    train_type = "ce" if registry.use_freq == "ce" else "scl"

                    if registry.alt_train and registry.use_freq == "ce" \
                        and iterId % registry.ce_freq == 1:
                        train_type = "scl"

                    if registry.alt_train and registry.use_freq == "scl" \
                        and iterId % registry.scl_freq == 1:
                        train_type = "ce"

                    # print(f"Train Type: {train_type}")
                    loss, score, losses = ForwardModelsTrain(
                        args,
                        task_cfg,
                        device,
                        task_id,
                        task_count,
                        task_iter_train,
                        task_dataloader_train,
                        model,
                        task_losses,
                        train_type
                    )
                    iter_time = time.time()
                    # print(f"Iter time: {iter_time - start_time}")

                    if task_cfg["TASK19"].get("pcgrad", False):
                        # if task_cfg["TASK19"].get("default_double", False):
                        #     torch.set_default_tensor_type(torch.DoubleTensor)

                        losses[0].backward(retain_graph=True)
                        l1_grad = [k.grad.clone().view(-1) for k in model.parameters()]
                        l1_grad = torch.cat(l1_grad)
                        optimizer.zero_grad()
                        model.zero_grad()
                        losses[1].backward(retain_graph=True)
                        l2_grad = [k.grad.clone().view(-1) for k in model.parameters()]
                        l2_grad = torch.cat(l2_grad)
                        l_dot = torch.dot(l1_grad, l2_grad)
                        # grad_dots.append(float(l_dot))
                        l2_grad_mag = torch.dot(l2_grad, l2_grad)
                        l1_grad_mag = torch.dot(l1_grad, l1_grad)

                        # l1: SCL and l2: CE
                        if l_dot < 0 and task_cfg["TASK19"].get("pcgrad_adjust", True):
                            l1_proj = (l_dot/l2_grad_mag) * l2_grad
                            l2_proj = (l_dot/l1_grad_mag) * l1_grad

                            if "scl" in task_cfg["TASK19"]["pcgrad_on"]:
                                l1_grad = l1_grad - l1_proj

                            if "ce" in task_cfg["TASK19"]["pcgrad_on"]:
                                l2_grad = l2_grad - l2_proj

                            # assert torch.dot(l1_grad, l2_grad) >= 0

                        l_grad = l1_grad + l2_grad
                        optimizer.zero_grad()
                        model.zero_grad()
                        (losses[0] + losses[1]).backward()
                        l3_grad = [k.grad.clone().view(-1) for k in model.parameters()]
                        l3_grad = torch.cat(l3_grad)
                        grad_diff.append(float((l3_grad - l_grad).sum()))

                        del l1_grad
                        del l2_grad
                        del l3_grad

                        prev_sum = 0
                        for param in model.parameters():
                            slice_start, slice_end = prev_sum, prev_sum + np.prod(list(param.grad.shape))
                            param.grad = l_grad[slice_start:slice_end].view_as(param.grad).clone()
                            prev_sum = slice_end

                        # if task_cfg["TASK19"].get("default_double", False):
                        #     torch.set_default_tensor_type(torch.FloatTensor)

                    elif task_cfg["TASK19"].get("pcgrad_module", False):
                        losses[0].backward(retain_graph=True)
                        l1_grad = [k.grad.clone().view(-1) for k in model.parameters()]
                        # l1_grad = torch.cat(l1_grad)
                        optimizer.zero_grad()
                        model.zero_grad()
                        losses[1].backward()
                        l2_grad = [k.grad.clone().view(-1) for k in model.parameters()]
                        # l2_grad = torch.cat(l2_grad)
                        # grad_dots.append(float(l_dot))

                        l_grad = []
                        l_dots = []
                        l1_grad_mag_mods = []
                        l2_grad_mag_mods = []

                        # l1: SCL and l2: CE
                        for l1_mod_grad, l2_mod_grad in zip(l1_grad, l2_grad):
                            l_mod_dot = torch.dot(l1_mod_grad, l2_mod_grad)
                            l2_grad_mag = torch.dot(l2_mod_grad, l2_mod_grad)
                            l1_grad_mag = torch.dot(l1_mod_grad, l1_mod_grad)

                            if l_mod_dot < 0 and task_cfg["TASK19"].get("pcgrad_adjust", True):
                                l1_proj = (l_mod_dot/l2_grad_mag) * l2_mod_grad
                                l2_proj = (l_mod_dot/l1_grad_mag) * l1_mod_grad

                                if "scl" in task_cfg["TASK19"]["pcgrad_on"]:
                                    l1_mod_grad = l1_mod_grad - l1_proj

                                if "ce" in task_cfg["TASK19"]["pcgrad_on"]:
                                    l2_mod_grad = l2_mod_grad - l2_proj

                            l_grad.append(l1_mod_grad + l2_mod_grad)
                            l_dots.append(float(l_mod_dot))
                            l1_grad_mag_mods.append(l1_grad_mag)
                            l2_grad_mag_mods.append(l2_grad_mag)


                        optimizer.zero_grad()
                        model.zero_grad()
                        # (losses[0] + losses[1]).backward()
                        # l3_grad = [k.grad.clone().view(-1) for k in model.parameters()]
                        # l3_grad = torch.cat(l3_grad)
                        # grad_diff.append(float((l3_grad - l_grad).sum()))

                        del l1_grad
                        del l2_grad
                        # del l3_grad

                        for param, l_g in zip(model.parameters(), l_grad):
                            param.grad = l_g.view_as(param.grad).clone()

                        l_dot = sum(l_dots)/len(l_dots)
                        l1_grad_mag = sum(l1_grad_mag_mods)/len(l1_grad_mag_mods)
                        l2_grad_mag = sum(l2_grad_mag_mods)/len(l2_grad_mag_mods)

                    elif task_cfg["TASK19"].get("step_scl", False):
                        if iterId > task_cfg["TASK19"].get("scl_start_step", None):
                            loss.backward()
                        else:
                            losses[1].backward()

                    elif task_cfg["TASK19"].get("ramp_scl", False):
                        ramp_start = task_cfg["TASK19"].get("scl_start_ramp", None)
                        ramp_stop = task_cfg["TASK19"].get("scl_stop_ramp", None)
                        if iterId < ramp_start:
                            losses[1].backward()
                        elif ramp_stop > iterId > ramp_start:
                            ramp_loss = losses[1] + ((iterId - ramp_start)/(ramp_stop-ramp_start)) * losses[0]
                            ramp_loss.backward()
                        else:
                            loss.backward()

                    else:
                        loss.backward()

                    back_time = time.time()
                    # print(f"Backward time: {back_time - iter_time}")


                    if task_cfg["TASK19"]["grad_clip_mode"] == "all":
                        clip_gradients(model, task_cfg["TASK19"]["max_grad_norm"], task_cfg["TASK19"]["grad_clip_mode"])

                    optimizer.step()
                    warmup_scheduler.step()
                    model.zero_grad()
                    optimizer.zero_grad()
                    if first_task:
                        global_step += 1
                        first_task = False

                    if train_type == "ce" or (not registry.alt_train):
                        tbLogger.step_train(
                            epochId,
                            iterId,
                            float(loss),
                            float(score),
                            optimizer.param_groups[0]["lr"],
                            task_id,
                            "train",
                            extra_dict={
                                "grad_dot": l_dot,
                                "grad_mag_scl": l1_grad_mag,
                                "grad_mag_ce": l2_grad_mag,
                            } if (task_cfg["TASK19"].get("pcgrad", False) or
                                  task_cfg["TASK19"].get("pcgrad_module", False)) else None
                        )

                    del loss
                    del score
                    del losses

            # gc.collect()
            finish_time = time.time()
            # print(f"Finish time: {finish_time - back_time}")

            if "cosine" in lr_scheduler_config and global_step > warmpu_steps:
                lr_scheduler.step()

            if (
                    step % (20 * args.gradient_accumulation_steps) == 0
                    and step != 0
                    and default_gpu
            ):
                tbLogger.showLossTrain()
                logger.info(f"LR rates: {[grp['lr'] for grp in optimizer.param_groups]}, "
                            # f"Grad Dot: {sum(grad_dots) / len(grad_dots) if len(grad_dots) > 0 else None}, '"
                            f"Grad Diff: {sum(grad_diff) / len(grad_diff) if len(grad_diff) > 0 else None}")
                grad_dots, grad_diff = [], []

            # decided whether to evaluate on each tasks.
            for task_id in task_ids:
                # don't run validation during debug runs
                # if task_cfg["TASK19"]["debug"]:
                #     break

                if (iterId != 0 and iterId % eval_iter_factor == 0) or (
                        epochId == args.num_train_epochs - 1 and step == median_num_iter - 1
                ) or (global_step == args.hard_stop):
                    logger.info("Starting Validation Run....")
                    curr_val_score, curr_val_loss, cs_scores, cs_bt_scores = evaluate(
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
                        tbLogger,
                    )

                    if default_gpu:
                        # Save a trained model
                        logger.info("** ** * Saving fine - tuned model ** ** * ")
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Only save the model it-self

                        checkpoint_dict = \
                            {
                                    "model_state_dict": model_to_save.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                                    "global_step": global_step,
                                    "epoch_id": epochId,
                                    "tb_logger": tbLogger
                            }

                        if curr_val_score >= best_val_score and task_cfg["TASK19"].get("monitor_value",
                                                                                       "val_score") == "val_score" and cs_scores is None:
                            output_checkpoint = os.path.join(savePath, "pytorch_ckpt_latest.tar")
                            logger.info(f"Saving Checkpoint: {output_checkpoint}")
                            logger.info(
                                f"Current Validation Score: {curr_val_score} | Previous Best Validation Score: {best_val_score}")
                            best_val_score = curr_val_score
                            best_val_epoch = epochId
                            torch.save(checkpoint_dict, output_checkpoint)

                        elif curr_val_loss <= best_val_loss and task_cfg["TASK19"].get("monitor_value",
                                                                                       "val_loss") == "val_loss" and cs_scores is None:
                            output_checkpoint = os.path.join(savePath, "pytorch_ckpt_latest.tar")
                            logger.info(f"Saving Checkpoint: {output_checkpoint}")
                            logger.info(
                                f"Current Validation Loss: {curr_val_loss} | Previous Best Validation Loss: {best_val_loss}")
                            best_val_loss = curr_val_loss
                            torch.save(checkpoint_dict, output_checkpoint)

                        elif cs_scores is not None:
                            try:
                                save_thresh = registry.save_top
                                curr_cs_score = cs_scores[-1]
                                curr_cs_bt_score = cs_bt_scores[-1]

                                top_vqa_scores = [ ckpt[1] for ckpt in sorted(best_checkpoints, key=lambda x: x[1])][-save_thresh:]
                                top_cs_scores = [ ckpt[2] for ckpt in sorted(best_checkpoints, key=lambda x: x[2])][-save_thresh:]
                                top_cs_bt_scores = [ckpt[3] for ckpt in sorted(best_checkpoints, key=lambda x: x[3])][-save_thresh:]

                                vqa_rank = bisect.bisect(top_vqa_scores, curr_val_score)
                                cs_rank = bisect.bisect(top_cs_scores, curr_cs_score)
                                cs_bt_rank = bisect.bisect(top_cs_bt_scores, curr_cs_bt_score)


                                logger.info(f"Current Val Score and Rank: {curr_val_score} / {vqa_rank} | Previous Best Val Scores: {top_vqa_scores}")
                                logger.info(f"Current CS Score and Rank: {curr_cs_score} / {cs_rank} | Previous Best CS Scores: {top_cs_scores}")
                                logger.info(f"Current CS BT Score and Rank: {curr_cs_bt_score} / {cs_bt_rank} | Previous Best CS BT Scores: {top_cs_bt_scores}")

                                logger.info(f"Top CS checkpoint: {sorted(best_checkpoints, key=lambda x: x[2])[-1]}")
                                logger.info(f"Top CS BT checkpoint: {sorted(best_checkpoints, key=lambda x: x[3])[-1]}")
                                logger.info(f"Checkpoint Dir: {savePath}")
                                checkpoint_dict.update(
                                    {
                                        "vqa_rank": vqa_rank,
                                        "cs_rank": cs_rank,
                                        "cs_bt_rank": cs_bt_rank,
                                        "vqa_acc": curr_val_score,
                                        "cs_score": curr_cs_score,
                                        "cs_bt_score": curr_cs_bt_score
                                    }
                                )

                                if vqa_rank != 0:
                                    logger.info(f"Saving for VQA score w/ rank: {vqa_rank}")
                                    output_checkpoint = os.path.join(savePath, f"vqa_rank_{vqa_rank}.tar")
                                    torch.save(checkpoint_dict, output_checkpoint)

                                if cs_rank != 0:
                                    logger.info(f"Saving for CS score w/ rank: {cs_rank}")
                                    output_checkpoint = os.path.join(savePath, f"cs_rank_{cs_rank}.tar")
                                    torch.save(checkpoint_dict, output_checkpoint)

                                if cs_bt_rank != 0:
                                    logger.info(f"Saving for CS BT score w/ rank: {cs_bt_rank}")
                                    output_checkpoint = os.path.join(savePath, f"cs_bt_rank_{cs_bt_rank}.tar")
                                    torch.save(checkpoint_dict, output_checkpoint)

                                best_checkpoints.append((global_step, curr_val_score, curr_cs_score, curr_cs_bt_score))
                                cs_checkpoints.append(cs_scores)
                                cs_bt_checkpoints.append(cs_bt_scores)

                                with open(ckpts_log_file, "w") as outfile:
                                    dump_str = ""
                                    assert len(best_checkpoints) == len(cs_bt_checkpoints) == len(cs_checkpoints)
                                    for ckpt, cs_ckpt, cs_bt_ckpt in zip(best_checkpoints,
                                                                         cs_checkpoints,
                                                                         cs_bt_checkpoints):
                                        dump_str += f"Ckpt: {ckpt} | CS : {cs_ckpt}| CS-BT: {cs_bt_ckpt} \n"
                                    outfile.write(dump_str)
                                logger.info(f"Dumped File: {ckpts_log_file}")
                            except:
                                import pdb
                                pdb.set_trace()

            residue_time = time.time()
            # print(f"Finish time: {residue_time - finish_time}")

        if global_step > args.hard_stop:
            break

        if lr_scheduler_config == "automatic":
            lr_scheduler.step(sum(val_scores.values()))
            logger.info("best average score is %3f" % lr_scheduler.best)
        elif lr_scheduler_config == "mannul":
            lr_scheduler.step()
    # Todo: Add config for final_evaluation splits [re-vqa, val, test]

    if len(best_checkpoints) > 1:
        top_vqa_ckpts = [(c[0], c[1]) for c in sorted(best_checkpoints, key=lambda x: x[1], reverse=True)[:registry.save_top]]
        top_cs_ckpts = [(c[0], c[2]) for c in sorted(best_checkpoints, key=lambda x: x[2], reverse=True)[:registry.save_top]]
        print(f"Top CS checkpoints: {top_cs_ckpts}")
        print(f"Top VQA checkpoints: {top_vqa_ckpts}")
        # with open(ckpts_log_file, "w") as outfile:
        #     outfile.write("\n".join([str(s) for s in best_checkpoints]))
        # logger.info(f"Dumped File: {ckpts_log_file}")
    else:
        print(f"Best Validation Score: {best_val_score} and Best Validation Epoch: {best_val_epoch}")
    tbLogger.txt_close()

    # Run final-evaluation for EvalAI file.
    for split in ["test", "val"]:
        final_evaluate(
            args, task_cfg, device, task_ids[0], model, task_losses, savePath, split
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
        default_gpu,
        tbLogger,
):
    # reset revqa_bins for each evaluation!
    if registry.revqa_eval or registry.revqa_eval_on_val:
        from easydict import EasyDict
        dd = defaultdict(list)
        dd_bt = defaultdict(list)

        super(EasyDict, registry).__setattr__("revqa_bins", dd)
        super(EasyDict, registry).__setitem__("revqa_bins", dd)

        super(EasyDict, registry).__setattr__("revqa_bt_bins", dd_bt)
        super(EasyDict, registry).__setitem__("revqa_bt_bins", dd_bt)

    from vilbert.task_utils import ForwardModelsVal
    model.eval()  # turn off dropout/batch-norm
    for i, batch in enumerate(task_dataloader_val[task_id]):
        with torch.no_grad():  # turn off autograd engine
            loss, score, batch_size = ForwardModelsVal(
                args, task_cfg, device, task_id, batch, model, task_losses, revqa_eval=registry.revqa_eval_on_val
            )
        tbLogger.step_val(
            epochId, float(loss), float(score), task_id, batch_size, "val"
        )
        if default_gpu:
            sys.stdout.write("%d/%d\r" % (i, len(task_dataloader_val[task_id])))
            sys.stdout.flush()

    c_scores = None
    c_scores_bt = None

    if registry.revqa_eval:
        for batch in tqdm(task_dataloader_val["revqa"], desc="Revqa Eval"):
            with torch.no_grad():  # turn off autograd engine
                loss, score, batch_size = ForwardModelsVal(
                    args, task_cfg, device, task_id, batch, model, task_losses, revqa_eval=True, revqa_split="re_total"
                )
        c_scores = get_consistency_score()

        for batch in tqdm(task_dataloader_val["revqa_bt"], desc="Revqa BT Eval"):
            with torch.no_grad():  # turn off autograd engine
                loss, score, batch_size = ForwardModelsVal(
                    args, task_cfg, device, task_id, batch, model, task_losses, revqa_eval=True, revqa_split="re_total_bt"
                )
        c_scores_bt = get_consistency_score(bins_key="revqa_bt_bins")

        # merge all key-values
        c_scores.update(c_scores_bt)

    elif registry.revqa_eval_on_val:
        logger.info("Ran re-vqa eval on validation!")
        c_scores = get_consistency_score()

    score, loss = tbLogger.showLossVal(task_id, task_stop_controller=None, c_scores=c_scores)
    model.train()

    # return only a list of c_scores
    if c_scores is not None:
        c_scores_bt = [c_scores[key] if key in c_scores else -1 for key in ["1_bt", "2_bt", "3_bt","4_bt"]]
        c_scores = [c_scores[str(key)] for key in [1,2,3,4]]

    return score, loss, c_scores, c_scores_bt


if __name__ == "__main__":
    main()
