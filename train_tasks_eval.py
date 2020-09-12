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
from copy import deepcopy
from io import open
from itertools import combinations

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

from tools.registry import registry

import vilbert.utils as utils
import torch.distributed as dist

from vilbert.metrics import get_consistency_score

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
        "--use_bt_re",
        default=False,
        type=bool,
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
        default=20,
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
        "--val_split",
        default=None,
        type=str,
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

    arg_keys = [
        "use_bt_re",
    ]

    for key in arg_keys:
        registry[key] = getattr(args, key)

    print(json.dumps(vars(registry), indent=2))


def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.task_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    if args.val_split is not None:
        task_cfg["TASK19"]["val_split"] = args.val_split

    logger.info("-" * 20 + "Config Start" + "-" * 20)
    print(json.dumps(vars(args), indent=2))
    print(json.dumps(vars(task_cfg["TASK19"]), indent=2))
    seed = task_cfg["TASK19"].get("seed", args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Using seed: {seed}")

    assert_add_registry(task_cfg, args)
    logger.info("-"*20 + "Config End" + "-"*20)

    from vilbert.task_utils import (
        LoadDatasets,
        LoadLosses,
        ForwardModelsTrain,
        clip_gradients,
        get_optim_scheduler)

    model_type = task_cfg["TASK19"]["model_type"] if args.model_type is None else args.model_type

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
        torch.backends.cudnn.benchmark = True
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
    task_dataloader_val = LoadDatasets(args, task_cfg, args.tasks.split("-"), split="val")


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
        model.load_state_dict(new_dict)

        # Add checkpoint string
        log_keys = ["cs_rank", "vqa_rank", "vqa_acc", "cs_score", "global_step"]
        ckpt_string = f"-------------- \n Checkpoint information: \n"
        for key in log_keys:
            ckpt_string += f"{key}: {checkpoint[key]} \n"
        ckpt_string += "---------------"
        logger.info(ckpt_string)


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

    if start_epoch >= args.num_train_epochs:
        logger.info("Resetting Train Epochs to 0")
        start_epoch = 0

    # This validation score/loss is used for model-saving.
    best_val_score = 0
    best_val_loss = np.inf
    best_val_epoch = 0
    import time
    task_id = "TASK19"

    # import pdb
    # pdb.set_trace()

    logger.info("Starting Validation Run....")
    evaluate(
        args,
        task_dataloader_val,
        None,
        task_cfg,
        device,
        task_id,
        model,
        task_losses,
    )

    print(f"Best Validation Score: {best_val_score} and Best Validation Epoch: {best_val_epoch}")
    tbLogger.txt_close()


def evaluate(
    args,
    task_dataloader_val,
    task_stop_controller,
    task_cfg,
    device,
    task_id,
    model,
    task_losses,
):

    from vilbert.task_utils import ForwardModelsVal
    registry.eval_only = True

    # reset revqa_bins for each evaluation!
    if registry.revqa_eval or registry.revqa_eval_on_val:
        from easydict import EasyDict
        dd = defaultdict(list)
        super(EasyDict, registry).__setattr__("revqa_bins", dd)
        super(EasyDict, registry).__setitem__("revqa_bins", dd)

    model.eval()
    results = {}

    for batch in tqdm(task_dataloader_val[task_id]):
        with torch.no_grad():  # turn off autograd engine
            batch_dict = ForwardModelsVal(
                args, task_cfg, device, task_id, batch, model, task_losses, revqa_eval=registry.revqa_eval_on_val
            )

        # Eval-AI file
        if registry["eval_only"]:
            # build the json file here!
            logits = torch.max(batch_dict["vil_prediction"], 1)[1].data  # argmax

            for idx in range(logits.size(0)):
                results[batch_dict["question_id"][idx].item()] = \
                    {
                        "question_id": batch_dict["question_id"][idx].item(),
                        "answer": task_dataloader_val[task_id].dataset.label2ans[
                            logits[idx].item()
                        ],
                        "vqa_score": np.round(batch_dict["vqa_scores"][idx], 1) if "vqa_scores" in batch_dict else None
                    }

    c_scores = None
    revqa_bins_scores = None
    # if registry.revqa_eval:
    #     cs_results = {}
    #     for batch in tqdm(task_dataloader_val["revqa"], desc="Revqa Eval"):
    #         with torch.no_grad():  # turn off autograd engine
    #             batch_dict = ForwardModelsVal(
    #                 args, task_cfg, device, task_id, batch, model, task_losses, revqa_eval=True
    #             )
    #
    #             # build the json file here!
    #             logits = torch.max(batch_dict["vil_prediction"], 1)[1].data  # argmax
    #
    #             for idx in range(logits.size(0)):
    #                 cs_results[batch_dict["question_id"][idx].item()] = \
    #                     {
    #                         "question_id": batch_dict["question_id"][idx].item(),
    #                         "answer": task_dataloader_val[task_id].dataset.label2ans[
    #                             logits[idx].item()
    #                         ],
    #                         "vqa_score": np.round(batch_dict["vqa_scores"][idx], 1) if "vqa_scores" in batch_dict else None
    #                     }
    #     c_scores, revqa_bins_scores = get_consistency_score(results=cs_results, bins_scores=True)
    #
    # elif registry.revqa_eval_on_val:
    #     logger.info("Ran re-vqa eval on validation!")
    #     c_scores, revqa_bins_scores = get_consistency_score(results=results, bins_scores=True)

    final_results = {}
    final_results["results"] = results
    final_results["revqa_bins_scores"] = revqa_bins_scores
    final_results["c_scores"] = c_scores
    for key in registry:
        if "question_rephrase_dict" in key:
            final_results[key] = registry[key]

    evalai_results = deepcopy(list(results.values()))
    for item in evalai_results:
        del item["vqa_score"]

    save_dir = os.path.split(args.resume_file)[0]
    evalai_path = f"{save_dir}/evalai_{task_cfg['TASK19']['val_split']}.json"
    preds_path = f"{save_dir}/preds_revqa_{task_cfg['TASK19']['val_split']}.json"

    # if registry.use_bt_re:
    #     preds_path = f"{save_dir}/preds_revqa_bt.json"

    json.dump(evalai_results, open(evalai_path, "w"))
    json.dump(final_results, open(preds_path, "w"))

    print(f"Dumped: {evalai_path}")
    print(f"Dumped: {preds_path}")

    model.train()


if __name__ == "__main__":
    try:
        main()
    finally:
        import os
        os.system("watch -n 1 session-quit-error")
