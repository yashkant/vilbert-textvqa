# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from evaluator import Evaluator
from vilbert.task_utils import (
    load_datasets,
    ForwardModelsTrain,
    ForwardModelsVal,
    clip_gradients,
    get_optim_scheduler)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
pprint = lambda _dict: json.dumps(_dict, indent=4, sort_keys=True)


def get_parser_cfg():
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
    # parser.add_argument(
    #     "--config_file",
    #     required=True,
    #     type=str,
    #     help="The config file which specified the model details.",
    # )
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
    # parser.add_argument(
    #     "--loss_scale",
    #     type=float,
    #     default=0,
    #     help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
    #          "0 (default value): dynamic loss scaling.\n"
    #          "Positive power of 2: static loss scaling value.\n",
    # )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers in the dataloader.",
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
    # parser.add_argument(
    #     "--tasks", default="", type=str, help="1-2-3... training task separate by -"
    # )
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
        "--task_file", required=True, type=str, help="joint config file"
    )

    parser.add_argument(
        "--tag", default="debug", type=str, help="tag for the experiment", required=True
    )

    parser.add_argument(
        "--model_type", default=None, type=str, help="Type of model 22 or 31 or 22nf or 22lf"
    )

    # parser.add_argument(
    #     "--from_scratch", action="store_true", help="Initialize ViLBERT weights from scratch/ does it make"
    #                                                 "what about question encodings!?")

    parser.add_argument(
        "--only_eval", action="store_true", help="Initialize ViLBERT weights from scratch/ does it make"
                                                 "what about question encodings!?")

    parser.add_argument(
        "--use_share2", action="store_true", help="Initialize ViLBERT weights from scratch/ does it make"
                                                 "what about question encodings!?")

    args = parser.parse_args()
    with open(args.task_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    # Set seed for reproducibility
    seed = task_cfg.get("seed", args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # assert os.path.exists(args.config_file)
    logger.info("-"*20 + "Command Line Config: " + "-"*20)
    print(pprint(vars(args)))
    logger.info("-"*20 + "Task File Config: " + "-"*20)
    print(pprint(task_cfg))
    return task_cfg, args


def build_save_path(args):
    tag = f"{args.tag}"
    save_path = os.path.join(args.output_dir, tag)
    output_checkpoint_path = os.path.join(save_path, "pytorch_ckpt_latest.tar")
    return save_path, output_checkpoint_path


def send_to(batch_dict, device):
    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor):
            batch_dict[key] = value.cuda(device=device, non_blocking=True)
        if isinstance(value, dict):
            for k,v in value.items():
                batch_dict[key][k] = v.cuda(device=device, non_blocking=True)


def main():
    task_cfg, args = get_parser_cfg()
    # Todo: Fix num_epochs
    assert task_cfg['num_epoch'] == args.num_train_epochs
    model_type = task_cfg["model_type"] if args.model_type is None else args.model_type
    savePath, output_checkpoint_path = build_save_path(args)

    # running only in evaluation mode
    if args.only_eval:
        return args.task_file, output_checkpoint_path, args.use_share2

    if model_type == "m4c_spatial":
        logger.info("Using M4C-Spatial model")
        from vilbert.m4c_spatial import BertConfig, M4C
    else:
        logger.info("Did not recognize model!")
        raise ValueError

    base_lr = task_cfg["lr"]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device}, Num. GPUs: {n_gpu}")

    # logger.info(
    #     "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #         a, n_gpu, bool(args.local_rank != -1), args.fp16
    #     )
    # )

    # Apply gradient-clipping!
    if task_cfg["grad_clip_mode"] == "all":
        logger.info(f"Using gradients clipping mode: {task_cfg['grad_clip_mode']}")

    if not os.path.exists(savePath):
        os.makedirs(savePath)


    dataloaders = load_datasets(args, task_cfg, ["train", "val"])
    logdir = os.path.join(savePath, "logs")

    # Todo: Add new logger
    # tbLogger = utils.tbLogger(
    #     logdir,
    #     savePath,
    #     task_names,
    #     task_ids,
    #     task_num_iters,
    #     args.gradient_accumulation_steps,
    # )

    # if args.visual_target == 0:
    #     config.v_target_size = 1601
    #     config.visual_target = args.visual_target
    # else:
    #     config.v_target_size = 2048
    #     config.visual_target = args.visual_target

    # if args.task_specific_tokens:
    #     config.task_specific_tokens = True

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    # task_ave_iter = {}
    # # task_stop_controller = {}
    # for task_id, num_iter in task_num_iters.items():
    #     task_ave_iter[task_id] = int(
    #         task_cfg["num_epoch"]
    #         * num_iter
    #         * args.train_iter_multiplier
    #         / args.num_train_epochs
    #     )
    #
    # task_ave_iter_list = sorted(task_ave_iter.values())
    median_num_iter = len(dataloaders["train"])

    # Build config
    mmt_config = BertConfig.from_dict(task_cfg["M4C"])
    text_bert_config = BertConfig.from_dict(task_cfg["TextBert"])
    model = M4C(mmt_config, text_bert_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training Parameters: {trainable_params}")
    # LOAD LOSSES
    # Todo: Just use the loss from dictionary in task_utils
    # task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))
    optimizer_grouped_parameters = model.get_optimizer_parameters(base_lr)
    # Todo: what is this?
    print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    optimizer, warmup_scheduler = get_optim_scheduler(task_cfg, optimizer_grouped_parameters, base_lr)

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
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
        global_step = checkpoint["global_step"]
        start_epoch = int(checkpoint["epoch_id"]) + 1
        tbLogger = checkpoint["tb_logger"]
        del checkpoint

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print("***** Running training *****")
    print("  Num Iters: ", len(dataloaders["train"]))
    print("  Batch size: ", task_cfg["batch_size"])

    # task_iter_train = {name: None for name in task_ids}
    # task_count = {name: 0 for name in task_ids}

    # if start_epoch >= args.num_train_epochs:
    #     logger.info("Resetting Train Epochs to 0")
    #     start_epoch = 0

    # This validation score is used for model-saving.
    best_val_epoch, best_val_score = -1, 0

    if task_cfg["debug"]:
        median_num_iter = 2

    start_iter_id = 0
    global_step = 0
    start_epoch = 0

    loss_values = []
    # TRAINING LOOP
    import pdb
    pdb.set_trace()
    for epoch_id in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"):
        model.train()
        for step, batch in tqdm(enumerate(dataloaders["train"]), desc="Iters"):
            send_to(batch, device)
            iterId = start_iter_id + step + (epoch_id * median_num_iter)
            loss, score = ForwardModelsTrain(
                task_cfg,
                model,
                batch
            )
            loss.backward()
            if task_cfg["grad_clip_mode"] == "all":
                clip_gradients(model, task_cfg["max_grad_norm"], task_cfg["grad_clip_mode"])

            warmup_scheduler.step()
            model.zero_grad()
            # if first_task:
            #     global_step += 1
            #     first_task = False
            #
            # if default_gpu:
            #     tbLogger.step_train(
            #         epochId,
            #         iterId,
            #         float(loss),
            #         float(score),
            #         optimizer.param_groups[0]["lr"],
            #         task_id,
            #         "train",
            #     )

            if step % 20 == 0 and step != 0:
                # tbLogger.showLossTrain()
                print(f"Loss: {sum(loss_values)/len(loss_values)}")
                loss_values = []
                if step % (100 * args.gradient_accumulation_steps) == 0:
                    logger.info(f"LR rates: {[grp['lr'] for grp in optimizer.param_groups]}")

            # decided whether to evaluate on each tasks.
            if (iterId != 0 and iterId % median_num_iter == 0) or (
                    epoch_id == args.num_train_epochs - 1 and step == median_num_iter - 1
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
                    tbLogger,
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
                    best_val_epoch = epochId
                    torch.save(
                        {
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                            "global_step": global_step,
                            "epoch_id": epoch_id,
                            "tb_logger": tbLogger,
                        },
                        output_checkpoint,
                    )

    tbLogger.txt_close()
    del model
    print(f"Best Validation Score: {best_val_score}, Best Validation Epoch: {best_val_epoch}")
    return args.task_file, output_checkpoint, args.use_share2


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
    with torch.no_grad():
        for i, batch in enumerate(task_dataloader_val[task_id]):
            loss, score, batch_size = ForwardModelsVal(
                args, task_cfg, device, task_id, batch, model, task_losses
            )
            tbLogger.step_val(
                epochId, float(loss), float(score), task_id, batch_size, "val"
            )
            if default_gpu:
                sys.stdout.write("%d/%d\r" % (i, len(task_dataloader_val[task_id])))
                sys.stdout.flush()

    score = tbLogger.showLossVal(task_id, task_stop_controller=None)
    model.train()
    return score


if __name__ == "__main__":
    try:
        task_file_path, output_checkpoint_path, use_share2 = main()
        assert os.path.exists(task_file_path)
        assert os.path.exists(output_checkpoint_path)

        evaluator = Evaluator(
            task_file=task_file_path,
            config_file="config/spatial_m4c_mmt_textvqa.json",
            batch_size=64,
            model_ckpt=output_checkpoint_path,
            short_eval=False,
            use_share2=False,
        )
        for beam_size in [1, 5]:
            for split in ["val", "test"]:
                evaluator.load_split(split, beam_size)
                evaluator.evaluate()
    finally:
        import os
        os.system("watch -n 1 session-quit-error")
