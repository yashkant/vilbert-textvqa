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
import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from vilbert.task_utils import (
    load_losses,
    forward_train,
    forward_val,
    clip_gradients,
    get_optim_scheduler, load_datasets)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

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
        "--tasks", default="", type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--freeze",
        default=-1,
        type=int,
        help="till which layer of textual stream of vilbert need to fixed.",
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
    print(task_cfg)
    logger.info("-"*20 + "Config End" + "-"*20)

    seed = task_cfg.get("seed", args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Using seed: {seed}")

    model_type = task_cfg["model_type"]

    task_names = []
    task_lr = []
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        name = task_cfg["name"]
        task_names.append(name)
        task_lr.append(task_cfg["lr"])

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
        save_path = os.path.join(args.output_dir, timeStamp)
        output_checkpoint_path = os.path.join(save_path, "pytorch_ckpt_latest.tar")
        return args.task_file, output_checkpoint_path, args.use_share2

    from vilbert.m4c_spatial import BertConfig, M4C

    base_lr = min(task_lr)
    save_path = os.path.join(args.output_dir, args.tag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device}, Numer of GPUs: {n_gpu}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = BertConfig.from_json_file(args.config_file)
    # save all the hidden parameters.
    with open(os.path.join(save_path, "command.txt"), "w") as f:
        print(args, file=f)  # Python 3.x
        print("\n", file=f)
        print(config, file=f)

    dataloaders = load_datasets(task_cfg, ["train"])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    median_num_iter = len(dataloaders["train"])
    # num_train_optimization_steps = (
    #         median_num_iter * args.num_train_epochs
    # )
    # LOAD PRETRAINED VILBERT
    # if args.from_scratch:
    #     if model_type in ["m4c_spatial", "m4c_topk", "m4c_regat", "m4c_regat_spatial"]:
    #         if "m4c_spatial" in model_type:
    #             assert "attention_mask_quadrants" in task_cfg
    #         # assert "spatial_type" in task_cfg
    #         # Transfer keys from config to BertConfig
    #         transfer_keys = ["attention_mask_quadrants",
    #                          "hidden_size",
    #                          "num_implicit_relations",
    #                          "spatial_type",
    #                          "num_hidden_layers",
    #                          "num_spatial_layers",
    #                          "layer_type_list",
    #                          "cond_type",
    #                          "use_bias",
    #                          "no_drop",
    #                          "mix_list"]
    #     elif model_type == "m4c" or model_type == "m4c_rd":
    #         # Transfer keys from config to BertConfig
    #         transfer_keys = ["num_hidden_layers"]
    #     else:
    #         raise ValueError
    #
    #     # Common keys
    #     transfer_keys.extend(["aux_spatial_fusion", "use_aux_heads"])
    #
    #     # Load config-file M4C
    #     with open(args.config_file, "r") as file:
    #         config_dict = json.load(file)
    #
    #     # Adding blank keys that could be dynamically replaced later
    #     config_dict["layer_type_list"] = None
    #     config_dict["mix_list"] = None
    #
    #     # Always use beam-size 1 for training
    #     config_dict["beam_size"] = 1
    #
    #     # Replace keys
    #     for key in transfer_keys:
    #         if key in task_cfg:
    #             config_dict[key] = task_cfg[key]
    #             logger.info(f"Transferring keys:  {key}, {config_dict[key]}")
    #     mmt_config = BertConfig.from_dict(config_dict)
    #     text_bert_config = BertConfig.from_json_file("config/m4c_textbert_textvqa.json")

    mmt_config = BertConfig.from_dict(task_cfg["SA-M4C"])
    text_bert_config = BertConfig.from_dict(task_cfg["TextBERT"])
    model = M4C(mmt_config, text_bert_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training Parameters: {trainable_params}")

    # LOAD LOSSES
    # task_losses = load_losses(task_cfg, args.tasks.split("-"))
    optimizer_grouped_parameters = model.get_optimizer_parameters(base_lr)
    print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    optimizer, warmup_scheduler = get_optim_scheduler(task_cfg, optimizer_grouped_parameters, base_lr)

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

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if task_cfg["debug"]:
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

            loss, score = forward_train(
                dataloaders,
                task_cfg,
                device,
                "TASK19",
                model,
            )
            loss_values.append(loss)
            score_values.append(score)
            loss.backward()
            clip_gradients(model, task_cfg["max_grad_norm"])
            optimizer.step()
            warmup_scheduler.step()
            model.zero_grad()
            global_step += 1

            if (
                    step % 20 == 0
                    and step != 0
            ):
                log_str = f"Batch: loss = {float(sum(loss_values)/len(loss_values))}; " \
                          f"accuracy  = {float(sum(score_values)/len(score_values))}; "
                loss_values, score_values = [], []
                if step % 100 == 0:
                    log_str += f"\n lr rates = {[float(grp['lr']) for grp in optimizer.param_groups]}"
                logger.info(log_str)

            # don't run validation during debug runs
            if not task_cfg["debug"] and (iterId != 0 and iterId % task_num_iters[task_id] == 0) or (
                    epochId == args.num_train_epochs - 1 and step == median_num_iter - 1
            ):
                curr_val_score = evaluate(
                    args,
                    task_dataloader_val,
                    task_cfg,
                    device,
                    task_id,
                    model,
                    task_losses,
                )

                # Save a trained model
                logger.info("** ** * Saving fine - tuned model ** ** * ")
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )
                if curr_val_score > best_val_score:
                    output_checkpoint = os.path.join(save_path, "pytorch_ckpt_latest.tar")
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
    task_cfg,
    device,
    task_id,
    model,
    task_losses,
):

    model.eval()
    for batch in tqdm(task_dataloader_val[task_id], desc="Validation"):
        loss, score, batch_size = forward_val(
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
