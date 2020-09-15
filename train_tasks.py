# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bisect
import json
import logging
import os
import random
import sys
from collections import defaultdict
from io import open

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from evaluator import final_evaluate
from tools.registry import registry
from vilbert.metrics import get_consistency_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


def get_config():
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
        default=5,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--tasks", default="", type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--hard_stop",
        type=int,
        required=True
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

    args = parser.parse_args()
    with open(args.task_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    set_seeds(task_cfg)
    registry.update(task_cfg)

    logger.info("-" * 20 + "Config Start" + "-" * 20)
    print(json.dumps(vars(args), indent=2))
    print(json.dumps(vars(task_cfg), indent=2))
    logger.info("-" * 20 + "Config End" + "-" * 20)

    return task_cfg, args


def set_seeds(task_cfg):
    seed = task_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    task_cfg, args = get_config()

    from vilbert.task_utils import (
        LoadDatasets,
        LoadLosses,
        ForwardModelsTrain,
        clip_gradients,
        get_optim_scheduler)

    from vilbert.m4c import BertConfig, M4C
    model_type = task_cfg["model_type"]

    # task_names = []
    # task_lr = []
    # for i, task_id in enumerate(args.tasks.split("-")):
    #     task = "TASK" + task_id
    #     name = task_cfg[task]["name"]
    #     task_names.append(name)
    #     task_lr.append(task_cfg[task]["lr"])

    base_lr = task_cfg["lr"]
    # loss_scale = {}
    # for i, task_id in enumerate(args.tasks.split("-")):
    #     task = "TASK" + task_id
    #     loss_scale[task] = task_lr[i] / base_lr

    timeStamp = (args.config_file.split("/")[1].split(".")[0] + f"-{args.tag}")
    savePath = os.path.join(args.output_dir, timeStamp)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        raise ValueError

    if task_cfg["grad_clip_mode"] == "all":
        logger.info(f"Using gradients clipping mode: {task_cfg['grad_clip_mode']}")

    # (YK): default_gpu decides whether to log outputs
    default_gpu = True

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

    # # LOAD DATASETS
    dataloaders = LoadDatasets(task_cfg)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_train_optimization_steps = 1000 * task_cfg['num_epoch']

    mmt_config = BertConfig.from_dict(task_cfg["MMT"])
    # text_bert_config = "config/m4c_textbert_vqa.json"
    # text_bert_config = BertConfig.from_json_file(text_bert_config)
    text_bert_config = BertConfig.from_dict(task_cfg["TextBERT"])

    model = M4C(mmt_config, text_bert_config)


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training Parameters: {trainable_params}")
    # LOAD LOSSES
    from vilbert.task_utils import LossMap
    task_loss = LossMap[task_cfg["loss"]]
    optimizer_grouped_parameters = model.get_optimizer_parameters(base_lr)

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    optimizer, warmup_scheduler, lr_scheduler, lr_scheduler_config, warmpu_steps = get_optim_scheduler(
        task_cfg, optimizer_grouped_parameters, num_train_optimization_steps, base_lr)

    global_step = 0
    start_epoch = 0

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    best_val_score = 0
    best_val_loss = np.inf
    best_val_epoch = 0
    grad_diff = []
    eval_iter_factor = task_cfg.get("eval_iter_factor", 1500)

    # list of (iter, vqa_score, cs_score, cs_score_bt)
    best_checkpoints = [(-1, -1, -1, -1)]
    cs_checkpoints = [(-1, -1, -1, -1)]
    cs_bt_checkpoints = [(-1, -1, -1, -1)]

    ckpts_log_file = os.path.join(savePath, "ckpts.log")

    loss_hist, score_hist = [], []
    # TRAINING LOOP
    for epochId in tqdm(range(start_epoch, task_cfg['num_epoch']), desc="Epoch"):
        model.train()
        for step in tqdm(range(1000), desc="Iters"):

            if global_step > args.hard_stop:
                logger.info(f"Breaking w/ hard-stop at {args.hard_stop}")
                break

            iterId = step + (epochId * 1000)
            train_type = "ce" if registry.use_freq == "ce" else "scl"

            if registry.alt_train and registry.use_freq == "ce" \
                and iterId % registry.ce_freq == 1:
                train_type = "scl"

            if registry.alt_train and registry.use_freq == "scl" \
                and iterId % registry.scl_freq == 1:
                train_type = "ce"

            loss, score = ForwardModelsTrain(
                device,
                dataloaders,
                model,
                train_type
            )

            loss.backward()

            if task_cfg["grad_clip_mode"] == "all":
                clip_gradients(model, task_cfg["max_grad_norm"], task_cfg["grad_clip_mode"])

            optimizer.step()
            warmup_scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()

            if train_type == "ce" or (not registry.alt_train):
                loss_hist.append(float(loss))
                score_hist.append(float(score))

            del loss
            del score

            if (
                    step % (20 * 1) == 0
                    and step != 0
                    and default_gpu
            ):
                # tbLogger.showLossTrain()
                logger.info(f"Score: {sum(score_hist)/len(score_hist)}, Loss: {sum(loss_hist)/len(loss_hist)}")
                logger.info(f"LR rates: {[grp['lr'] for grp in optimizer.param_groups]}, "
                            f"Grad Diff: {sum(grad_diff) / len(grad_diff) if len(grad_diff) > 0 else None}")
                grad_dots, grad_diff = [], []
                loss_hist, score_hist = [], []

            # if (iterId != 0 and iterId % eval_iter_factor == 0) or (
            #         epochId == task_cfg['num_epoch'] - 1 and step == median_num_iter - 1
            # ) or (global_step == args.hard_stop):
            #     logger.info("Starting Validation Run....")
            #     curr_val_score, curr_val_loss, cs_scores, cs_bt_scores = evaluate(
            #         args,
            #         task_dataloader_val,
            #         None,
            #         task_cfg,
            #         device,
            #         task_id,
            #         model,
            #         task_losses,
            #         epochId,
            #         default_gpu,
            #         tbLogger,
            #     )
            #
            #     if default_gpu:
            #         # Save a trained model
            #         logger.info("** ** * Saving fine - tuned model ** ** * ")
            #         model_to_save = (
            #             model.module if hasattr(model, "module") else model
            #         )  # Only save the model it-self
            #
            #         checkpoint_dict = \
            #             {
            #                     "model_state_dict": model_to_save.state_dict(),
            #                     "optimizer_state_dict": optimizer.state_dict(),
            #                     "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
            #                     "global_step": global_step,
            #                     "epoch_id": epochId,
            #                     "tb_logger": tbLogger
            #             }
            #
            #         if task_cfg.get("monitor_value","val_score") == "val_score" and cs_scores is None:
            #             save_thresh = registry.save_top
            #             top_vqa_scores = [ckpt[1] for ckpt in sorted(best_checkpoints, key=lambda x: x[1])][
            #                              -save_thresh:]
            #             vqa_rank = bisect.bisect(top_vqa_scores, curr_val_score)
            #             checkpoint_dict.update(
            #                 {
            #                     "vqa_rank": vqa_rank,
            #                     "vqa_acc": curr_val_score,
            #                 }
            #             )
            #
            #             if vqa_rank != 0:
            #                 logger.info(f"Saving for VQA score w/ rank: {vqa_rank}")
            #                 output_checkpoint = os.path.join(savePath, f"vqa_rank_{vqa_rank}.tar")
            #                 torch.save(checkpoint_dict, output_checkpoint)
            #
            #             logger.info(
            #                 f"Current Val Score and Rank: {curr_val_score} / {vqa_rank} | Previous Best Val Scores: {top_vqa_scores}")
            #
            #             best_checkpoints.append((global_step, curr_val_score, -1, -1))
            #
            #             try:
            #                 with open(ckpts_log_file, "w") as outfile:
            #                     dump_str = ""
            #                     for ckpt in best_checkpoints:
            #                         dump_str += f"Ckpt: {ckpt} \n"
            #                     outfile.write(dump_str)
            #                 logger.info(f"Dumped File: {ckpts_log_file}")
            #             except:
            #                 import pdb
            #                 pdb.set_trace()
            #             best_val_score = curr_val_score
            #             best_val_epoch = epochId
            #
            #         elif curr_val_loss <= best_val_loss and task_cfg.get("monitor_value",
            #                                                                        "val_loss") == "val_loss" and cs_scores is None:
            #             output_checkpoint = os.path.join(savePath, "pytorch_ckpt_latest.tar")
            #             logger.info(f"Saving Checkpoint: {output_checkpoint}")
            #             logger.info(
            #                 f"Current Validation Loss: {curr_val_loss} | Previous Best Validation Loss: {best_val_loss}")
            #             best_val_loss = curr_val_loss
            #             torch.save(checkpoint_dict, output_checkpoint)
            #
            #         elif cs_scores is not None:
            #             try:
            #                 save_thresh = registry.save_top
            #                 curr_cs_score = cs_scores[-1]
            #                 curr_cs_bt_score = cs_bt_scores[-1]
            #
            #                 top_vqa_scores = [ ckpt[1] for ckpt in sorted(best_checkpoints, key=lambda x: x[1])][-save_thresh:]
            #                 top_cs_scores = [ ckpt[2] for ckpt in sorted(best_checkpoints, key=lambda x: x[2])][-save_thresh:]
            #                 top_cs_bt_scores = [ckpt[3] for ckpt in sorted(best_checkpoints, key=lambda x: x[3])][-save_thresh:]
            #
            #                 vqa_rank = bisect.bisect(top_vqa_scores, curr_val_score)
            #                 cs_rank = bisect.bisect(top_cs_scores, curr_cs_score)
            #                 cs_bt_rank = bisect.bisect(top_cs_bt_scores, curr_cs_bt_score)
            #
            #
            #                 logger.info(f"Current Val Score and Rank: {curr_val_score} / {vqa_rank} | Previous Best Val Scores: {top_vqa_scores}")
            #                 logger.info(f"Current CS Score and Rank: {curr_cs_score} / {cs_rank} | Previous Best CS Scores: {top_cs_scores}")
            #                 logger.info(f"Current CS BT Score and Rank: {curr_cs_bt_score} / {cs_bt_rank} | Previous Best CS BT Scores: {top_cs_bt_scores}")
            #
            #                 logger.info(f"Top CS checkpoint: {sorted(best_checkpoints, key=lambda x: x[2])[-1]}")
            #                 logger.info(f"Top CS BT checkpoint: {sorted(best_checkpoints, key=lambda x: x[3])[-1]}")
            #                 logger.info(f"Checkpoint Dir: {savePath}")
            #                 checkpoint_dict.update(
            #                     {
            #                         "vqa_rank": vqa_rank,
            #                         "cs_rank": cs_rank,
            #                         "cs_bt_rank": cs_bt_rank,
            #                         "vqa_acc": curr_val_score,
            #                         "cs_score": curr_cs_score,
            #                         "cs_bt_score": curr_cs_bt_score
            #                     }
            #                 )
            #
            #                 if vqa_rank != 0:
            #                     logger.info(f"Saving for VQA score w/ rank: {vqa_rank}")
            #                     output_checkpoint = os.path.join(savePath, f"vqa_rank_{vqa_rank}.tar")
            #                     torch.save(checkpoint_dict, output_checkpoint)
            #
            #                 if cs_rank != 0:
            #                     logger.info(f"Saving for CS score w/ rank: {cs_rank}")
            #                     output_checkpoint = os.path.join(savePath, f"cs_rank_{cs_rank}.tar")
            #                     torch.save(checkpoint_dict, output_checkpoint)
            #
            #                 if cs_bt_rank != 0:
            #                     logger.info(f"Saving for CS BT score w/ rank: {cs_bt_rank}")
            #                     output_checkpoint = os.path.join(savePath, f"cs_bt_rank_{cs_bt_rank}.tar")
            #                     torch.save(checkpoint_dict, output_checkpoint)
            #
            #                 best_checkpoints.append((global_step, curr_val_score, curr_cs_score, curr_cs_bt_score))
            #                 cs_checkpoints.append(cs_scores)
            #                 cs_bt_checkpoints.append(cs_bt_scores)
            #
            #                 with open(ckpts_log_file, "w") as outfile:
            #                     dump_str = ""
            #                     assert len(best_checkpoints) == len(cs_bt_checkpoints) == len(cs_checkpoints)
            #                     for ckpt, cs_ckpt, cs_bt_ckpt in zip(best_checkpoints,
            #                                                          cs_checkpoints,
            #                                                          cs_bt_checkpoints):
            #                         dump_str += f"Ckpt: {ckpt} | CS : {cs_ckpt}| CS-BT: {cs_bt_ckpt} \n"
            #                     outfile.write(dump_str)
            #                 logger.info(f"Dumped File: {ckpts_log_file}")
            #             except:
            #                 import pdb
            #                 pdb.set_trace()

        if global_step > args.hard_stop:
            break

    # tbLogger.txt_close()

    # Run final-evaluation for EvalAI file.
    for split in ["test", "val"]:
        final_evaluate(
            args, task_cfg, device, task_ids[0], model, task_losses, savePath, split
        )

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
