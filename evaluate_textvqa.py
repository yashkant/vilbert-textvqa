import argparse
import sys
import os
import json
import random
import pdb
import logging
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import yaml

from tqdm import tqdm
from bisect import bisect
from easydict import EasyDict as edict
from tools.registry import registry

import vilbert.utils as utils
from vilbert.m4c_spatial import BertConfig, M4C
from vilbert.task_utils import (
    LoadDatasets,
    LoadLosses,
    ForwardModelsTrain,
    ForwardModelsVal,
    clip_gradients,
    get_optim_scheduler
)

from vilbert.datasets.textvqa_metrics import TextVQAAccuracyEvaluator, EvalAIAnswerProcessor
import multiprocessing
multiprocessing.set_start_method('spawn', True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# A bunch of ugly global variables! 

logger = logging.getLogger(__name__)

project_dir = '/srv/share/ykant3/common/vilbert-multi-task/'
val_data_path = '/srv/share3/hagrawal9/project/m4c/data/m4c_textvqa/imdb_val_ocr_en.npy'
obj_npy_file_path = '/srv/share/ykant3/vilbert-mt/features/obj/train/'
ocr_npy_file_path = '/srv/share/ykant3/vilbert-mt/features/ocr/train/'
evaluator = TextVQAAccuracyEvaluator()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
n_gpu = torch.cuda.device_count()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_file",
        type=str,
        help="Task file"
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        help="Full path to the vocab_file"
    )
    parser.add_argument(
        "--batch_size", default=30, type=int, help="what is the batch size?"
    )
    parser.add_argument(
        "--model_ckpt", type=str, help="full path to model checkpoint"
    )
    parser.add_argument(
        "--beam_size", type=int, help="number of beams"
    )
    parser.add_argument(
        "--save_file", default="", type=str, help="Path to save the results (if you want to save it)"
    )
    parser.add_argument(
        "--short_eval", default=False, type=bool, help="Run only three iterations of val "
    )
    
    command_args = parser.parse_args()

    # if default value is going to change, add those in command line arguments.
    args = edict({
        'bert_model': 'bert-base-uncased', 
        'val_df_file': '/srv/share3/hagrawal9/data/textvqa_val_googleocr_cocodetector.pkl',
        'tasks': '19',
        'do_lower_case': True, 
        'in_memory': True,
        'gradient_accumulation_steps': 1, 
        'num_workers': 0, 
        'local_rank': -1, 
        'clean_train_sets': False, 
        'num_train_epochs': 100,
        'train_iter_multiplier': 1.0,
        'is_running_validation': True
    })
    # Extremely important to set this. Controls the behavior of evaluation during training or not. 
    registry["is_running_validation"] = True
    args.update(vars(command_args))
    return args

def load_model():
    args = registry.get("args")
    task_config = registry.get("task_config")
    
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
    transfer_keys.extend(["aux_spatial_fusion", "use_aux_heads"])

    with open(args.config_file, "r") as file:
        config_dict = json.load(file)

    # Adding blank keys that could be dynamically replaced later
    config_dict["layer_type_list"] = None
    config_dict["beam_size"] = args.beam_size 
    config_dict["mix_list"] = None

    mmt_config = BertConfig.from_dict(config_dict)
    # Replace keys
    for key in transfer_keys:
        if key in task_config["TASK19"]:
            config_dict[key] = task_config["TASK19"][key]
            logger.info(f"Transferring keys:  {key}, {config_dict[key]}")
    mmt_config = BertConfig.from_dict(config_dict)

    text_bert_config = BertConfig.from_json_file("config/m4c_textbert_textvqa.json")
    model = M4C(mmt_config, text_bert_config)
    
    logger.info(f"Resuming from Checkpoint: {args.model_ckpt}")
    checkpoint = torch.load(args.model_ckpt, map_location="cpu")
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

    model = model.to(device)
    return model

def load_data_for_evaluation():
    # This merges all the data sources into one nice DataFrame
    # We can use this DataFrame for all sorts of complex queries. 

    val_data = np.load(val_data_path, allow_pickle=True)
    val_data_list = val_data[1:].tolist()
    val_data_df = pd.DataFrame(val_data_list)

    google_ocr_data = []
    for f in os.listdir(ocr_npy_file_path):
        image_id = f.split('.')[0]
        if image_id in val_data_df['image_id'].tolist():
            d = np.load(os.path.join(ocr_npy_file_path, f), allow_pickle=True)
            google_ocr_data.append(d.tolist())
            
    ocr_df = pd.DataFrame(google_ocr_data)
    val_data_with_ocr_df = pd.merge(val_data_df, ocr_df, how='left', on='image_id')

    return val_data_with_ocr_df

def evaluate(
    args,
    task_dataloader_val,
    task_stop_controller,
    task_cfg,
    device,
    task_id,
    model,
    task_losses
):

    predictions = {
        'question_id': [],
        'topkscores': [], 
        'complete_seqs': []
    }
    scores = 0.0
    data_size = 0
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(task_dataloader_val[task_id]):
            ForwardModelsVal(
                args, task_cfg, device, task_id, batch, model, task_losses
            )

            save_keys = ['question_id', 'topkscores', 'complete_seqs']
            for key in save_keys:
                predictions[key].append(batch[key])

            sys.stdout.write("%d/%d\r" % (i, len(task_dataloader_val[task_id])))
            sys.stdout.flush()
        
            # if args.short_eval and i == 3:
            #     break
    
    return predictions

def calculate(batch_dict):
    pred_answers = batch_dict['pred_answers']
    ocr_tokens_enc = batch_dict["ocr_tokens"]
    gt_answers_enc = batch_dict["answers"]
    topkscores = batch_dict['topkscores']
    answer_space_size = len(registry["vocab"])

    predictions = []
    
    #TODO: This is a single list - remove for loop
    for idx, question_id in enumerate([batch_dict["question_id"]]):
        context_tokens = ocr_tokens_enc[idx]
        answer_words = []
        belongs_to = []

        for answer_id in pred_answers[idx].tolist():
            if answer_id >= answer_space_size:
                belongs_to.append("ocr")
                answer_id -= answer_space_size
                answer_words.append(context_tokens[answer_id])
            else:
                if answer_id == registry['EOS_IDX']:
                    belongs_to.append("vocab+eos")
                    break
                belongs_to.append("vocab")
                answer_words.append(registry["vocab"][answer_id])

        answer = ' '.join(answer_words).replace(" 's", "'s")
        gt_answers = gt_answers_enc[idx]

        predictions.append({
            "question_id": question_id,
            "gt_answers": gt_answers,
            "pred_answer": answer,
            "belongs_to": belongs_to,
            "answer_words": answer_words,
            "topkscores": topkscores
        })

    accuracy, pred_scores = evaluator.eval_pred_list(predictions)
    return {
        'question_id': predictions[0]['question_id'],
        'accuracy': accuracy,
        'pred_answer': predictions[0]['pred_answer'],
        'belongs_to': predictions[0]['belongs_to'],
        'answer_words': predictions[0]['answer_words'],
        'topkscores': predictions[0]['topkscores']
    }

def evaluate_predictions(eval_df, results_df):
    predictions = []
    for i in range(results_df.shape[0]):
        r = results_df.iloc[i]
        question_id = r.question_id
        vd = eval_df[eval_df['question_id']==question_id].iloc[0]

        batch = {
            'question_id': r.question_id,
            'answers': [vd.answers],
            'ocr_tokens': [vd.ocr_tokens_y], 
            'topkscores': [r.topkscores],
            'pred_answers': np.array([r.complete_seqs[1:]])
        }

        predictions.append(calculate(batch))

    accuracies_df = pd.DataFrame(predictions)
    oracle_accuracies = 0.0
    for qid, row in accuracies_df.groupby('question_id'):
        idx = np.argmax(row.topkscores) 
        # idx = np.random.randint(row.topkscores.shape[0])
        oracle_accuracies += row.accuracy.tolist()[idx]
    
    mean_acc = oracle_accuracies / accuracies_df['question_id'].unique().shape[0]
    return {
        "vqa_accuracy": mean_acc,
        "accuracies_df": accuracies_df
    }

def main():
    args = parse_args()
    registry['args'] = args
    
    #Load task config
    with open(args.task_file, "r") as f:
            task_cfg = edict(yaml.safe_load(f))
    registry['task_config'] = task_cfg

    # Build vocab
    vocab = []
    with open(args.vocab_file) as f:
        for line in f.readlines():
            vocab.append(line.strip())
    registry['vocab'] = vocab

    # Load Dataset
    (
        task_batch_size, 
        task_num_iters, 
        task_ids, 
        task_datasets_train, 
        task_datasets_val, 
        task_dataloader_train, 
        task_dataloader_val
    ) = LoadDatasets(args, task_cfg, args.tasks.split("-"))
    
    # eval_df = load_data_for_evaluation()
    eval_df = pd.read_pickle(args.val_df_file)
    
    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))
    task = "TASK" + str(args.tasks)
    task_id = 19
    
    # Load Model
    model = load_model()

    # Run evaluation
    curr_val_score = evaluate(
        args,
        task_dataloader_val,
        None,
        task_cfg,
        device,
        task,
        model,
        task_losses
    )
    
    curr_val_score['complete_seqs'] = np.concatenate(
        [x.reshape(-1, 12) for x in curr_val_score['complete_seqs']], axis=0
    ).tolist()
    curr_val_score['topkscores'] = np.concatenate(curr_val_score['topkscores'], axis=0).tolist()
    curr_val_score['question_id'] = np.concatenate(curr_val_score['question_id'], axis=0).tolist()
    
    # Compute VQA Accuracies
    results_df = pd.DataFrame.from_dict(curr_val_score, orient='columns')
    # pd.to_pickle(results, '/srv/share3/hagrawal9/data/bs_results.pkl')

    accuracies = evaluate_predictions(eval_df, results_df)
    
    #  if not args.short_eval: 
    logger.info("VQA Accuracy: {} for {} questions".format(accuracies['vqa_accuracy'], accuracies['accuracies_df'].shape))
    
    if len(args.save_file) != 0:
        accuracies['accuracies_df'].to_json(args.save_file)

if __name__ == "__main__":
    main()
