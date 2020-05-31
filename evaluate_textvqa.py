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
from vilbert.task_utils import (
    load_datasets,
    LoadLosses,
    ForwardModelsTrain,
    ForwardModelsVal,
    clip_gradients,
    get_optim_scheduler
)

from vilbert.datasets.textvqa_metrics import TextVQAAccuracyEvaluator, EvalAIAnswerProcessor, STVQAANLSEvaluator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# A bunch of ugly global variables!

logger = logging.getLogger(__name__)

project_dir = '/srv/share/ykant3/common/vilbert-multi-task/'

val_data_path = {
    "textvqa": '/srv/share3/hagrawal9/project/m4c/data/m4c_textvqa/imdb_val_ocr_en.npy',
    "textvqa_test": '/srv/share3/hagrawal9/project/m4c/data/m4c_textvqa/imdb_test_ocr_en.npy',
    "stvqa": '/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_val.npy',
    "stvqa_test": '/srv/share/ykant3/scene-text/test/imdb/test_task3_response_meta_fixed_processed.npy',
}

ocr_features_path = {
    "textvqa": '/srv/share/ykant3/vilbert-mt/features/ocr/train/',
    "stvqa": '/srv/share/ykant3/scene-text/features/ocr/train/train_task/',
    "textvqa_test": '/srv/share/ykant3/vilbert-mt/features/ocr/test/',
    "stvqa_test": '/srv/share/ykant3/scene-text/features/ocr/test/test_task3/',

}

images_path = {
    "stvqa_test": '/srv/share/ykant3/scene-text/test/test_task3/',
    "stvqa": '/srv/share/ykant3/scene-text/train/train_task/'
}

vocab_paths = {
    "stvqa": "/srv/share/ykant3/m4c-release/data/m4c_vocabs/stvqa/fixed_answer_vocab_stvqa_5k.txt",
    "textvqa": "/srv/share/ykant3/m4c-release/data/m4c_vocabs/textvqa/fixed_answer_vocab_textvqa_5k.txt"
}

vqa_evaluator = TextVQAAccuracyEvaluator()
anls_evaluator = STVQAANLSEvaluator()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
n_gpu = torch.cuda.device_count()

eval_df_path_tvqa = "/srv/share/ykant3/vilbert-mt/eval/tvqa_eval_df.pkl"
eval_df_path_tvqa_test = "/srv/share/ykant3/vilbert-mt/eval/tvqa_eval_df_test.pkl"
eval_df_path_stvqa = "/srv/share/ykant3/vilbert-mt/eval/stvqa_eval_df.pkl"
eval_df_path_stvqa_test = "/srv/share/ykant3/vilbert-mt/eval/stvqa_eval_df_test.pkl"


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
        "--short_eval", default=False, type=bool, help="Run only three iterations of val "
    )
    parser.add_argument(
        "--split", default="val", type=str, help="Which split do you want to evaluate"
    )

    parser.add_argument(
        "--use_share2", default=False, type=bool, help="Which split do you want to evaluate"
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
        'num_workers': 8,
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
    model_type = task_config["TASK19"]["model_type"]

    if model_type == "m4c":
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
    elif model_type == "m4c_topk":
        logger.info("Using M4C-Topk model")
        from vilbert.m4c_topk import BertConfig, M4C
    elif model_type == "m4c_topk_20x":
        logger.info("Using M4C-Topk model")
        from vilbert.m4c_topk_20x import BertConfig, M4C
    elif model_type == "m4c_regat":
        logger.info("Using M4C-Regat model")
        from vilbert.m4c_regat import BertConfig, M4C
    elif model_type == "m4c_regat_spatial":
        logger.info("Using M4C-Regat_spatial model")
        from vilbert.m4c_regat_spatial import BertConfig, M4C
    elif model_type == "m4c_spatial_que_cond":
        logger.info("Using M4C-Spatial Question Cond. model")
        from vilbert.m4c_spatial_que_cond import BertConfig, M4C
    else:
        raise ValueError

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
                     "mix_list"
                     ]
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


def load_data_for_evaluation_tvqa(tag):
    # This merges all the data sources into one nice DataFrame
    # We can use this DataFrame for all sorts of complex queries.

    val_data = np.load(val_data_path[tag], allow_pickle=True)
    val_data_list = val_data[1:].tolist()
    val_data_df = pd.DataFrame(val_data_list)

    google_ocr_data = []
    for f in tqdm(os.listdir(ocr_features_path[tag]), total=len(os.listdir(ocr_features_path[tag]))):
        image_id = f.split('.')[0]
        if image_id in val_data_df['image_id'].tolist():
            d = np.load(os.path.join(ocr_features_path[tag], f), allow_pickle=True)
            google_ocr_data.append(d.tolist())

    ocr_df = pd.DataFrame(google_ocr_data)
    val_data_with_ocr_df = pd.merge(val_data_df, ocr_df, how='left', on='image_id')

    return val_data_with_ocr_df

def load_data_for_evaluation_stvqa(tag):
    # This merges all the data sources into one nice DataFrame
    # We can use this DataFrame for all sorts of complex queries.
    val_data = np.load(val_data_path[tag], allow_pickle=True)
    val_data_list = val_data[1:].tolist()
    val_data_df = pd.DataFrame(val_data_list)
    google_ocr_data = []

    for instance in tqdm(val_data_list):
        feature_path = instance["image_path"].replace(images_path[tag], ocr_features_path[tag]).split(".")[0] + ".npy"
        assert os.path.exists(feature_path)
        feature_data = np.load(feature_path, allow_pickle=True).tolist()
        google_ocr_data.append(feature_data)

    ocr_df = pd.DataFrame(google_ocr_data)
    assert len(ocr_df) == len(val_data_df)
    val_data_with_ocr_df = pd.merge(val_data_df, ocr_df, how='left', left_index=True, right_index=True)
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
        'complete_seqs': [],
        # 'ocr_tokens': []
    }
    scores = 0.0
    data_size = 0
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(task_dataloader_val[task_id]):

            # Don't bother for loss (beam-search changes output)
            ForwardModelsVal(
                args, task_cfg, device, task_id, batch, model, task_losses,
            )

            save_keys = ['question_id', 'topkscores', 'complete_seqs']

            # Shapes:
            # topk-scores: (bs, dec_steps, beam_size)
            # complete-seqs: (bs, beam_size, dec_steps)
            for key in save_keys:
                # if key == "ocr_tokens":
                #     from tools.objects_to_byte_tensor import dec_bytes2obj
                #     batch[key] = [dec_bytes2obj(x) for x in batch[key]]
                predictions[key].append(batch[key])

            sys.stdout.write("%d/%d\r" % (i, len(task_dataloader_val[task_id])))
            sys.stdout.flush()

            if args.short_eval and i == 1:
                break

    return predictions


def vqa_calculate(batch_dict):
    pred_answers = batch_dict['pred_answers']
    ocr_tokens_enc = batch_dict["ocr_tokens"]
    gt_answers_enc = batch_dict["answers"]
    topkscores = batch_dict['topkscores']
    answer_space_size = len(registry["vocab"])

    predictions = []

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

    accuracy, pred_scores = vqa_evaluator.eval_pred_list(predictions)
    return {
        'question_id': predictions[0]['question_id'],
        'accuracy': accuracy,
        'pred_answer': predictions[0]['pred_answer'],
        'belongs_to': predictions[0]['belongs_to'],
        'answer_words': predictions[0]['answer_words'],
        'topkscores': predictions[0]['topkscores']
    }


def anls_calculate(batch_dict):
    pred_answers = batch_dict['pred_answers']
    ocr_tokens_enc = batch_dict["ocr_tokens"]
    gt_answers_enc = batch_dict["answers"]
    topkscores = batch_dict['topkscores']
    answer_space_size = len(registry["vocab"])

    predictions = []

    # TODO: This is a single list - remove for loop
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

    try:
        accuracy, pred_scores = anls_evaluator.eval_pred_list(predictions)
    except:
        import pdb
        pdb.set_trace()

    return {
        'question_id': predictions[0]['question_id'],
        'accuracy': accuracy,
        'pred_answer': predictions[0]['pred_answer'],
        'belongs_to': predictions[0]['belongs_to'],
        'answer_words': predictions[0]['answer_words'],
        'topkscores': predictions[0]['topkscores']
    }


def evaluate_predictions(eval_df, results_df, acc_type="vqa", tokens_from="vd"):

    if acc_type == "vqa":
        calculate = vqa_calculate
    elif acc_type == "anls":
        calculate = anls_calculate
    else:
        raise AssertionError

    predictions = []
    for i in range(results_df.shape[0]):
        re = results_df.iloc[i]
        question_id = re.question_id
        vd = eval_df[eval_df['question_id'] == question_id].iloc[0]

        tokens_key = "ocr_tokens_y"
        if tokens_key not in eval_df:
            tokens_key = "ocr_tokens"
            assert tokens_key in eval_df

        if tokens_from == "re":
            tokens = re[tokens_key]
        else:
            tokens = vd[tokens_key]

        batch = {
            'question_id': re.question_id,
            'answers': [vd.answers],
            'ocr_tokens': [tokens],
            'topkscores': [re.topkscores],
            'pred_answers': np.array([re.complete_seqs[1:]])
        }

        predictions.append(calculate(batch))

    accuracies_df = pd.DataFrame(predictions)
    best_result = []
    oracle_accuracies = 0.0
    for qid, row in accuracies_df.groupby('question_id'):
        idx = np.argmax(row.topkscores)
        # idx = np.random.randint(row.topkscores.shape[0])
        oracle_accuracies += row.accuracy.tolist()[idx]
        best_result.append(row.iloc[idx])

    best_result_df = pd.DataFrame(best_result)
    mean_acc = oracle_accuracies / accuracies_df['question_id'].unique().shape[0]
    return {
        "vqa_accuracy": mean_acc,
        "accuracies_df": accuracies_df,
        "best_result_df": best_result_df
    }


def main():
    args = parse_args()
    registry['args'] = args

    # Load task config
    with open(args.task_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    registry['task_config'] = task_cfg


    # Load Dataset
    return_tuple = load_datasets(args, task_cfg, args.tasks.split("-"),
                                 split=args.split,
                                 only_val=True,
                                 test_val_bs=64,
                                 test_val_workers=8)

    if "test" in args.split:
        (
            task_batch_size,
            task_num_iters,
            task_ids,
            task_datasets_train,
            task_datasets_val,
            task_datasets_test,
            task_dataloader_train,
            task_dataloader_val,
            task_dataloader_test
        ) = return_tuple
    else:
        (
            task_batch_size,
            task_num_iters,
            task_ids,
            task_datasets_train,
            task_datasets_val,
            task_dataloader_train,
            task_dataloader_val
        ) = return_tuple

    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))

    # tvqa_eval_df = load_data_for_evaluation_tvqa()
    # tvqa_eval_df_test = load_data_for_evaluation_tvqa_test()
    # stvqa_eval_df = load_data_for_evaluation_stvqa("stvqa")
    # stvqa_eval_df_test = load_data_for_evaluation_stvqa("stvqa_test")

    # pd.to_pickle(tvqa_eval_df, eval_df_path_tvqa)
    # pd.to_pickle(tvqa_eval_df_test, eval_df_path_tvqa_test)
    # pd.to_pickle(stvqa_eval_df, eval_df_path_stvqa)
    # pd.to_pickle(stvqa_eval_df_test, eval_df_path_stvqa_test)


    if task_cfg["TASK19"]["val_on"][0] == "textvqa":
        vocab_path = vocab_paths["textvqa"]
        path = eval_df_path_tvqa
        if args.split == "test":
            path = eval_df_path_tvqa_test
        eval_df = pd.read_pickle(path)
    elif task_cfg["TASK19"]["val_on"][0] == "stvqa":
        vocab_path = vocab_paths["stvqa"]
        path = eval_df_path_stvqa
        if args.split == "test":
            path = eval_df_path_stvqa_test
        eval_df = pd.read_pickle(path)
    else:
        raise AssertionError

    # Build vocab
    vocab = []
    with open(vocab_path) as f:
        for line in f.readlines():
            vocab.append(line.strip())
    registry['vocab'] = vocab
    logger.info(f"Using vocab from: {vocab_path}")


    # Load Model
    model = load_model()

    for split in ["test", "val"]:

        if split not in args.split:
            continue

        if "test" in split:
            dataloader = task_dataloader_test
        elif "val" in split:
            dataloader = task_dataloader_val

        logger.info(f"Processing split: {split}")
        if split in args.split:
            # Run evaluation
            curr_score = evaluate(
                args,
                dataloader,
                None,
                task_cfg,
                device,
                "TASK19",
                model,
                task_losses
            )

            curr_score['complete_seqs'] = np.concatenate(
                [x.reshape(-1, 12) for x in curr_score['complete_seqs']], axis=0
            ).tolist()
            curr_score['topkscores'] = np.concatenate(curr_score['topkscores'], axis=0).tolist()
            curr_score['question_id'] = np.concatenate(curr_score['question_id'], axis=0).tolist()
            # curr_score['ocr_tokens'] = np.concatenate(curr_score['ocr_tokens'], axis=0).tolist()

            if 'answers' not in eval_df:
                eval_df["answers"] = [["none"]*10]*len((eval_df["question_id"]))

            # Compute VQA Accuracies
            results_df = pd.DataFrame.from_dict(curr_score, orient='columns')

            # # Get both type of accuracies!
            # if task_cfg["TASK19"]["val_on"][0] == "stvqa":
            #     accuracies = evaluate_predictions(eval_df, results_df, acc_type="vqa", tokens_from="re")
            # else:
            accuracies = evaluate_predictions(eval_df, results_df, acc_type="vqa")


            if "test" not in split:
                logger.info(
                    "{} Accuracy: {} for {} questions, split {}, dataset {}".format(
                        "vqa",
                        accuracies['vqa_accuracy'],
                        accuracies['accuracies_df'].shape,
                        args.split,
                        task_cfg["TASK19"]["val_on"][0])
                )

                accuracies = evaluate_predictions(eval_df, results_df, acc_type="anls")
                logger.info(
                    "{} Accuracy: {} for {} questions, split {}, dataset {}".format(
                        "anls",
                        accuracies['vqa_accuracy'],
                        accuracies['accuracies_df'].shape,
                        args.split,
                        task_cfg["TASK19"]["val_on"][0])
                )

            evalai_file = os.path.join(os.path.dirname(args.model_ckpt),'run2_{}_evalai_beam_{}_short_eval_{}_share2_{}.json'.
                                       format(split, args.beam_size, args.short_eval, args.use_share2))
            df_file = os.path.join(os.path.dirname(args.model_ckpt),'run2_{}_evalai_beam_{}_short_eval_{}_share2_{}.df'.
                                       format(split, args.beam_size, args.short_eval, args.use_share2))

            # Accuracies DF
            accuracies['accuracies_df'].to_json(df_file)

            # EvalAI/ST-VQA file
            answer_dict = []
            for i, pred in accuracies['best_result_df'].iterrows():
                answer_dict.append({
                    'question_id': pred['question_id'],
                    'answer': pred['pred_answer'].strip()
                })
            with open(evalai_file, 'w') as f:
                json.dump(answer_dict, f)
            print(f"Dumping file: {evalai_file}")

if __name__ == "__main__":
    main()
    # try:
    #     main()
    # finally:
    #     import pdb
    #     pdb.set_trace()
