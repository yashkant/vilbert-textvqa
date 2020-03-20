import json
import os
import sys
import numpy as np
sys.path.append("../../")
from vilbert.datasets.textvqa_metrics import TextVQAAccuracyEvaluator

evaluator = TextVQAAccuracyEvaluator()

# file_path = "../../save/TextVQA_spatial_m4c_mmt_textvqa-finetune_from_multi_task_model-m4c-spatial-mask-1-2-layers-4-share3-train-ts-val-t-run3/" \
#             "random_reln_val_evalai_beam_1_short_eval_False_share2_True.json"

file_path = "../../save/TextVQA_spatial_m4c_mmt_textvqa-finetune_from_multi_task_model-m4c-spatial-mask-1-2-layers-4-share3-train-ts-val-t-run3/" \
            "rev_reln_val_evalai_beam_1_short_eval_False_share2_True.json"

# file_path = "../../save/TextVQA_spatial_m4c_mmt_textvqa-finetune_from_multi_task_model-m4c-spatial-mask-1-2-layers-4-share3-train-ts-val-t-run3/" \
#             "random_reln_val_evalai_beam_1_short_eval_False_share2_True.json"



imdb_path = "../../datasets/textvqa/imdb/textvqa_0.5/" \
    "imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_val.npy"

dir_path = os.path.split(file_path)[0]
file_name = os.path.split(file_path)[1]

json_data = json.load(open(file_path))
imdb_data = np.load(imdb_path, allow_pickle=True)

def word_cleaner(word):
    word = word.lower()
    word = word.replace(",", "").replace("?", "").replace("'s", " 's")
    return word.strip()

# import pdb
# pdb.set_trace()


question_id_instance = {}
for instance in imdb_data[1:]:
    instance["gt_answers"] = [word_cleaner(ans) for ans in instance["answers"]]
    question_id_instance[instance["question_id"]] = instance

# import pdb
# pdb.set_trace()

predictions_list = []

for sample in json_data:
    predictions_list.append({
        "question_id": sample["question_id"],
        "pred_answer": sample["answer"],
        "gt_answers": question_id_instance[sample["question_id"]]["gt_answers"]
    })

# import pdb
# pdb.set_trace()

accuracy, pred_scores = evaluator.eval_pred_list(predictions_list)

for score, pred_dict in zip(pred_scores, predictions_list):
    pred_dict["vqa_score"] = score

import pdb
new_file_path = os.path.join(dir_path, file_name.split(".")[0] + "_scores.npy")
pdb.set_trace()
np.save(new_file_path, predictions_list)
