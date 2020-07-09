from collections import defaultdict
from copy import deepcopy

import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import _pickle as cPickle

que_split_path_dict = {
    "train": ("../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_train2014_questions.pkl",
              "../../datasets/VQA/v2_OpenEnded_mscoco_train2014_questions.json"),
    "val": ("../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
            "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json"),
    "test": ("../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_test2015_questions.pkl",
             "../../datasets/VQA/v2_OpenEnded_mscoco_test2015_questions.json"),
    "trainval": ("../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_trainval2014_questions.pkl",
             "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_minval2014_questions.pkl"),
}

ans_split_path_dict = {
    "train": ("../../datasets/VQA/back-translate/org2_bt_train_target.pkl",
              "../../datasets/VQA/cache/train_target.pkl"),
    "val": ("../../datasets/VQA/back-translate/org2_bt_val_target.pkl",
            "../../datasets/VQA/cache/val_target.pkl"),
    "test": ("../../datasets/VQA/back-translate/org2_bt_test_target.pkl",
             "../../datasets/VQA/cache/test_target.pkl"),
    "trainval": ("../../datasets/VQA/back-translate/org2_bt_trainval_target.pkl",
             "../../datasets/VQA/back-translate/org2_bt_minval_target.pkl"),
}


split_seq_holder = "../../datasets/VQA/back-translate/sim-result2/rep_{}_{}.npy"
max_seqs = 10
org_factor = int(1e10)


# Todo: Currently we are missing lanuages for original question fix that in final-run.
for split in ["train", "val", "trainval"]:

    # dump both trainval and minval here
    if split == "trainval":
        trainval_questions = cPickle.load(open(que_split_path_dict["train"][0], "rb"))["questions"] + \
                             cPickle.load(open(que_split_path_dict["val"][0], "rb"))["questions"]

        trainval_answers = cPickle.load(open(ans_split_path_dict["train"][0], "rb")) + \
                           cPickle.load(open(ans_split_path_dict["val"][0], "rb"))

        minval_questions, minval_answers = trainval_questions[-3000:], trainval_answers[-3000:]
        trainval_questions, trainval_answers = trainval_questions[:-3000], trainval_answers[:-3000]

        cPickle.dump(minval_answers, open(ans_split_path_dict[split][1], "wb"))
        cPickle.dump({"questions": minval_questions}, open(que_split_path_dict[split][1], "wb"))
        print(f"Dumped files: \n {que_split_path_dict[split][1]} \n {ans_split_path_dict[split][1]}")

        cPickle.dump(trainval_answers, open(ans_split_path_dict[split][0], "wb"))
        cPickle.dump({"questions": trainval_questions}, open(que_split_path_dict[split][0], "wb"))
        print(f"Dumped files: \n {que_split_path_dict[split][0]} \n {ans_split_path_dict[split][0]}")
        break

    que_data = json.load(open(que_split_path_dict[split][-1]))
    ans_data = cPickle.load(open(ans_split_path_dict[split][-1], "rb"))
    data_dict = defaultdict(dict)

    for qd, ad in zip(que_data["questions"], ans_data):
        data_dict[qd["question_id"]]["que_data"] = qd
        data_dict[ad["question_id"]]["ans_data"] = ad

    questions_list = []
    answers_list = []

    for seq_id in tqdm(range(max_seqs), total=max_seqs, desc=f"Processing {split}"):
        path = split_seq_holder.format(split, seq_id)
        file_data = np.load(path, allow_pickle=True).item()

        for qid, value in file_data.items():
            q_data = data_dict[qid]["que_data"]
            # q_data["languages"], q_data["sim_score"] = ["en"], 1.0
            a_data = data_dict[qid]["ans_data"]

            questions = value["rephrasings_list"]
            questions = sorted(questions, key=lambda x: x["sim_score"], reverse=True)
            assert questions[0]["rephrasing"] == q_data["question"]

            rephrasing_ids = []
            for idx, que in enumerate(questions):
                que["question"] = que["rephrasing"]
                del que["rephrasing"]
                que["image_id"] = q_data["image_id"]
                if idx > 0:
                    que["question_id"] = q_data["question_id"] * org_factor + (idx-1)
                else:
                    que["question_id"] = q_data["question_id"]
                rephrasing_ids.append(que["question_id"])

            answers = []
            for que in questions:
                # These change after filtering so not using them in the first place!
                # que["rephrasing_ids"] = [x for x in rephrasing_ids if x != que["question_id"]]
                answer = deepcopy(a_data)
                answer["question_id"] = que["question_id"]
                answers.append(answer)

            assert len(answers) == len(questions)
            answers_list.append(answers)
            questions_list.append(questions)

    questions_list = sorted(questions_list, key=lambda item: item[0]["question_id"])
    answers_list = sorted(answers_list, key=lambda item: item[0]["question_id"])
    assert len(questions_list) == len(answers_list)
    que_data["questions"] = questions_list

    cPickle.dump(answers_list, open(ans_split_path_dict[split][0], "wb"))
    cPickle.dump(que_data, open(que_split_path_dict[split][0], "wb"))
    print(f"Dumped files: \n {que_split_path_dict[split][0]} \n {ans_split_path_dict[split][0]}")
