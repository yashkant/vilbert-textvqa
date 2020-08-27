from collections import defaultdict
from copy import deepcopy

import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import _pickle as cPickle
import nltk
from nltk.corpus import stopwords
import contractions
ignore_articles = stopwords.words('english')
import string


split_seq_holder = "../../datasets/VQA/back-translate/sim-result2/rep_{}_{}.npy"
max_seqs = 10
org_factor = int(1e10)

# decontract, depunctuate
filter_type = "dcp"

# sampling: ["top", "random"]
sampling_type = "random"
# sampling_type = "top"

# filter_thresh
sim_thresh = 0.8


def filter_rephrasings(rephrasings_list, q_data):
    # reduced-question -> most-similar rephrasing
    filter_bins = {}

    questions = sorted(rephrasings_list, key=lambda x: x["sim_score"], reverse=True)
    questions = [item for item in questions if item["sim_score"] >= 0.8]

    for idx, item in enumerate(questions):
        question = item["rephrasing"].lower()
        question = contractions.fix(question)
        question = "".join([char for char in question if char not in string.punctuation])
        question = " ".join([word for word in question.split(" ") if word not in ignore_articles])

        if question in filter_bins:
            if item["sim_score"] > filter_bins[question]["sim_score"]:
                filter_bins[question] = item
        else:
            filter_bins[question] = item

    questions = sorted(list(filter_bins.values()), key=lambda x: x["sim_score"], reverse=True)
    del_keys = ["rephrasing", "languages"]

    # assert that first element is original question
    try:
        assert questions[0]["rephrasing"] == q_data["question"]
    except:
        import pdb
        pdb.set_trace()
    for idx, que in enumerate(questions):
        que["question"] = que["rephrasing"]
        for key in del_keys:
            del que[key]
        que["image_id"] = q_data["image_id"]
        if idx > 0:
            que["question_id"] = q_data["question_id"] * org_factor + (idx - 1)
        else:
            que["question_id"] = q_data["question_id"]
    return questions

# Dumping Paths
que_split_path_dict = {
    "train": ("../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_train2014_questions.pkl".format(filter_type, sampling_type),
              "../../datasets/VQA/v2_OpenEnded_mscoco_train2014_questions.json"),

    "val": ("../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_val2014_questions.pkl".format(filter_type, sampling_type),
            "../../datasets/VQA/v2_OpenEnded_mscoco_val2014_questions.json"),

    "test": ("../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_test2015_questions.pkl".format(filter_type, sampling_type),
             "../../datasets/VQA/v2_OpenEnded_mscoco_test2015_questions.json"),

    "trainval": (
    "../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_trainval2014_questions.pkl".format(filter_type, sampling_type),
    "../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_v2_OpenEnded_mscoco_minval2014_questions.pkl".format(filter_type, sampling_type)),
}


ans_split_path_dict = {
    "train": ("../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_train_target.pkl".format(filter_type, sampling_type),
              "../../datasets/VQA/cache/train_target.pkl"),

    "val": ("../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_val_target.pkl".format(filter_type, sampling_type),
            "../../datasets/VQA/cache/val_target.pkl"),

    "test": ("../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_test_target.pkl".format(filter_type, sampling_type),
             "../../datasets/VQA/cache/test_target.pkl"),

    "trainval": ("../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_trainval_target.pkl".format(filter_type, sampling_type),
                 "../../datasets/VQA/back-translate/bt_fil_{}_sampling_{}_minval_target.pkl".format(filter_type, sampling_type)),
}

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

    # import pdb
    # pdb.set_trace()

    for qd, ad in zip(que_data["questions"], ans_data):
        data_dict[qd["question_id"]]["que_data"] = qd
        data_dict[ad["question_id"]]["ans_data"] = ad

    questions_list = []
    answers_list = []

    for seq_id in tqdm(range(max_seqs), total=max_seqs, desc=f"Processing {split}"):
        path = split_seq_holder.format(split, seq_id)
        file_data = np.load(path, allow_pickle=True).item()

        for qid, value in tqdm(file_data.items(), "Sequence Progress"):
            q_data = data_dict[qid]["que_data"]
            # q_data["languages"], q_data["sim_score"] = ["en"], 1.0
            a_data = data_dict[qid]["ans_data"]

            questions = filter_rephrasings(value["rephrasings_list"], q_data)

            if sampling_type == "top":
                questions = questions[:4]
            else:
                sampled_questions = list(np.random.choice(questions[1:], min(len(questions[1:]), 3), replace=False))
                questions = [questions[0]] + sampled_questions

            assert len([item["question_id"] for item in questions]) == \
                   len(set([item["question_id"] for item in questions]))

            answers = []
            for que in questions:
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
