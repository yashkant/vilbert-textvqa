import os
import json
import _pickle as cPickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict


registry = edict({
    "aug_filter": {
        "max_re_per_sample": 4,
        "sim_threshold": 0.95,
        "sampling": "top"
    },
    "use_rephrasings": True
})

def filter_aug(questions_list, answers_list):
    questions, answers = [], []
    max_samples = registry.aug_filter["max_re_per_sample"]
    sim_threshold = registry.aug_filter["sim_threshold"]
    sampling = registry.aug_filter["sampling"]

    rephrasings_data = []
    assert len(questions_list) == len(answers_list)

    if not registry.use_rephrasings:
        if max_samples != 1:
            max_samples = 1
            registry.aug_filter["max_re_per_sample"] = 1
            print(f"Use rephrasings is False, setting max-samples to : {max_samples}")
        else:
            print(f"Use rephrasings is False w/ max-samples : {max_samples}")


    for idx, (que_list, ans_list) in tqdm(enumerate(zip(questions_list, answers_list)), total=len(questions_list),
                                          desc="Filtering Data"):
        assert len(que_list) == len(ans_list)
        # filter for sim-threshold
        if sim_threshold > 0:
            que_list, ans_list = zip(*[(q,a) for q,a in zip(que_list, ans_list) if q["sim_score"] > sim_threshold])
        # filter for max-samples
        if max_samples > 0:
            if sampling == "top":
                que_list, ans_list = que_list[:max_samples], ans_list[:max_samples]
            elif sampling == "bottom":
                que_list, ans_list = que_list[-max_samples:], ans_list[-max_samples:]
            elif sampling == "random":
                # use only original question
                if len(que_list) == 1:
                    que_list, ans_list = que_list[0:1], ans_list[0:1]
                else:
                    rand_indices = np.random.choice(range(1, len(que_list)), min(max_samples - 1, len(que_list) - 1), replace=False)
                    # add original question
                    rand_indices = [0] + sorted(rand_indices)
                    que_list, ans_list = np.array(que_list), np.array(ans_list)
                    que_list, ans_list = que_list[rand_indices], ans_list[rand_indices]

            else:
                raise ValueError

        filtered_rephrasing_ids = [que["question_id"] for que in que_list]
        for que in que_list:
            que["rephrasing_ids"] = sorted([x for x in filtered_rephrasing_ids if x != que["question_id"]])

        # add them to main list
        questions.extend(que_list)
        answers.extend(ans_list)
        rephrasings_data.append(len(que_list))

    return questions, answers

def rephrasings_dict(split, questions):
    question_rephrase_dict = {}

    for question in questions:
        if "rephrasing_of" in question:
            question_rephrase_dict[question["question_id"]] = question["rephrasing_of"]
        elif "rephrasing_ids" in question:
            min_qid = min(question["rephrasing_ids"] + [question["question_id"]])
            question_rephrase_dict[question["question_id"]] = min_qid
        else:
            question_rephrase_dict[question["question_id"]] = question["question_id"]


    # used in evaluation, hack to set attribute
    from easydict import EasyDict
    super(EasyDict, registry).__setattr__(f"question_rephrase_dict_{split}", question_rephrase_dict)
    super(EasyDict, registry).__setitem__(f"question_rephrase_dict_{split}", question_rephrase_dict)
    print(f"Built dictionary: question_rephrase_dict_{split}")

image_path = "/srv/share/datasets/coco"

imdb_paths = {
    "re_val": ["data/re-vqa/data/revqa_val_proc.json", "datasets/VQA/cache/revqa_val_target.pkl", "val",
               "val2014/COCO_val2014_{}.jpg"],
    "re_train": ["data/re-vqa/data/revqa_train_proc.json", "datasets/VQA/cache/revqa_train_target.pkl", "train",
                 "val2014/COCO_val2014_{}.jpg"],

}

baseline_model_preds = json.load(open("../../baseline_model.json"))

ans2label_path = os.path.join("../../datasets/VQA/", "cache", "trainval_ans2label.pkl")
label2ans_path = os.path.join("../../datasets/VQA/", "cache", "trainval_label2ans.pkl")
ans2label = cPickle.load(open(ans2label_path, "rb"))
label2ans = cPickle.load(open(label2ans_path, "rb"))

isascii = lambda s: len(s) == len(s.encode())

def replace_str(str_value):
    if not isascii(str_value):
        return "Not ascii-string"
    else:
        return str_value


split_path_dict = {
    "train_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_train2014_questions.pkl",
                 "datasets/VQA/back-translate/org2_bt_train_target.pkl", "train"],
    "val_aug": ["../../datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
                 "../../datasets/VQA/back-translate/org2_bt_val_target.pkl", "val"],
    "trainval_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_trainval2014_questions.pkl",
                 "datasets/VQA/back-translate/org2_bt_trainval_target.pkl", "trainval"],
    "minval_aug": ["datasets/VQA/back-translate/org3_bt_v2_OpenEnded_mscoco_minval2014_questions.pkl",
                 "datasets/VQA/back-translate/org2_bt_minval_target.pkl", "minval"],
}
questions_path, answers_path, split = split_path_dict["val_aug"][0], split_path_dict["val_aug"][1], split_path_dict["val_aug"][-1]

exps_dir = {
    "baseline": "../../save/VQA_spatial_m4c_mmt_vqa-sup-baseline-train_aug4t_th0.95-minval-fixed-cs-eval",
    "best-aug-12": "../../save/VQA_spatial_m4c_mmt_vqa-scl-joint-base-strict-pos-1-th-4-4-train_aug4t_th0.95-minval-sampler-new-hard-mask_sc20-shuffle-alt-no-re-im-cf-4-fixed-ref-10-plot"
}

questions_list = cPickle.load(open(questions_path, "rb"))
answers_list = cPickle.load(open(answers_path, "rb"))

# filter-mech
questions, answers = filter_aug(questions_list, answers_list)
assert len(questions) == len(answers)
rephrasings_dict(split, questions)

bt_question_id_dict = {}
for q in questions:
    bt_question_id_dict[q["question_id"]] = q["question"]

print(f"Train Samples after filtering: {len(questions)}")
# this is needed for evaluation

for question, answer in zip(questions, answers):
    assert answer["question_id"] == question["question_id"]

def fetch_preds(ref_id):
    return_list = []
    for key in ['best-aug-12-bt', 'best-aug-12-human', 'baseline-bt', 'baseline-human']:
        _preds = preds_dict[key][ref_id]
        _preds["cs_scores"] = [_preds[str(k)] for k in [1, 2, 3, 4] if str(k) in _preds]

        if "human" in key:
            _preds["questions"] = [human_question_id_dict[qid] for qid in _preds["question_ids"]]
        else:
            _preds["questions"] = [bt_question_id_dict[qid] for qid in _preds["question_ids"]]
        return_list.append(_preds)

    return return_list


debug_dict = defaultdict(list)

def generate_html(data, output_path, type, wrt="human"):
    """
    Generates html placing images side-by-side from the folders_list.

    :param output_path: output html file-path
    :param titles: list<string> to label each column in html
    :param imdb: loaded imdb, pass only the instances which have drawn-images in both the folders
    :param folders_list: list<folder-paths> where we have drawn images stores (MAKE SURE THESE ARE ABSOLUTE PATHS)
    :param prepend_extension_list: list<(prepend-string, extension)> that is used to build image-path
    :return:
    """

    html = '<html lang="en">' + '\n <p></p> \n <p></p>'
    titles = ["Image", "Data"]
    col_string = [f"<th>{title}</th>" for title in titles]
    col_string = "".join(col_string)
    html += f'<table width="1500"><tr height="40">{col_string}</tr>'

    displayed_ids = []  # multiple questions based on same image
    instances_count = 0

    for instance in data:
        image_id = instance["image_id"]
        if image_id not in displayed_ids:
            displayed_ids.append(image_id)
        else:
            continue

        # add image
        image_path = instance["image_path"]
        instance_html = f'<tr><td><img src="{image_path}"></td>\n'
        # add image-data
        instance_html += f"<td>"
        # Add image-path
        instance_html += f"Image Path: {instance['image_path']} <br> \n"

        # Add ref. question and answer
        instance_html += f"Ref. Question: {instance['question']} <br> \n"
        instance_html += f"Ref. Answers: {instance['valid_answers']} <br> \n"
        instance_html += f"----- <br> \n"

        # Add model-preds for human and BT rephrasings
        ref_id = str(min([instance["question_id"]] + instance["rephrasing_ids"]))
        best_preds_bt, best_preds_h, base_preds_bt, base_preds_h = fetch_preds(ref_id)


        best_preds, base_preds = (best_preds_h, base_preds_h) if wrt == "human" else (best_preds_bt, base_preds_bt)

        # _best_tag = None
        # _baseline_tag = None
        #
        # for typ in ["best", "baseline"]:
        #     if typ == "best":
        #         _preds = best_preds
        #     else:
        #         _preds = base_preds
        #     _tag = None
        #     if sum(_preds["cs_scores"]) == len(_preds["cs_scores"]):
        #         _tag = "consistent"
        #     elif sum(_preds["cs_scores"]) == 0:
        #         _tag = "wrong"
        #     else:
        #         _tag = "inconsistent"
        #
        #     if typ == "best":
        #         _best_tag = _tag
        #     else:
        #         _baseline_tag = _tag
        #
        # tag_key = f"base-{_baseline_tag}-best-{_best_tag}"
        # tags_dict[tag_key].append(ref_id)


        if type == "both_wrong":
            if sum(best_preds["cs_scores"]) != 0 or sum(base_preds["cs_scores"]) != 0:
                continue
        elif type == "both_consistent":
            if sum(best_preds["cs_scores"]) != len(best_preds["cs_scores"]) or \
                    sum(base_preds["cs_scores"]) != len(base_preds["cs_scores"]):
                continue
        elif type == "baseline_consistent":
            if sum(base_preds["cs_scores"]) != len(base_preds["cs_scores"]) or \
                    sum(best_preds["cs_scores"]) == len(best_preds["cs_scores"]):

                # if baseline is inconsistent or best is consistent continue
                continue
        elif type == "best_consistent":
            if sum(best_preds["cs_scores"]) != len(best_preds["cs_scores"]) or \
                    sum(base_preds["cs_scores"]) == len(base_preds["cs_scores"]):

                # if best is inconsistent or baseline is consistent continue
                continue
        elif type == "any_inconsistent":
            if sum(best_preds["cs_scores"]) == len(best_preds["cs_scores"]) or \
                    sum(base_preds["cs_scores"]) == len(base_preds["cs_scores"]) or \
                    (sum(best_preds["cs_scores"]) == 0 and sum(base_preds["cs_scores"]) == 0):

                # if best is consistent or baseline is consistent continue
                continue
        else:
            raise ValueError

        debug_dict[type].append(ref_id)
        instances_count += 1

        for idx in range(len(best_preds["questions"])):
            assert best_preds["questions"] == base_preds["questions"]
            assert best_preds["question_ids"] == base_preds["question_ids"]

            qid = best_preds["question_ids"][idx]
            q = best_preds["questions"][idx]

            if qid == ref_id:
                instance_html += f"Ref Question: {q} <br> \n"
            else:
                instance_html += f"{wrt} Question {idx + 1}: {q} <br> \n"

            instance_html += f"Baseline Answer: {base_preds['answers'][idx]} | VQA Score: {base_preds['vqa_scores'][idx]} <br> \n"
            instance_html += f"Best Answer: {best_preds['answers'][idx]} | VQA Score: {best_preds['vqa_scores'][idx]} <br> \n"
            instance_html += "\n -- <br> \n"

        instance_html += f"Base Model Cs Scores: {base_preds['cs_scores']} <br> \n"
        instance_html += f"Best Model Cs Scores: {best_preds['cs_scores']} <br> \n"
        instance_html += f"------------------------------------------- <br> \n"
        instance_html += ' </td> </tr>\n'
        html += instance_html

    html += '</table></body></html>'

    output_path = os.path.join(output_path, f"{type}-{wrt}.html")
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Dumped: {output_path} w/ {instances_count} instances and {round(instances_count/len(data),4)*100}% of data")

def add_image_path(instance, images_folder, image_holder):
    id = "0" * (int_len - len(str(instance["image_id"]))) + str(instance["image_id"])
    instance["image_path"] = os.path.join(images_folder, image_holder.format(id))
    try:
        assert os.path.exists(instance["image_path"])
    except:
        import pdb
        pdb.set_trace()

# just fixes vqa-scores
def build_dict(preds):
    for key, value in preds["revqa_bins_scores"].items():
        value["vqa_scores"] = list(zip(* value["vqa_scores"]))[1]
    return preds["revqa_bins_scores"]

root_holder = "../../{}"
int_len = 12


# store all preds available
preds_dict = {}
for model_type, exp_dir in exps_dir.items():
    human_preds = json.load(open(os.path.join(exp_dir, "preds_revqa.json")))
    human_preds_dict = build_dict(human_preds)

    bt_preds = json.load(open(os.path.join(exp_dir, "preds_revqa_bt.json")))
    bt_preds_dict = build_dict(bt_preds)

    preds_dict[f"{model_type}-human"] = human_preds_dict
    preds_dict[f"{model_type}-bt"] = bt_preds_dict

# Build qid vs question dicts
# stores qid -> human instance
human_question_id_dict = {}

total_questions = []
for key in imdb_paths:
    json_file, ans_file, split, image_holder = imdb_paths[key]
    json_file, ans_file = root_holder.format(json_file), root_holder.format(ans_file)
    assert os.path.exists(json_file)
    assert os.path.exists(ans_file)

    que_data = json.load(open(json_file))
    ans_data = cPickle.load(open(ans_file, "rb"))

    assert len(ans_data) == len(que_data["questions"])

    count = 0
    for ans_instance, question_instance in tqdm(zip(ans_data, que_data["questions"]), total=len(ans_data)):

        # fill human-qid-dict
        human_question_id_dict[question_instance["question_id"]] = question_instance["question"]


        # skip non-reference questions
        ref_id = min([question_instance["question_id"]] + question_instance["rephrasing_ids"])
        if ref_id != question_instance["question_id"]:
            continue

        question_instance.update(ans_instance)
        add_image_path(question_instance, image_path, image_holder)
        count += 1
        # if count == 100:
        #     break
        total_questions.append(question_instance)

for wrt in ["human", "bt"]:
    for type in ["both_wrong", "both_consistent", "best_consistent", "baseline_consistent", "any_inconsistent"]:
        generate_html(total_questions, "fail-modes/", type=type, wrt=wrt)

    # for j in debug_dict.keys():
    #     for k in debug_dict.keys():
    #         print(j, k, len(set(debug_dict[j]).intersection(set(debug_dict[k]))))

