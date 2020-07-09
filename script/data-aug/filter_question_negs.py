import itertools
from collections import defaultdict
from tqdm import tqdm
import _pickle as cPickle
import os

def filter_negatives(sample):

    import pdb
    pdb.set_trace()

    # filter same-image questions
    same_image_ids = image_dict[sample["image_id"]]
    fil_same_image_ids = []
    ref_answers = answer_dict[sample["question_id"]]
    for qid in same_image_ids:
        if qid == sample["question_id"]:
            continue
        cand_answers = answer_dict[qid]
        if len(set(ref_answers).intersection(set(cand_answers))) == 0:
            fil_same_image_ids.append(qid)
    sample["same_image_questions_neg"] = fil_same_image_ids

    # filter top-k questions
    if sample["question_id"] not in negs_dict:
        return True

    top_k_sim_scores, top_k_questions = negs_dict[sample["question_id"]]
    fil_top_k_questions = []
    for qid in top_k_questions:
        cand_answers = answer_dict[qid]
        if len(set(ref_answers).intersection(set(cand_answers))) == 0:
            fil_top_k_questions.append(qid)
    sample["top_k_questions_neg"] = fil_top_k_questions

    return False
    # print(f"Ref: {sample['question']}, Ans: {[label2ans[x] for x in ref_answers]}")
    # try:
    #     for qid in fil_top_k_questions[:10]:
    #         print(f"Neg Cand: {question_dict[qid]}, Ans: {[label2ans[x] for x in answer_dict[qid]]}")
    # except:
    #     import pdb
    #     pdb.set_trace()

def multi_process(questions):
    for que in questions:
        filter_negatives(que)

que_split_path_dict = {

    "minval":  "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_minval2014_questions.pkl",
    # "val":  "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_val2014_questions.pkl",
    # "train": "../../datasets/VQA/back-translate/org2_bt_v2_OpenEnded_mscoco_train2014_questions.pkl",
}

answer_paths = [
    "../../datasets/VQA/back-translate/org_bt_minval_target.pkl",
    # "../../datasets/VQA/back-translate/org2_bt_val_target.pkl",
    # "../../datasets/VQA/back-translate/org2_bt_train_target.pkl",
    ]


ans2label_path = os.path.join("../../datasets/VQA/", "cache", "trainval_ans2label.pkl")
label2ans_path = os.path.join("../../datasets/VQA/", "cache", "trainval_label2ans.pkl")

# ans2label_path = os.path.join("../../datasets/VQA/", "cache", "minval_ans2label.pkl")
# label2ans_path = os.path.join("../../datasets/VQA/", "cache", "minval_label2ans.pkl")

ans2label = cPickle.load(open(ans2label_path, "rb"))
label2ans = cPickle.load(open(label2ans_path, "rb"))
answer_dict = {}
# use later
answers_data = []

for path in (answer_paths):
    answers = cPickle.load(open(path, "rb"))
    answers_data.append(answers)
    answers = list(itertools.chain.from_iterable(answers))
    for ans in answers:
        answer_dict[ans["question_id"]] = ans["labels"]

print("Read Answers")

negs_path = "../../datasets/VQA/back-translate/train_val_question_negs.pkl"
negs_data = cPickle.load(open(negs_path, "rb"))
negs_dict = {}
for qid, sim_scores, sim_qids in zip(negs_data["qids"], negs_data["sim_scores"], negs_data["sim_qids"]):
    negs_dict[qid] = (sim_scores, sim_qids)

print("Read Matrix")

# create dicts
image_dict = defaultdict(list)
questions_rephrasings = defaultdict(list)
question_dict = {}

# use later
questions_data = []

for que_path in que_split_path_dict.values():
    data = cPickle.load(open(que_path, "rb"))
    questions_data.append(data)
    questions_list = data["questions"]

    # add "rephrasing_of" key
    for _questions in questions_list:
        # only keep the min-qid in same-image ids
        min_qid = min([x['question_id'] for x in _questions])
        assert len(set([x['image_id'] for x in _questions])) == 1
        image_dict[_questions[0]["image_id"]].append(min_qid)
        for _que in _questions:
            question_dict[_que["question_id"]] = _que["question"]

print("Read Questions")

for que_data, ans_data, que_path, ans_path in zip(questions_data, answers_data, que_split_path_dict.values(), answer_paths):
    data = que_data
    answers = ans_data

    questions_list = data["questions"]

    # add "rephrasing_of" key
    for _questions in questions_list:
        rep_id = min([s['question_id'] for s in _questions])
        for _que in _questions:
            _que["rephrasing_of"] = rep_id

    assert len(questions_list) == len(answers)

    import pdb
    pdb.set_trace()

    # remove questions w/o negatives
    for idx in tqdm(range(len(questions_list)), total=len(questions_list)):
        _updated_ques = []
        _updated_answers = []
        _questions = questions_list[idx]
        _answers = answers[idx]

        for _que, _ans in zip(_questions, _answers):
            delete = filter_negatives(_que)
            if not delete:
                _updated_ques.append(_que)
                _updated_answers.append(_ans)
            else:
                import pdb
                pdb.set_trace()
        questions_list[idx] = _updated_ques
        answers[idx] = _updated_answers

    import pdb
    pdb.set_trace()

    cPickle.dump(data, open(que_path, "wb"), protocol=2)
    cPickle.dump(data, open(que_path, "wb"), protocol=2)
