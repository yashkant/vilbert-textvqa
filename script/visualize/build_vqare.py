import os
import json
import _pickle as cPickle

from tqdm import tqdm

image_path = "/srv/share/datasets/coco/val2014"

imdb_paths = {
    "re_train": ["data/re-vqa/data/revqa_train_proc.json", "datasets/VQA/cache/revqa_train_target.pkl", "train",
                 "COCO_val2014_{}.jpg"],
    "re_val": ["data/re-vqa/data/revqa_val_proc.json", "datasets/VQA/cache/revqa_val_target.pkl", "val",
               "COCO_val2014_{}.jpg"],
}

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

def generate_html(data,
                  output_path):
    """
    Generates html placing images side-by-side from the folders_list.

    :param output_path: output html file-path
    :param titles: list<string> to label each column in html
    :param imdb: loaded imdb, pass only the instances which have drawn-images in both the folders
    :param folders_list: list<folder-paths> where we have drawn images stores (MAKE SURE THESE ARE ABSOLUTE PATHS)
    :param prepend_extension_list: list<(prepend-string, extension)> that is used to build image-path
    :return:
    """
    # import pdb
    # pdb.set_trace()
    question_id_dict = {}

    for instance in data:
        question_id_dict[instance["question_id"]] = instance

    html = '<html lang="en">' + '\n <p></p> \n <p></p>'
    titles = ["Image", "Data"]
    col_string = [f"<th>{title}</th>" for title in titles]
    col_string = "".join(col_string)
    html += f'<table width="1500"><tr height="40">{col_string}</tr>'

    displayed_ids = []  # multiple questions based on same image

    for instance in data:
        image_id = instance["image_id"]

        if image_id not in displayed_ids:
            displayed_ids.append(image_id)
        else:
            continue

        # add image
        image_path = instance["image_path"]
        html += f'<tr><td><img src="{image_path}"></td>\n'

        # import pdb
        # pdb.set_trace()

        keys = [
            ("Image Path", "image_path"),
            ("Answers", "valid_answers"),
            ("Primary Question", "question"),
        ]

        # add image-data
        html += f"<td>"

        for str_value, key in keys:
            key_value = instance[key]

            if isinstance(key_value, list):
                key_value = [replace_str(k) for k in key_value]
            elif isinstance(key_value, str):
                key_value = replace_str(key_value)

            html += f"{str_value}: {key_value} <br> \n"

        rephrased_questions = []
        for re_id in instance["rephrasing_ids"]:
            key_value = question_id_dict[re_id]["question"]
            if not isascii(key_value):
                key_value = "non-ascii string"
            rephrased_questions.append(key_value)
        html += f"Rephrased Questions: {rephrased_questions}"

        html += ' </td> </tr>\n'

    html += '</table></body></html>'

    with open(output_path, "w") as f:
        f.write(html)


def add_image_path(instance, images_folder, image_holder):
    id = "0" * (int_len - len(str(instance["image_id"]))) + str(instance["image_id"])
    instance["image_path"] = os.path.join(images_folder, image_holder.format(id))
    assert os.path.exists(instance["image_path"])


root_holder = "../../{}"
int_len = 12

total_jdata = []
for key in imdb_paths:
    json_file, ans_file, split, image_holder = imdb_paths[key]
    json_file, ans_file = root_holder.format(json_file), root_holder.format(ans_file)
    assert os.path.exists(json_file)
    assert os.path.exists(ans_file)

    json_data = json.load(open(json_file))
    pickle_data = cPickle.load(open(ans_file, "rb"))

    assert len(pickle_data) == len(json_data["questions"])

    for ans_instance, question_instance in tqdm(zip(pickle_data, json_data["questions"]), total=len(pickle_data)):
        question_instance.update(ans_instance)
        add_image_path(question_instance, image_path, image_holder)

    total_jdata.extend(json_data["questions"])

generate_html(total_jdata, "vqa_re.html")
print(f"Number of Samples Dumped: {len(total_jdata)}")