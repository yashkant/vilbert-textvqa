from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from transformers.hf_api import HfApi
import json

split_path_dict = {
    "re_train": ["data/re-vqa/data/revqa_train_proc.json", "datasets/VQA/cache/revqa_train_target.pkl", "train"],
    "re_val": ["data/re-vqa/data/revqa_val_proc.json", "datasets/VQA/cache/revqa_val_target.pkl", "val"],
    "re_train_negs": ["data/re-vqa/data/revqa_train_proc_image_negs.json",
                      "datasets/VQA/cache/revqa_train_target_image_negs.pkl", "train"],
    "re_val_negs": ["data/re-vqa/data/revqa_val_proc_image_negs.json",
                    "datasets/VQA/cache/revqa_val_target_image_negs.pkl", "val"],
}

questions = json.load(open(f"../../{split_path_dict['re_train'][0]}"))["questions"]
questions = [q["question"] for q in questions[:100]]

common_languages = [
    "de",  # german
    "fr",  # french
    "zh",  # chinese
    "es",  # spanish
    "ar",  # arabic
    "ru",  # russian
    "nl",  # dutch
    "it",  # italian
]


# check avialable models
model_list = HfApi().model_list()
org = "Helsinki-NLP"
model_ids = [x.modelId for x in model_list if x.modelId.startswith(org)]
# filter english models
suffix = [x.split('/opus-mt-')[1] for x in model_ids if "en" in x]
# filter for BT (single hop only)
all_bt_langs = [x for x in suffix if f'{x.split("-")[1]}-{x.split("-")[0]}' in suffix]
# bt_langs = [f'{s}-en' for s in common_languages]
# bt_langs = [m for m in bt_langs if m in all_bt_langs]

data = {
    "en": questions
}
use_single = True

for lang_pair in all_bt_langs:
    lang = lang_pair.split("-")[0]
    if lang == "en" or lang in data:
        continue
    print(f"Processing: {lang}")
    model_name_for = f'Helsinki-NLP/opus-mt-en-{lang}'
    model_name_back = f'Helsinki-NLP/opus-mt-{lang}-en'
    tokenizer_for = MarianTokenizer.from_pretrained(model_name_for)
    tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)

    model_for = MarianMTModel.from_pretrained(model_name_for)
    model_back = MarianMTModel.from_pretrained(model_name_back)

    translated_text = model_for.generate(**tokenizer_for.prepare_translation_batch(questions))
    translated_text_single = []
    for que in tqdm(questions):
        translated_que = model_for.generate(**tokenizer_for.prepare_translation_batch([que]))
        translated_text_single.extend(translated_que)

    translated_text = [tokenizer_for.decode(t, skip_special_tokens=True) for t in translated_text]
    translated_text_single = [tokenizer_for.decode(t, skip_special_tokens=True) for t in translated_text_single]

    back_translated_text = model_back.generate(**tokenizer_back.prepare_translation_batch(translated_text))
    back_translated_text_single = []
    for que in tqdm(translated_text_single):
        back_translated_que = model_back.generate(**tokenizer_back.prepare_translation_batch([que]))
        back_translated_text_single.extend(back_translated_que)

    back_translated_text = [tokenizer_back.decode(t, skip_special_tokens=True) for t in back_translated_text]
    back_translated_text_single = [tokenizer_back.decode(t, skip_special_tokens=True) for t in back_translated_text_single]

    assert (back_translated_text) == (back_translated_text_single)
    # print(f"-----------------------{lang}-----------------------")
    # for org, bt in zip(questions, back_translated_text):
    #     print(f"{org}                      {bt}")

    # data[lang] = back_translated_text

import pdb
pdb.set_trace()
import pandas as pd
df = pd.DataFrame(data)
df.to_csv("play_bt_all.csv", encoding='utf-8', index=False)

# ["c'est une phrase en anglais que nous voulons traduire en français",
# 'Isto deve ir para o português.',
# 'Y esto al español']
