import json
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sacrebleu
warnings.filterwarnings("ignore")

def cos_sim(A, B):
    # A and B -> [N x dim]
    Bt = np.transpose(B)
    AB_mul = np.matmul(A, Bt)
    A_mag = np.linalg.norm(A, axis=1, keepdims=True)
    B_mag = np.linalg.norm(B, axis=1, keepdims=True)
    AB_mag = np.matmul(A_mag, np.transpose(B_mag))
    assert AB_mag.shape == AB_mul.shape
    return AB_mul/AB_mag

df = pd.read_csv('play_bt_all.csv')
data = []
col_names = list(df.keys())
embeddings_dict = {}

sim_model = model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
for index, row in tqdm(df.iterrows(), total=len(df)):
    batch = list(row)
    ref_question = batch[0]
    batch_embeddings = model.encode(batch)
    sim_scores = cos_sim([batch_embeddings[0]], batch_embeddings).round(2)[0]
    # bleu_scores = [round(sacrebleu.corpus_bleu([que], [[ref_question]]).score, 2) for que in batch]

    # sort batch based on sim_scores
    question_sim_pairs = [(x, y) for x, y in zip(batch, sim_scores)]
    # question_sim_pairs = set(question_sim_pairs)
    # question_sim_pairs = sorted(question_sim_pairs, key=lambda pair: pair[1], reverse=True)
    question_sim_pairs = [x[1] for x in question_sim_pairs]

    # filtered_batch = []
    # for idx, (sim_score, bleu_score) in enumerate(zip(sim_scores, bleu_scores)):
    #     if sim_score > 0.9:
    #         filtered_batch.append(batch[idx] + f"| bleu: {bleu_score}, sim: {sim_score.round(2)}")
    #         batch[idx] = batch[idx] + f"| bleu: {bleu_score}, sim: {sim_score.round(2)}"
    #     else:
    #         break
    data.append(question_sim_pairs)

import pdb
pdb.set_trace()
df = pd.DataFrame(data)
df.columns = col_names
df.loc[len(df)] = df.mean()
# set the threshold here!
df.to_csv("avg_sim.csv")
# df.to_csv("play_bt_all_sim_bleu.csv", encoding='utf-8', index=False)

# for lang_pair in all_bt_langs:
#     lang = lang_pair.split("-")[0]
#     if lang == "en" or lang in data:
#         continue
#     print(f"Processing: {lang}")
#     model_name_for = f'Helsinki-NLP/opus-mt-en-{lang}'
#     model_name_back = f'Helsinki-NLP/opus-mt-{lang}-en'
#
#     try:
#         tokenizer_for = MarianTokenizer.from_pretrained(model_name_for)
#         tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
#
#         model_for = MarianMTModel.from_pretrained(model_name_for)
#         model_back = MarianMTModel.from_pretrained(model_name_back)
#     except:
#         print(f"Error with {lang}")
#         continue
#
#     translated_text = model_for.generate(**tokenizer_for.prepare_translation_batch(questions))
#
#
#     translated_text = [tokenizer_for.decode(t, skip_special_tokens=True) for t in translated_text]
#
#     back_translated_text = model_back.generate(**tokenizer_back.prepare_translation_batch(translated_text))
#     back_translated_text = [tokenizer_back.decode(t, skip_special_tokens=True) for t in back_translated_text]
#
#     assert len(back_translated_text) == len(translated_text)
#     data[lang] = back_translated_text
#     df = pd.DataFrame(data)
#     df.to_csv("play_bt_all.csv", encoding='utf-8', index=False)
#
# # ["c'est une phrase en anglais que nous voulons traduire en français",
# # 'Isto deve ir para o português.',
# # 'Y esto al español']
