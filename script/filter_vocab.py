# vocab5k = "/srv/share/ykant3/m4c-release/data/m4c_vocabs/textvqa/fixed_answer_vocab_textvqa_5k.txt"
# vocab4k = "/srv/share/ykant3/pythia/vocabs/answers_textvqa_more_than_1_m4c_no_dups.txt"
#
# vocab_4k_list = []
# with open(vocab4k) as file:
#     for line in file.readlines():
#         vocab_4k_list.append(line.strip())
#
# vocab_5k_list = []
# with open(vocab5k) as file:
#     for line in file.readlines():
#         vocab_5k_list.append(line.strip())
#
#
# set(vocab_4k_list).intersection(set(vocab_5k_list))
# diff = set(vocab_5k_list) - set(vocab_4k_list)
# diff = set(vocab_4k_list) - set(vocab_5k_list)


import json
import numpy as np
from collections import Counter
import string
from tqdm import tqdm

prev_imdb_file = '../datasets/textvqa/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_train.npy'
max_vocab_num = 3996
save_vocab_file = '../datasets/textvqa/vocabs/answer_vocab_textvqa_4k_filtered.txt'
imdb = np.load(prev_imdb_file, allow_pickle=True)

def word_tokenize(word):
    word = word.lower()
    word = word.replace(",", "").replace("?", "").replace("'s", " 's")
    return word.strip()

isascii = lambda s: len(s) == len(s.encode())

def filter_vocab(counter, most):
    word_freq_list = counter.most_common(most)
    filtered_list = []

    for word, freq in word_freq_list:
        if freq <= 35:
            special_chars = set(string.punctuation)
            numbers = sum(c.isdigit() for c in word)
            if numbers > 2:
                print(word)
                continue
            if any(char in special_chars for char in word) and len(word) >= 2:
                print(word)
                continue
        filtered_list.append((word, freq))
    return filtered_list


vocab_counter = Counter()
multi_word_counter = Counter()

for instance in tqdm(imdb):
    if 'answers' not in instance:
        continue
    for answer in instance['answers']:
        if not isascii(answer):
            continue
        answer = word_tokenize(answer)
        answer_words = answer.split()
        if len(answer_words) > 1:
            multi_word_counter[answer] += 1
        vocab_counter.update(answer_words)


filter_multi = filter_vocab(multi_word_counter, 500)
filter_single = filter_vocab(vocab_counter, 5000)

vocab_save = [w for w, _ in filter_multi]
remaining_len = max_vocab_num - len(vocab_save)
vocab_save = vocab_save + [w for w, _ in filter_single[:remaining_len]]
assert len(vocab_save) == max_vocab_num
vocab_save = ["<pad>", "<s>", "</s>", "<unk>"] + vocab_save

with open(save_vocab_file, 'w') as f:
    f.writelines([w + '\n' for w in vocab_save])

print("saved", save_vocab_file)