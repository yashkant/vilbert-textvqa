import numpy as np
import os

from tqdm import tqdm

STVQA_IMDBS = [
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_val.npy",
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_train.npy",
    "/srv/share/ykant3/scene-text/test/imdb/test_task3_response_meta_fixed_processed.npy",
]

TVQA_IMDBS = [
    "../../datasets/textvqa/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_val.npy",
    "../../datasets/textvqa/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_train.npy",
    "../../datasets/textvqa/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_info_test.npy",
]

CLEANED_STVQA_IMDBS = [
    "/srv/share/ykant3/scene-text/train/imdb/stvqa_val.npy",
    "/srv/share/ykant3/scene-text/train/imdb/stvqa_train.npy",
    "/srv/share/ykant3/scene-text/test/imdb/stvqa_test_task3.npy",
]

CLEANED_TVQA_IMDBS = [
    "../../datasets/textvqa/imdb/textvqa_0.5/textvqa_val.npy",
    "../../datasets/textvqa/imdb/textvqa_0.5/textvqa_train.npy",
    "../../datasets/textvqa/imdb/textvqa_0.5/textvqa_test.npy",
]

store_keys = [
    "question",
    "question_id",
    "image_id",
    "answers",
    "image_height",
    "image_width",
    "google_ocr_tokens_filtered",
]

# textvqa cleaning
for (imdb_path, clean_path) in zip(TVQA_IMDBS, CLEANED_TVQA_IMDBS):
    data = np.load(imdb_path, allow_pickle=True)
    for item in tqdm(data):
        if "question_id" not in item:
            continue
        avail_keys = list(item.keys())
        for key in avail_keys:
            if key not in store_keys:
                del item[key]
    print(f"Dumped: {clean_path}")
    np.save(clean_path, data)


# stvqa cleaning
for (imdb_path, clean_path) in zip(STVQA_IMDBS, CLEANED_STVQA_IMDBS):
    data = np.load(imdb_path, allow_pickle=True)
    for item in tqdm(data):
        if "question_id" not in item:
            continue
        avail_keys = list(item.keys())
        for key in avail_keys:
            if key not in store_keys:
                del item[key]
    print(f"Dumped: {clean_path}")
    np.save(clean_path, data)
