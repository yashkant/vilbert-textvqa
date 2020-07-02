import numpy as np
import os

from tqdm import tqdm

STVQA_IMDBS = [
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_val.npy",
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_train.npy",
    "/srv/share/ykant3/scene-text/test/imdb/test_task3_response_meta_fixed_processed.npy",
]

CLEANED_STVQA_IMDBS = [
    "/srv/share/ykant3/scene-text/train/imdb/stvqa_val.npy",
    "/srv/share/ykant3/scene-text/train/imdb/stvqa_train.npy",
    "/srv/share/ykant3/scene-text/test/imdb/stvqa_test_task3.npy",
]

store_keys = [
    "question",
    "question_id",
    "answers",
    "image_height",
    "image_width",
    "google_ocr_tokens_filtered",
]

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

# stvqa cleaning
for (imdb_path, clean_path) in zip(STVQA_IMDBS, CLEANED_STVQA_IMDBS):
    data = np.load(imdb_path, allow_pickle=True)
    for item in tqdm(data):
        if "question_id" not in item:
            continue
        avail_keys = list(item.keys())
        for key in avail_keys:

            if key == "image_path":
                image_id_new = splitall(item[key])
                image_id_new = f"{image_id_new[-2]}/{image_id_new[-1].split('.')[0]}"
                item["image_id"] = image_id_new

            if key not in store_keys:
                del item[key]


    print(f"Dumped: {clean_path}")
    np.save(clean_path, data)
