import numpy as np
from tqdm import tqdm


IMDB_SCENETEXT_RESPONSE_FIXED_PROCESSED_SPLIT = [
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_train.npy",
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_val.npy",
    "/srv/share/ykant3/scene-text/test/imdb/test_task3_response_meta_fixed_processed.npy",
]

OCR_FEATURES_SCENETEXT = [
    "/srv/share/ykant3/scene-text/features/ocr/train/train_task/",
    "/srv/share/ykant3/scene-text/features/ocr/test/test_task3/"
]

# image_folders
IMAGES_SCENETEXT = [
    "/srv/share/ykant3/scene-text/train/train_task/",
    "/srv/share/ykant3/scene-text/test/test_task3/"
]

IMAGES_COCOTEXT  = "/srv/share/datasets/coco/train2014/"



imdb_data =[]

for imdb_path in tqdm(IMDB_SCENETEXT_RESPONSE_FIXED_PROCESSED_SPLIT, desc="imdbs"):
    data = np.load(imdb_path, allow_pickle=True)[1:]
    imdb_data.extend(data)

import pdb
pdb.set_trace()

for instance in tqdm(imdb_data):

    # These were images that were never used in task-3 or train-val set!
    # Todo: not fixed for ocr-features
    if "COCO_train2014_000000084731" in instance["image_path"]:
        import pdb
        pdb.set_trace()
    if "COCO_train2014_000000380889" in instance["image_path"]:
        import pdb
        pdb.set_trace()
    if "COCO_train2014_000000194000" in instance["image_path"]:
        import pdb
        pdb.set_trace()

