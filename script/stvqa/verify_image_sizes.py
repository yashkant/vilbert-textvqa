from tqdm import tqdm

IMDB_SCENETEXT_RESPONSE_FIXED_PROCESSED_SPLIT = [
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_train.npy",
    "/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_val.npy",
    "/srv/share/ykant3/scene-text/test/imdb/test_task3_response_meta_fixed_processed.npy",
]

IMDB_M4C = [
    "/srv/share/ykant3/m4c-release/data/imdb/m4c_stvqa/imdb_subtrain.npy",
    "/srv/share/ykant3/m4c-release/data/imdb/m4c_stvqa/imdb_subval.npy",
    "/srv/share/ykant3/m4c-release/data/imdb/m4c_stvqa/imdb_test_task3.npy",
]

import numpy as np

for st_imdb, m4c_imdb in zip(IMDB_SCENETEXT_RESPONSE_FIXED_PROCESSED_SPLIT, IMDB_M4C):
    m4c_imdb, st_imdb = np.load(m4c_imdb, allow_pickle=True), np.load(st_imdb, allow_pickle=True)
    assert len(m4c_imdb) == len(st_imdb)

    qids1 = []
    qids2 = []

    for item_m4c, item_st in tqdm(zip(m4c_imdb[1:], st_imdb[1:]), total=len(m4c_imdb)-1):

        # if item_m4c["question_id"] != item_st["question_id"]:
        #     import pdb
        #     pdb.set_trace()
        # print("Question: ", item_st["question"], item_m4c["question"])
        # print("Answer: ", item_st["answers"], item_m4c["answers"])
        #     print("image_paths", item_m4c["image_path"], item_st["image_path"])
        
        assert item_st["question"] == item_m4c["question"]
        # assert item_st["answers"] == item_m4c["answers"]
        assert item_st["question_id"] == item_m4c["question_id"]

        qids1.append(item_m4c["question_id"])
        qids2.append(item_st["question_id"])

        # print("Question ids: ", item_st["question_id"], item_m4c["question_id"])
        # try:
        #     assert item_m4c["image_height"] == item_st["image_height"]
        #     assert item_m4c["image_width"] == item_st["image_width"]
        # except:
        #     try:
        #         assert item_m4c["image_height"] == item_st["image_width"]
        #         assert item_m4c["image_width"] == item_st["image_height"]
        #     except:
        #         import pdb
        #         pdb.set_trace()
        #
        #     print("Heights", item_m4c["image_height"], item_st["image_height"])
        #     print("image_width", item_m4c["image_width"], item_st["image_width"])


    print(len(set(qids1)))
    print(len(set(qids2)))
import pdb
pdb.set_trace()
