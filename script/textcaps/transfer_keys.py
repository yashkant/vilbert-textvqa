from copy import deepcopy
import numpy as np
from tqdm import tqdm

textvqa_imdbs = {
    "train": "/srv/testing/ykant3/pythia/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_train.npy",
    "val": "/srv/testing/ykant3/pythia/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_multidec_sa_info_ascii_val.npy",
    "test": "/srv/testing/ykant3/pythia/imdb/textvqa_0.5/imdb_google_det_bbox_textvqa_info_test.npy",

}

textcaps_imdbs = {
    "train": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_train.npy",
    "val": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_val.npy",
    "test": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_test.npy",
}

# textcaps_imdbs = {
#     "train": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_train_filtered_by_image_id.npy",
#     "val": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_val_filtered_by_image_id.npy",
#     "test": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_test_filtered_by_image_id.npy",
# }
#

image_id_data = {}
keys = ["google_ocr_tokens_filtered"]

for key in textcaps_imdbs.keys():
    tvqa_data = np.load(textvqa_imdbs[key], allow_pickle=True)
    tcaps_data = np.load(textcaps_imdbs[key], allow_pickle=True)

    for item in tqdm(tvqa_data):
        if "image_id" not in item:
            continue
        image_id_data[item["image_id"]] = item

    for item in tqdm(tcaps_data):
        if "image_id" not in item:
            continue
        data = image_id_data[item["image_id"]]

        for k in keys:
            item[k] = deepcopy(data[k])

    # import pdb
    # pdb.set_trace()

    save_path = textcaps_imdbs[key].split(".")[0] + "_proc.npy"
    np.save(save_path, tcaps_data)

