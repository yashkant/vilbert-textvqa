import numpy as np

paths_dict = {
    "train": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_train_proc.npy",
    "val": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_val_filtered_by_image_id_proc.npy",
    "test": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_test_filtered_by_image_id_proc.npy",

    "debug_train": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_train_proc_debug.npy",
    "debug_val": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_val_filtered_by_image_id_proc_debug.npy",
    "debug_test": "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_test_filtered_by_image_id_proc_debug.npy",

}

trainval_path = "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_trainval_proc.npy"
minival_path = "/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_minival_filtered_by_image_id_proc.npy"

val_data = np.load("/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_val_proc.npy", allow_pickle=True)
train_data = np.load("/srv/testing/ykant3/pythia/textcaps/m4c_textcaps/imdb_train_proc.npy", allow_pickle=True)




