from typing import List

import h5py

class ImageFeaturesH5Reader(object):
    """
    A reader for H5 files containing pre-extracted image features. A typical
    H5 file is expected to have a column named "image_id", and another column
    named "features".

    Example of an H5 file:
    ```
    faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    # TODO (kd): Add support to read boxes, classes and scores.

    Parameters
    ----------
    features_h5path : str
        Path to an H5 file containing COCO train / val image features.
    in_memory : bool
        Whether to load the whole H5 file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.
    """

    def __init__(self, features_h5path: str, in_memory: bool = False):
        self.features_h5path = features_h5path
        self._in_memory = in_memory

        with h5py.File(self.features_h5path, "r") as features_h5:
            self._split = features_h5.attrs["split"]
            self._image_ids = list(features_h5["image_id"])
            # "features" is List[np.ndarray] if the dataset is loaded in-memory
            # If not loaded in memory, then list of None.
            self.features = [None] * len(self._image_ids)

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id: int):
        index = self._image_ids.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.features[index] is not None:
                features = self.features[index]
            else:
                with h5py.File(self.features_h5path, "r") as features_h5:
                    features = features_h5["features"][index]
                    self.features[index] = features
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.features_h5path, "r") as features_h5:
                features = features_h5["features"][index]

        return features

    def keys(self) -> List[int]:
        return self._image_ids

    @property
    def split(self):
        return self._split