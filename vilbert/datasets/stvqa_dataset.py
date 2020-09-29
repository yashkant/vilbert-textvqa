import _pickle as cPickle
import logging

from torch.utils.data import Dataset

from vilbert.datasets.textvqa_dataset import TextVQADataset, Processors, ImageDatabase
from ._image_features_reader import ImageFeaturesH5Reader
from .processors import *

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_dataset(name, debug):
    """Load entries from Imdb

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'

    (YK): We load questions and answers corresponding to
        the splits, and return entries.
    """

    if name == "train" or name == "val" or name == "test":
        imdb_path = f"/srv/share/ykant3/scene-text/train/imdb/train_task_response_meta_fixed_processed_{name}.npy"
        if name == "test":
            imdb_path = "/srv/share/ykant3/scene-text/test/imdb/test_task3_response_meta_fixed_processed.npy"
        if debug:
            imdb_path = f"/srv/share/ykant3/scene-text/train/imdb/debug_train_task_response_meta_fixed_processed_{name}.npy"
        logger.info(f"Loading IMDB for {name}" if not debug else f"Loading IMDB for {name} in debug mode")
        imdb_data = ImageDatabase(imdb_path)
    else:
        assert False, "data split is not recognized."

    # build entries with only the essential keys
    entries = []
    store_keys = [
        "question",
        "question_id",
        "image_path",
        "answers",
        "image_height",
        "image_width",
        "google_ocr_tokens_filtered",
        # "google_ocr_info_filtered",
    ]

    logger.info(f"Building Entries for {name}")
    for instance in imdb_data:
        entry = dict([(key, instance[key]) for key in store_keys if key in instance])
        # Also need to add features-dir
        entry["image_id"] = entry["image_path"].split(".")[0] + ".npy"
        entries.append(entry)

    return entries, imdb_data.metadata


class STVQADataset(TextVQADataset):
    def __init__(
        self,
        split,
        tokenizer,
        padding_index=0,
        max_seq_length=16,
        processing_threads=32,
        extra_args=None
    ):
        """
        (YK): Builds self.entries by reading questions and answers and caches them.
        """

        # Just initialize the grand-parent classs
        Dataset.__init__(self)

        self.split = split
        self._max_seq_length = max_seq_length

        features_split = "trainval" if "test" not in self.split else "test"
        self.obj_features_reader = ImageFeaturesH5Reader(features_path=extra_args["stvqa_obj"].format(features_split))
        self.ocr_features_reader = ImageFeaturesH5Reader(features_path=extra_args["stvqa_ocr"].format(features_split))

        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.max_obj_num = extra_args["max_obj_num"]
        self.max_ocr_num = extra_args["max_ocr_num"]
        self.debug = extra_args.get("debug", False)
        self.vocab_type = extra_args["vocab_type"]
        self.dynamic_sampling = extra_args["dynamic_sampling"]
        self.distance_threshold = extra_args.get("distance_threshold", 0.5)
        self.processing_threads = processing_threads
        self.heads_type = extra_args["SA-M4C"]["heads_type"]
        self.clean_answers = extra_args["clean_answers"]

        registry.vocab_type = self.vocab_type
        registry.distance_threshold = self.distance_threshold

        logger.info(f"Dynamic Sampling is {self.dynamic_sampling}")
        logger.info(f"distance_threshold is {self.distance_threshold}")
        logger.info(f"heads_type: {self.heads_type}")
        logger.info(f"Clean Answers is {self.clean_answers}")

        cache_path = extra_args["stvqa_spatial_cache"].format(self.split)
        logger.info(f"Cache Name:  {cache_path}")

        if not os.path.exists(cache_path) or self.debug:
            # Initialize Processors

            if "processors" not in registry:
                self.processors = Processors(self._tokenizer, vocab_type=self.vocab_type)
                registry.processors = self.processors
            else:
                self.processors = registry.processors

            self.entries, _ = _load_dataset(split, self.debug)
            # convert questions to tokens, create masks, segment_ids
            self.process()
            self.process_spatials()

            if self.heads_type != "none":
                self.process_spatial_extras()
            if not self.debug and False:
                cPickle.dump(self.entries, open(cache_path, "wb"))
                logger.info(f"Cache dumped at: {cache_path}")
        else:
            if "processors_only_registry" not in registry:
                self.processors = Processors(
                    self._tokenizer,
                    only_registry=True,
                    vocab_type=self.vocab_type
                )  # only initialize the M4C processor (for registry)
                registry.processors_only_registry = self.processors
            else:
                self.processors = registry.processors_only_registry

            # otherwise load cache!
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))