"""
The processors exist in Pythia to make data processing pipelines in various
datasets as similar as possible while allowing code reuse.

The processors also help maintain proper abstractions to keep only what matters
inside the dataset's code. This allows us to keep the dataset ``get_item``
logic really clean and no need about maintaining opinions about data type.
Processors can work on both images and text due to their generic structure.

To create a new processor, follow these steps:

1. Inherit the ``BaseProcessor`` class.
2. Implement ``_call`` function which takes in a dict and returns a dict with
   same keys preprocessed as well as any extra keys that need to be returned.
3. Register the processor using ``@registry.register_processor('name')`` to
   registry where 'name' will be used to refer to your processor later.

In processor's config you can specify ``preprocessor`` option to specify
different kind of preprocessors you want in your dataset.

Let's break down processor's config inside a dataset (VQA2.0) a bit to understand
different moving parts.

Config::


    dataset_attributes:
        vqa2:
            processors:
                text_processor:
                type: vocab
                params:
                    max_length: 14
                    vocab:
                    type: intersected
                    embedding_name: glove.6B.300d
                    vocab_file: vocabs/vocabulary_100k.txt
                    answer_processor:
                    type: vqa_answer
                    params:
                        num_answers: 10
                        vocab_file: vocabs/answers_vqa.txt
                        preprocessor:
                        type: simple_word
                        params: {}

``BaseDataset`` will init the processors and they will available inside your
dataset with same attribute name as the key name, for e.g. `text_processor` will
be available as `self.text_processor` inside your dataset. As is with every module
in Pythia, processor also accept a ``ConfigNode`` with a `type` and `params`
attributes. `params` defined the custom parameters for each of the processors.
By default, processor initialization process will also init `preprocessor` attribute
which can be a processor config in itself. `preprocessor` can be then be accessed
inside the processor's functions.

Example::

    from pythia.common.registry import registry
    from pythia.datasets.processors import BaseProcessor


    class MyProcessor(BaseProcessor):
        def __init__(self, config, *args, **kwargs):
            return

        def __call__(self, item, *args, **kwargs):
            text = item['text']
            text = [t.strip() for t in text.split(" ")]
            return {"text": text}
"""
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch
from easydict import EasyDict as edict

from tools.registry import registry
from vilbert.spatial_utils import build_graph_using_normalized_boxes_share
from .textvqa_vocab import VocabDict
from ..phoc import build_phoc
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _pad_tokens(tokens, PAD_TOKEN, max_length):
    padded_tokens = [PAD_TOKEN] * max_length
    token_length = min(len(tokens), max_length)
    padded_tokens[:token_length] = tokens[:token_length]
    token_length = torch.tensor(token_length, dtype=torch.long)
    return padded_tokens, token_length


class WordToVectorDict:
    def __init__(self, model):
        self.model = model

    def __getitem__(self, word):
        # Check if mean for word split needs to be done here
        return np.mean([self.model.get_word_vector(w) for w in word.split(" ")], axis=0)


class BaseProcessor:
    """Every processor in Pythia needs to inherit this class for compatability
    with Pythia. End user mainly needs to implement ``__call__`` function.

    Args:
        config (ConfigNode): Config for this processor, containing `type` and
                             `params` attributes if available.

    """

    def __init__(self, config, *args, **kwargs):
        return

    def __call__(self, item, *args, **kwargs):
        """Main function of the processor. Takes in a dict and returns back
        a dict

        Args:
            item (Dict): Some item that needs to be processed.

        Returns:
            Dict: Processed dict.

        """
        return item


class Processor:
    """Wrapper class used by Pythia to initialized processor based on their
    ``type`` as passed in configuration. It retrieves the processor class
    registered in registry corresponding to the ``type`` key and initializes
    with ``params`` passed in configuration. All functions and attributes of
    the processor initialized are directly available via this class.

    Args:
        config (ConfigNode): ConfigNode containing ``type`` of the processor to
                             be initialized and ``params`` of that procesor.

    """

    def __init__(self, config, *args, **kwargs):
        self.writer = registry.get("writer")

        if not hasattr(config, "type"):
            raise AttributeError(
                "Config must have 'type' attribute to specify type of processor"
            )

        processor_class = registry.get_processor_class(config.type)

        params = {}
        if not hasattr(config, "params"):
            self.writer.write(
                "Config doesn't have 'params' attribute to "
                "specify parameters of the processor "
                "of type {}. Setting to default \{\}".format(config.type)
            )
        else:
            params = config.params

        self.processor = processor_class(params, *args, **kwargs)

        self._dir_representation = dir(self)

    def __call__(self, item, *args, **kwargs):
        return self.processor(item, *args, **kwargs)

    def __getattr__(self, name):
        if name in self._dir_representation:
            return getattr(self, name)
        elif hasattr(self.processor, name):
            return getattr(self.processor, name)
        else:
            raise AttributeError(name)


class VocabProcessor(BaseProcessor):
    """Use VocabProcessor when you have vocab file and you want to process
    words to indices. Expects UNK token as "<unk>" and pads sentences using
    "<pad>" token. Config parameters can have ``preprocessor`` property which
    is used to preprocess the item passed and ``max_length`` property which
    points to maximum length of the sentence/tokens which can be convert to
    indices. If the length is smaller, the sentence will be padded. Parameters
    for "vocab" are necessary to be passed.

    **Key**: vocab

    Example Config::

        task_attributes:
            vqa:
                vqa2:
                    processors:
                      text_processor:
                        type: vocab
                        params:
                          max_length: 14
                          vocab:
                            type: intersected
                            embedding_name: glove.6B.300d
                            vocab_file: vocabs/vocabulary_100k.txt

    Args:
        config (ConfigNode): node containing configuration parameters of
                             the processor

    Attributes:
        vocab (Vocab): Vocab class object which is abstraction over the vocab
                       file passed.
    """

    MAX_LENGTH_DEFAULT = 50
    PAD_TOKEN = "<pad>"
    PAD_INDEX = 0

    def __init__(self, config, *args, **kwargs):
        if not hasattr(config, "vocab"):
            raise AttributeError(
                "config passed to the processor has no attribute vocab"
            )

        self.vocab = Vocab(*args, **config.vocab, **kwargs)
        self._init_extras(config)

    def _init_extras(self, config, *args, **kwargs):
        self.writer = registry.get("writer")
        self.preprocessor = None

        if hasattr(config, "max_length"):
            self.max_length = config.max_length
        else:
            warnings.warn(
                "No 'max_length' parameter in Processor's "
                "configuration. Setting to {}.".format(self.MAX_LENGTH_DEFAULT)
            )
            self.max_length = self.MAX_LENGTH_DEFAULT

        if hasattr(config, "preprocessor"):
            self.preprocessor = Processor(config.preprocessor, *args, **kwargs)

            if self.preprocessor is None:
                raise ValueError(
                    "No text processor named {} is defined.".format(config.preprocessor)
                )

    def __call__(self, item):
        """Call requires item to have either "tokens" attribute or either
        "text" attribute. If "text" is present, it will tokenized using
        the preprocessor.

        Args:
            item (Dict): Dict containing the "text" or "tokens".

        Returns:
            Dict: Dict containing indices in "text" key, "tokens" in "tokens"
                  key and "length" of the string in "length" key.

        """
        indices = None
        if not isinstance(item, dict):
            raise TypeError(
                "Argument passed to the processor must be "
                "a dict with either 'text' or 'tokens' as "
                "keys"
            )
        if "tokens" in item:
            tokens = item["tokens"]
            indices = self._map_strings_to_indices(item["tokens"])
        elif "text" in item:
            if self.preprocessor is None:
                raise AssertionError(
                    "If tokens are not provided, a text "
                    "processor must be defined in the config"
                )

            tokens = self.preprocessor({"text": item["text"]})["text"]
            indices = self._map_strings_to_indices(tokens)
        else:
            raise AssertionError(
                "A dict with either 'text' or 'tokens' keys "
                "must be passed to the processor"
            )

        tokens, length = self._pad_tokens(tokens)

        return {"text": indices, "tokens": tokens, "length": length}

    def _pad_tokens(self, tokens):
        padded_tokens = [self.PAD_TOKEN] * self.max_length
        token_length = min(len(tokens), self.max_length)
        padded_tokens[:token_length] = tokens[:token_length]
        token_length = torch.tensor(token_length, dtype=torch.long)
        return padded_tokens, token_length

    def get_pad_index(self):
        """Get index of padding <pad> token in vocabulary.

        Returns:
            int: index of the padding token.

        """
        return self.vocab.get_pad_index()

    def get_vocab_size(self):
        """Get size of the vocabulary.

        Returns:
            int: size of the vocabulary.

        """
        return self.vocab.get_size()

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.zeros(self.max_length, dtype=torch.long)
        output.fill_(self.vocab.get_pad_index())

        for idx, token in enumerate(tokens):
            output[idx] = self.vocab.stoi[token]

        return output


class FastTextProcessor:
    """FastText processor, similar to GloVe processor but returns FastText vectors.

    Args:
        config (ConfigNode): Configuration values for the processor.

    """

    def __init__(self, config, *args, **kwargs):
        self.max_length = config.max_length
        self._load_fasttext_model("/srv/share/ykant3/pythia/vector_cache/wiki.en.bin")
        self.PAD_INDEX = 0
        self.PAD_TOKEN = "<pad>"


    def _load_fasttext_model(self, model_file):
        from fasttext import load_model
        self.model = load_model(model_file)
        # String to Vector
        self.stov = WordToVectorDict(self.model)

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.full(
            (self.max_length, self.model.get_dimension()),
            fill_value=self.PAD_INDEX,
            dtype=torch.float,
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(self.stov[token])

        return output

    def __call__(self, item):
        # indices are padded
        indices = self._map_strings_to_indices(item["tokens"])
        # pad tokens
        tokens, length = _pad_tokens(item["tokens"], self.PAD_TOKEN, self.max_length)
        return {"padded_token_indices": indices, "padded_tokens": tokens, "length": length}


class VQAAnswerProcessor(BaseProcessor):
    """Processor for generating answer scores for answers passed using VQA
    accuracy formula. Using VocabDict class to represent answer vocabulary,
    so parameters must specify "vocab_file". "num_answers" in parameter config
    specify the max number of answers possible. Takes in dict containing
    "answers" or "answers_tokens". "answers" are preprocessed to generate
    "answers_tokens" if passed.

    Args:
        config (ConfigNode): Configuration for the processor

    Attributes:
        answer_vocab (VocabDict): Class representing answer vocabulary
    """

    DEFAULT_NUM_ANSWERS = 10

    def __init__(self, config, *args, **kwargs):
        self.writer = registry.get("writer")
        if not hasattr(config, "vocab_file"):
            raise AttributeError(
                "'vocab_file' argument required, but not "
                "present in AnswerProcessor's config"
            )

        self.answer_vocab = VocabDict(config.vocab_file, *args, **kwargs)

        self.preprocessor = None

        if hasattr(config, "preprocessor"):
            self.preprocessor = Processor(config.preprocessor)

            if self.preprocessor is None:
                raise ValueError(
                    "No processor named {} is defined.".format(config.preprocessor)
                )

        if hasattr(config, "num_answers"):
            self.num_answers = config.num_answers
        else:
            self.num_answers = self.DEFAULT_NUM_ANSWERS
            warnings.warn(
                "'num_answers' not defined in the config. "
                "Setting to default of {}".format(self.DEFAULT_NUM_ANSWERS)
            )

    def __call__(self, item):
        """Takes in dict with answers or answers_tokens, and returns back
        a dict with answers (processed), "answers_indices" which point to
        indices of the answers if present and "answers_scores" which represent
        VQA style scores for the answers.

        Args:
            item (Dict): Dict containing answers or answers_tokens

        Returns:
            Dict: Processed answers, indices and scores.

        """
        tokens = None

        if not isinstance(item, dict):
            raise TypeError("'item' passed to processor must be a dict")

        if "answer_tokens" in item:
            tokens = item["answer_tokens"]
        elif "answers" in item:
            if self.preprocessor is None:
                raise AssertionError(
                    "'preprocessor' must be defined if you "
                    "don't pass 'answer_tokens'"
                )

            tokens = [
                self.preprocessor({"text": answer})["text"]
                for answer in item["answers"]
            ]
        else:
            raise AssertionError(
                "'answers' or 'answer_tokens' must be passed"
                " to answer processor in a dict"
            )

        tokens = self._increase_to_ten(tokens)
        answers_indices = torch.zeros(self.DEFAULT_NUM_ANSWERS, dtype=torch.long)
        answers_indices.fill_(self.answer_vocab.get_unk_index())

        for idx, token in enumerate(tokens):
            answers_indices[idx] = self.answer_vocab.word2idx(token)

        answers_scores = self.compute_answers_scores(answers_indices)

        return {
            "answers": tokens,
            "answers_indices": answers_indices,
            "answers_scores": answers_scores,
        }

    def get_vocab_size(self):
        """Get vocab size of the answer vocabulary. Can also include
        soft copy dynamic answer space size.

        Returns:
            int: size of the answer vocabulary

        """
        return self.answer_vocab.num_vocab

    def get_true_vocab_size(self):
        """True vocab size can be different from normal vocab size in some cases
        such as soft copy where dynamic answer space is added.

        Returns:
            int: True vocab size.

        """
        return self.answer_vocab.num_vocab

    def word2idx(self, word):
        """Convert a word to its index according to vocabulary

        Args:
            word (str): Word to be converted to index.

        Returns:
            int: Index of the word.

        """
        return self.answer_vocab.word2idx(word)

    def idx2word(self, idx):
        """Index to word according to the vocabulary.

        Args:
            idx (int): Index to be converted to the word.

        Returns:
            str: Word corresponding to the index.

        """
        return self.answer_vocab.idx2word(idx)

    def compute_answers_scores(self, answers_indices):
        """Generate VQA based answer scores for answers_indices.

        Args:
            answers_indices (torch.LongTensor): tensor containing indices of the answers

        Returns:
            torch.FloatTensor: tensor containing scores.

        """
        scores = torch.zeros(self.get_vocab_size(), dtype=torch.float)
        gt_answers = list(enumerate(answers_indices))
        unique_answers = set(answers_indices.tolist())

        for answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]

                matching_answers = [item for item in other_answers if item[1] == answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            avg_acc = sum(accs) / len(accs)

            if answer != self.answer_vocab.UNK_INDEX:
                scores[answer] = avg_acc

        return scores

    def _increase_to_ten(self, tokens):
        while len(tokens) < self.DEFAULT_NUM_ANSWERS:
            tokens += tokens[:self.DEFAULT_NUM_ANSWERS - len(tokens)]

        return tokens


class MultiHotAnswerFromVocabProcessor(VQAAnswerProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def compute_answers_scores(self, answers_indices):
        scores = torch.zeros(self.get_vocab_size(), dtype=torch.float)
        scores[answers_indices] = 1
        scores[self.answer_vocab.UNK_INDEX] = 0
        return scores


class SoftCopyAnswerProcessor(VQAAnswerProcessor):
    """Similar to Answer Processor but adds soft copy dynamic answer space to it.
    Read https://arxiv.org/abs/1904.08920 for extra information on soft copy
    and LoRRA.

    Args:
        config (ConfigNode): Configuration for soft copy processor.

    """

    DEFAULT_MAX_LENGTH = 50

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        if hasattr(config, "max_length"):
            self.max_length = config.max_length
        else:
            self.max_length = self.DEFAULT_MAX_LENGTH
            warnings.warn(
                "'max_length' not defined in the config. "
                "Setting to default of {}".format(self.DEFAULT_MAX_LENGTH)
            )

        self.context_preprocessor = None
        if hasattr(config, "context_preprocessor"):
            self.context_preprocessor = Processor(config.context_preprocessor)

    def get_vocab_size(self):
        """Size of Vocab + Size of Dynamic soft-copy based answer space

        Returns:
            int: Size of vocab + size of dynamic soft-copy answer space.

        """
        answer_vocab_nums = self.answer_vocab.num_vocab
        answer_vocab_nums += self.max_length

        return answer_vocab_nums

    def get_true_vocab_size(self):
        """Actual vocab size which only include size of the vocabulary file.

        Returns:
            int: Actual size of vocabs.

        """
        return self.answer_vocab.num_vocab

    def __call__(self, item):
        answers = item["answers"]
        scores = super().__call__({"answers": answers})

        indices = scores["answers_indices"]
        answers = scores["answers"]
        scores = scores["answers_scores"]

        tokens_scores = scores.new_zeros(self.max_length)
        tokens = item["tokens"]
        length = min(len(tokens), self.max_length)

        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)

        if self.context_preprocessor is not None:
            tokens = [
                self.context_preprocessor({"text": token})["text"] for token in tokens
            ]

        answer_counter = Counter(answers)

        for idx, token in enumerate(tokens[:length]):
            if answer_counter[token] == 0:
                continue
            accs = []

            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == token]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)

            tokens_scores[idx] = sum(accs) / len(accs)

        # Scores are already proper size, see L314. Now,
        # fix scores for soft copy candidates
        scores[-len(tokens_scores) :] = tokens_scores
        return {
            "answers": answers,
            "answers_indices": indices,
            "answers_scores": scores,
        }


class SimpleWordProcessor(BaseProcessor):
    """Tokenizes a word and processes it.

    Attributes:
        tokenizer (function): Type of tokenizer to be used.

    """

    def __init__(self, *args, **kwargs):
        from pythia.utils.text_utils import word_tokenize

        self.tokenizer = word_tokenize

    def __call__(self, item, *args, **kwargs):
        return {"text": self.tokenizer(item["text"], *args, **kwargs)}


class SimpleSentenceProcessor(BaseProcessor):
    """Tokenizes a sentence and processes it.

    Attributes:
        tokenizer (function): Type of tokenizer to be used.

    """

    def __init__(self, *args, **kwargs):
        from pythia.utils.text_utils import tokenize

        self.tokenizer = tokenize

    def __call__(self, item, *args, **kwargs):
        return {"text": self.tokenizer(item["text"], *args, **kwargs)}


class BBoxProcessor(VocabProcessor):
    """Generates bboxes in proper format.
    Takes in a dict which contains "info" key which is a list of dicts
    containing following for each of the the bounding box

    Example bbox input::

        {
            "info": [
                {
                    "bounding_box": {
                        "top_left_x": 100,
                        "top_left_y": 100,
                        "width": 200,
                        "height": 300
                    }
                },
                ...
            ]
        }


    This will further return a Sample in a dict with key "bbox" with last
    dimension of 4 corresponding to "xyxy". So sample will look like following:

    Example Sample::

        Sample({
            "coordinates": torch.Size(n, 4),
            "width": List[number], # size n
            "height": List[number], # size n
            "bbox_types": List[str] # size n, either xyxy or xywh.
            # currently only supports xyxy.
        })

    """

    def __init__(self, config, *args, **kwargs):
        from pythia.utils.dataset_utils import build_bbox_tensors

        self.lambda_fn = build_bbox_tensors
        self._init_extras(config)

    def __call__(self, item):
        info = item["info"]
        if self.preprocessor is not None:
            info = self.preprocessor(info)

        return {"bbox": self.lambda_fn(info, self.max_length)}


class CaptionProcessor(BaseProcessor):
    """Processes a caption with start, end and pad tokens and returns raw string.

    Args:
        config (ConfigNode): Configuration for caption processor.

    """

    def __init__(self,  config, *args, **kwargs):
        if not hasattr(config, "vocab"):
            raise AttributeError(
                "config passed to the processor has no " "attribute vocab"
            )

        self.vocab = Vocab(*args, **config.vocab, **kwargs)

    def __call__(self, item):
        for idx, v in enumerate(item):
            if v == self.vocab.EOS_INDEX:
                item = item[:idx]
                break
        tokens = [
            self.vocab.get_itos()[w]
            for w in item
            if w
            not in {self.vocab.SOS_INDEX, self.vocab.EOS_INDEX, self.vocab.PAD_INDEX}
        ]
        caption = " ".join(tokens)
        return {"tokens": tokens, "caption": caption}


class PhocProcessor:
    """
    Compute PHOC features from text tokens
    """
    def __init__(self, config, *args, **kwargs):
        self.max_length = config.max_length
        self.config = config
        self.PAD_INDEX = 0
        self.PAD_TOKEN = "<pad>"


    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        phoc_dim = 604
        output = torch.full(
            (self.max_length, phoc_dim),
            fill_value=self.PAD_INDEX,
            dtype=torch.float,
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(build_phoc(token))

        return output

    def __call__(self, item):
        indices = self._map_strings_to_indices(item["tokens"])
        tokens, length = _pad_tokens(item["tokens"], self.PAD_TOKEN, self.max_length)
        return {"padded_phoc_features": indices, "padded_tokens": tokens, "length": length}


class CopyProcessor(BaseProcessor):
    """
    Copy boxes from numpy array
    """
    def __init__(self, config, *args, **kwargs):
        self.max_length = config.max_length

    def __call__(self, item):
        blob = item["blob"]
        final_blob = np.zeros((self.max_length,) + blob.shape[1:], blob.dtype)
        final_blob[:len(blob)] = blob[:len(final_blob)]

        return {"blob": torch.from_numpy(final_blob)}


def SpatialProcessor(pad_obj_ocr_bboxes):
    adj_matrix_shared = build_graph_using_normalized_boxes_share(
        pad_obj_ocr_bboxes,
        distance_threshold=registry.distance_threshold,
        )
    return adj_matrix_shared


def RandomSpatialProcessor(pad_obj_ocr_bboxes):
    randomize = registry.randomize
    adj_matrix_shape = (len(pad_obj_ocr_bboxes), len(pad_obj_ocr_bboxes), randomize)
    adj_matrix = np.zeros(adj_matrix_shape, dtype=np.int8)
    spatial_relations_types = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    rev_replace_dict = {}
    for quad in [4, 5, 6, 7]:
        rev_replace_dict[quad] = quad + 4
        rev_replace_dict[quad + 4] = quad
    rev_replace_dict[2] = 1
    rev_replace_dict[1] = 2

    for row in range(adj_matrix_shape[0]):
        for col in range(row, adj_matrix_shape[1]):
            random_indices = np.random.choice(spatial_relations_types, size=randomize, replace=False)
            # remove none-edges
            if 0 not in random_indices:
                adj_matrix[row][col] = random_indices
                adj_matrix[col][row] = [rev_replace_dict[x] for x in random_indices]

    # Remove masked relations
    masked_inds = np.where(pad_obj_ocr_bboxes.sum(axis=-1) == 0)
    adj_matrix[masked_inds] = 0
    adj_matrix[:, masked_inds] = 0

    return adj_matrix


class BertTokenizerProcessor:
    """
    Tokenize a text string with BERT tokenizer, using Tokenizer passed to the dataset.
    """
    def __init__(self, config, tokenizer):
        self.max_length = config.max_length
        self.bert_tokenizer = tokenizer
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        assert self.bert_tokenizer.encode(self.bert_tokenizer.pad_token) == [0]

    def get_vocab_size(self):
        return self.bert_tokenizer.vocab_size

    def __call__(self, item):
        # [PAD] in self.bert_tokenizer is zero (as checked in assert above)
        token_inds = torch.zeros(self.max_length, dtype=torch.long)

        indices = self.bert_tokenizer.encode(
            item['question'], add_special_tokens=True)
        indices = indices[:self.max_length]
        token_inds[:len(indices)] = torch.tensor(indices)
        token_num = torch.tensor(len(indices), dtype=torch.long)

        tokens_mask = torch.zeros(self.max_length, dtype=torch.long)
        tokens_mask[:len(indices)] = 1

        results = {'token_inds': token_inds, 'token_num': token_num, "tokens_mask": tokens_mask}
        return results


class M4CAnswerProcessor:
    """
    Process a TextVQA answer for iterative decoding in M4C.
    # (YK): Modified to activate logits of the same word in ocr/vocabulary in targets.
    """
    def __init__(self, config, *args, **kwargs):
        vocab5k = "/srv/share/ykant3/m4c-release/data/m4c_vocabs/textvqa/fixed_answer_vocab_textvqa_5k.txt"
        vocab4k = "/srv/share/ykant3/pythia/vocabs/answers_textvqa_more_than_1_m4c_no_dups.txt"
        vocab4k_filtered = "/srv/share/ykant3/pythia/vocabs/answers_textvqa_more_than_1_m4c_no_dups_filtered.txt"
        vocab_none = "/srv/share/ykant3/pythia/vocabs/vocab_none.txt"
        vocab4k_latest = "/srv/share/ykant3/pythia/vocabs/answer_vocab_textvqa_4k_filtered.txt"
        vocab5k_stvqa = "/srv/share/ykant3/m4c-release/data/m4c_vocabs/stvqa/fixed_answer_vocab_stvqa_5k.txt"
        vocab_ocrvqa = "/srv/share/ykant3/m4c-release/data/m4c_vocabs/ocrvqa/fixed_answer_vocab_ocrvqa_82.txt"

        if config.vocab_type == "5k":
            self.answer_vocab = VocabDict(vocab5k, *args, **kwargs)
        elif config.vocab_type == "4k_filtered":
            self.answer_vocab = VocabDict(vocab4k_filtered, *args, **kwargs)
        elif config.vocab_type == "4k":
            self.answer_vocab = VocabDict(vocab4k, *args, **kwargs)
        elif config.vocab_type == "none":
            self.answer_vocab = VocabDict(vocab_none, *args, **kwargs)
        elif config.vocab_type == "4k_latest":
            self.answer_vocab = VocabDict(vocab4k_latest, *args, **kwargs)
        elif config.vocab_type == "5k_stvqa":
            self.answer_vocab = VocabDict(vocab5k_stvqa, *args, **kwargs)
        elif config.vocab_type == "ocrvqa":
            self.answer_vocab = VocabDict(vocab_ocrvqa, *args, **kwargs)
        else:
            raise ValueError

        self.PAD_IDX = self.answer_vocab.word2idx('<pad>')
        self.BOS_IDX = self.answer_vocab.word2idx('<s>')
        self.EOS_IDX = self.answer_vocab.word2idx('</s>')
        self.UNK_IDX = self.answer_vocab.UNK_INDEX

        registry.PAD_IDX = self.answer_vocab.word2idx('<pad>')
        registry.BOS_IDX = self.answer_vocab.word2idx('<s>')
        registry.EOS_IDX = self.answer_vocab.word2idx('</s>')
        registry.UNK_IDX = self.answer_vocab.UNK_INDEX
        registry.answer_vocab = self.answer_vocab

        # make sure PAD_IDX, BOS_IDX and PAD_IDX are valid (not <unk>)
        assert self.PAD_IDX != self.answer_vocab.UNK_INDEX
        assert self.BOS_IDX != self.answer_vocab.UNK_INDEX
        assert self.EOS_IDX != self.answer_vocab.UNK_INDEX
        assert self.PAD_IDX == 0

        self.num_answers = config.num_answers
        self.max_ocr_tokens = config.max_ocr_tokens
        self.max_copy_steps = config.max_copy_steps
        assert self.max_copy_steps >= 1

    def match_answer_to_vocab_ocr_seq(
        self, answer, vocab2idx_dict, ocr2inds_dict, max_match_num=20
    ):
        """
        Match an answer to a list of sequences of indices
        each index corresponds to either a fixed vocabulary or an OCR token
        (in the index address space, the OCR tokens are after the fixed vocab)
        """
        num_vocab = len(vocab2idx_dict)

        answer_words = answer.split()
        answer_word_matches = []
        for word in answer_words:
            # match answer word to fixed vocabulary
            matched_inds = []
            if word in vocab2idx_dict:
                matched_inds.append(vocab2idx_dict.get(word))
            # match answer word to OCR
            # we put OCR after the fixed vocabulary in the answer index space
            # so add num_vocab offset to the OCR index
            matched_inds.extend(
                [num_vocab + idx for idx in ocr2inds_dict[word]]
            )
            if len(matched_inds) == 0:
                return []
            answer_word_matches.append(matched_inds)

        # expand per-word matched indices into the list of matched sequences
        if len(answer_word_matches) == 0:
            return []
        idx_seq_list = [()]
        for matched_inds in answer_word_matches:
            idx_seq_list = [
                seq + (idx,)
                for seq in idx_seq_list for idx in matched_inds
            ]
            if len(idx_seq_list) > max_match_num:
                idx_seq_list = idx_seq_list[:max_match_num]

        return idx_seq_list

    def get_vocab_size(self):
        answer_vocab_nums = self.answer_vocab.num_vocab
        answer_vocab_nums += self.max_ocr_tokens

        return answer_vocab_nums

    def __call__(self, item):
        answers = item["answers"]
        item["context_tokens"] = item["context_tokens"][:self.max_ocr_tokens]
        assert len(answers) == self.num_answers
        assert len(self.answer_vocab) == len(self.answer_vocab.word2idx_dict)

        # Step 1: calculate the soft score of ground-truth answers
        gt_answers = list(enumerate(answers))
        unique_answers = sorted(set(answers))
        unique_answer_scores = [0] * len(unique_answers)
        for idx, unique_answer in enumerate(unique_answers):
            accs = []
            for gt_answer in gt_answers:
                other_answers = [
                    item for item in gt_answers if item != gt_answer
                ]
                matching_answers = [
                    item for item in other_answers if item[1] == unique_answer
                ]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[idx] = sum(accs) / len(accs)
        unique_answer2score = {
            a: s for a, s in zip(unique_answers, unique_answer_scores)
        }

        # Step 2: fill the first step soft scores for tokens
        scores = torch.zeros(
            self.max_copy_steps,
            self.get_vocab_size(),
            dtype=torch.float
        )

        # match answers to fixed vocabularies and OCR tokens.
        ocr2inds_dict = defaultdict(list)
        for idx, token in enumerate(item["context_tokens"]):
            ocr2inds_dict[token].append(idx)
        answer_dec_inds = [
            self.match_answer_to_vocab_ocr_seq(
                a, self.answer_vocab.word2idx_dict, ocr2inds_dict
            ) for a in answers
        ]

        # Collect all the valid decoding sequences for each answer.
        # This part (idx_seq_list) was pre-computed in imdb (instead of online)
        # to save time
        all_idx_seq_list = []
        for answer, idx_seq_list in zip(answers, answer_dec_inds):
            all_idx_seq_list.extend(idx_seq_list)
            # fill in the soft score for the first decoding step
            score = unique_answer2score[answer]
            for idx_seq in idx_seq_list:
                score_idx = idx_seq[0]
                # the scores for the decoding Step 0 will be the maximum
                # among all answers starting with that vocab
                # for example:
                # if "red apple" has score 0.7 and "red flag" has score 0.8
                # the score for "red" at Step 0 will be max(0.7, 0.8) = 0.8
                try:
                    scores[0, score_idx] = max(scores[0, score_idx], score)
                except:
                    import pdb
                    pdb.set_trace()

        # train_prev_inds is the previous prediction indices in auto-regressive
        # decoding
        train_prev_inds = torch.zeros(self.max_copy_steps, dtype=torch.long)
        # train_loss_mask records the decoding steps where losses are applied
        train_loss_mask = torch.zeros(self.max_copy_steps, dtype=torch.float)
        train_acc_mask = torch.zeros(self.max_copy_steps, dtype=torch.float)

        if len(all_idx_seq_list) > 0:
            # sample a random decoding answer sequence for teacher-forcing
            idx_seq = all_idx_seq_list[np.random.choice(len(all_idx_seq_list))]
            dec_step_num = min(1+len(idx_seq), self.max_copy_steps)
            train_loss_mask[:dec_step_num] = 1.
            train_acc_mask[:dec_step_num-1] = 1.

            train_prev_inds[0] = self.BOS_IDX
            for t in range(1, dec_step_num):
                train_prev_inds[t] = idx_seq[t-1]
                score_idx = idx_seq[t] if t < len(idx_seq) else self.EOS_IDX
                # if item["question_id"] == 35909:
                #     import pdb
                #     pdb.set_trace()
                # this means step 1:N have only one non-zero index
                # this means there will be no case with EOS_IDX_SCORE and OTHER score non-zero together!
                # gather indices from both ocr/vocabulary for the same word!
                all_indices = self.get_all_indices(ocr2inds_dict, item["context_tokens"], score_idx)
                assert self.UNK_IDX not in all_indices

                for idx in all_indices:
                    scores[t, idx] = 1.

                # scores[t, score_idx] = 1.
        else:
            idx_seq = ()

        answer_info = {
            'answers': answers,
            'targets': scores,
            # 'sampled_idx_seq': [train_prev_inds.new(idx_seq)],
            'train_prev_inds': train_prev_inds,
            'train_loss_mask': train_loss_mask,
            'train_acc_mask': train_acc_mask,
        }
        return answer_info

    def get_all_indices(self, ocr2indices, ocr_tokens, score_idx):
        return_indices = [score_idx]
        if score_idx >= len(self.answer_vocab):
            word = ocr_tokens[score_idx - len(self.answer_vocab)]
            assert word != "<pad>"
            vocab_idx = self.answer_vocab.word2idx(word)
            if vocab_idx != self.UNK_IDX:
                return_indices.append(vocab_idx)
        else:
            word = self.answer_vocab.idx2word(score_idx)
            ocr_indices = [x+len(self.answer_vocab) for x in ocr2indices[word]]
            return_indices.extend(ocr_indices)

        return return_indices


class Processors:
    """
    Contains static-processors used for processing question/ocr-tokens, image/ocr features,
        decoding answer.
    """

    def __init__(self, bert_tokenizer, vocab_type="4k", only_registry=False):
        logger.info("Loading Processors")
        logger.info(f"Vocab Type: {vocab_type}")
        # decode-answers
        answer_config = edict()
        answer_config.max_copy_steps = 12
        answer_config.num_answers = 10
        answer_config.max_ocr_tokens = 50
        answer_config.vocab_type = vocab_type
        self.answer_processor = M4CAnswerProcessor(answer_config)
        self.only_registry = only_registry

        # Attach bert-tokenizer
        registry["bert_tokenizer"] = bert_tokenizer

        if only_registry:
            logger.info("Only registry processor initialized")
            return

        # question
        question_config = edict()
        question_config.max_length = 20
        self.bert_processor = BertTokenizerProcessor(question_config, bert_tokenizer)

        # ocr-tokens
        ocr_config = edict()
        ocr_config.max_length = 50
        self.fasttext_processor = FastTextProcessor(ocr_config)
        self.phoc_processor = PhocProcessor(ocr_config)

    @staticmethod
    def word_cleaner(word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    @staticmethod
    def word_cleaner_lower(word):
        word = word.lower()
        return word.strip()


