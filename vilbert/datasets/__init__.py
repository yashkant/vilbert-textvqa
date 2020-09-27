# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .textvqa_dataset import TextVQADataset
from .stvqa_dataset import STVQADataset
from .ocrvqa_dataset import OCRVQADataset

DatasetMapTrain = {
    "TextVQA": TextVQADataset,
    "STVQA": STVQADataset,
    "OCRVQA": OCRVQADataset,
}


DatasetMapEval = {
}
