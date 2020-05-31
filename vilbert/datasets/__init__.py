from .textvqa_dataset import TextVQADataset
from .stvqa_dataset import STVQADataset

DatasetMapTrain = {
    "textvqa": TextVQADataset,
    "stvqa": STVQADataset,
}


