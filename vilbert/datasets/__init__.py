from .textvqa_dataset import TextVQADataset
from .stvqa_dataset import STVQADataset

DatasetMapTrain = {
    "TextVQA": TextVQADataset,
    "STVQA": STVQADataset,
}
