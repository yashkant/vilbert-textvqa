from .textvqa_dataset_new import TextVQADataset
from .stvqa_dataset_new import STVQADataset

DatasetMapTrain = {
    "textvqa": TextVQADataset,
    "stvqa": STVQADataset,
}
