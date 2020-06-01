# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .concept_cap_dataset import (
    ConceptCapLoaderTrain,
    ConceptCapLoaderVal,
    ConceptCapLoaderRetrieval,
)
from .foil_dataset import FoilClassificationDataset
from .vqa_dataset import VQAClassificationDataset
from .vqa_mc_dataset import VQAMultipleChoiceDataset
from .nlvr2_dataset import NLVR2Dataset
from .retreival_dataset import RetreivalDataset, RetreivalDatasetVal
from .vcr_dataset import VCRDataset
from .visdial_dataset import VisDialDataset
from .visual_entailment_dataset import VisualEntailmentDataset
from .visual_genome_dataset import GenomeQAClassificationDataset
from .gqa_dataset import GQAClassificationDataset
from .flickr_grounding_dataset import FlickrGroundingDataset

# from .flickr_retreival_dataset import FlickrRetreivalDatasetTrain, FlickrRetreivalDatasetVal
__all__ = [
    "FoilClassificationDataset",
    "VQAClassificationDataset",
    "GenomeQAClassificationDataset",
    "VQAMultipleChoiceDataset",
    "ConceptCapLoaderTrain",
    "ConceptCapLoaderVal",
    "NLVR2Dataset",
    "RetreivalDataset",
    "RetreivalDatasetVal",
    "VCRDataset",
    "VisDialDataset",
    "VisualEntailmentDataset",
    "GQAClassificationDataset",
    "ConceptCapLoaderRetrieval",
    "FlickrGroundingDataset",
    "",
]

DatasetMapTrain = {
    "VQA": VQAClassificationDataset,
    "GenomeQA": GenomeQAClassificationDataset,
    "VisualDialog": VisDialDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalCOCO": RetreivalDataset,
    "RetrievalFlickr30k": RetreivalDataset,
    "NLVR2": NLVR2Dataset,
    "VisualEntailment": VisualEntailmentDataset,
    "GQA": GQAClassificationDataset,
    "Foil": FoilClassificationDataset,
    "FlickrGrounding": FlickrGroundingDataset,
}


DatasetMapEval = {
    "VQA": VQAClassificationDataset,
    "GenomeQA": GenomeQAClassificationDataset,
    "VisualDialog": VisDialDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalCOCO": RetreivalDatasetVal,
    "RetrievalFlickr30k": RetreivalDatasetVal,
    "NLVR2": NLVR2Dataset,
    "VisualEntailment": VisualEntailmentDataset,
    "GQA": GQAClassificationDataset,
    "Foil": FoilClassificationDataset,
    "FlickrGrounding": FlickrGroundingDataset,
}
