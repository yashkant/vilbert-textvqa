Spatially Aware Multimodal Transformers for TextVQA
===================================================
Existing approaches to solve TextVQA are limited in their use of spatial relations and rely on fully-connected transformer-like architectures to implicitly learn the spatial structure of a scene. 
Rather, we propose a novel spatially aware self-attention layer such that each visual entity only looks at neighboring entities defined by a spatial graph. 
Each head in our multi-head self-attention layer focuses on a different subset of relations.


## Repository Setup

Create a fresh conda environment, and install all dependencies.

```text
conda create -n spat python=3.6
conda activate spat
cd spat-textvqa
pip install -r requirements.txt
```

Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

Finally, install apex from :https://github.com/NVIDIA/apex

## Data Setup
Run the download script to get features and dataset files.
```
cd data
bash download.sh
```

## Run Experiments
From the below table pick the suitable configuration file:

 | Method  |  context (c)   |  Train splits   |  Evaluation Splits  | Config File|
 | ------- | ------ | ------ | ------ | ------ |
 | SA-M4C  | 3 | TextVQA | TextVQA | train-tvqa-eval-tvqa-c3.yml |
 | SA-M4C  | 3 | TextVQA + STVQA | TextVQA | train-tvqa_stvqa-eval-tvqa-c3.yml |
 | SA-M4C  | 3 | STVQA | STVQA | train-stvqa-eval-stvqa-c3.yml |
 | SA-M4C  | 5 | TextVQA | TextVQA | train-tvqa-eval-tvqa-c5.yml |

To run the experiments use:
```
python train.py \
--config config.yml \
--tag experiment-name
```

## Citation
Cite this work as:
```
@article{Kant2020SpatiallyAM,
  title={Spatially Aware Multimodal Transformers for TextVQA},
  author={Yash Kant and Dhruv Batra and Peter Anderson and A. Schwing and D. Parikh and Jiasen Lu and Harsh Agrawal},
  journal={ArXiv},
  year={2020},
  volume={abs/2007.12146}
}
```



## License
BSD
