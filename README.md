# Spatially Aware Multimodal Transformers for TextVQA

Code and pre-trained models for [Spatially Aware Multimodal Transformers for TextVQA](arxiv link here):

```
Citation Here
```

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n spat python=3.6
conda activate spat
cd vilbert-textvqa
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

3. Install apex, follows https://github.com/NVIDIA/apex

4. Install this codebase as a package in this environment.
```text
python setup.py develop
```

## Data Setup

Check `README.md` under `data` for more details.  

## License

vilbert-multi-task is licensed under MIT license available in [LICENSE](LICENSE) file.
