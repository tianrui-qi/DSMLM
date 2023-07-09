# DL-SMLFM

## Installation

The code is tested with Python == 3.11, PyTorch == 2.0, and CUDA == 11.8. We recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an conda environment:
```
conda env create -f environment.yml
conda activate DL-SMLFM
```

## Data



## Evaluation

First download our pretrained model [here](https://gtvault-my.sharepoint.com/:f:/g/personal/tqi36_gatech_edu/Esv2NpS63PxApndfaBYz_AABn3lr3KuHDO7hxpWYIpCUIA?e=YNsQxj) where `test_7.pt` is the checkpoint with best loss during training and checkpoint inside folder `test_7` is the checkpoint we save during training process after each epoch, i.e., `test_7/i.pt` is the checkpoint after i's epoch. Download one or more checkpoint(s) you want to use to evaluate and specify the checkpoint path (without `.pt`) in `self.cpt_load_path` of class `Eval` of [config.py](https://github.com/tianrui-qi/DL-SMLFM/blob/main/config.py). For example, if the file structure is

```
DL-SMLFM
├── config.py
├── checkpoints
│   └── test_7.pt
├── ...
```

set `self.cpt_load_path` of class `Eval` of [config.py](https://github.com/tianrui-qi/DL-SMLFM/blob/main/config.py) as

```python
self.cpt_load_path = 'checkpoints/test_7'
```

Make sure you set up correct `self.batch_size` to match the memory of your GPU and the correct `self.num` to match the number of data after crop (not raw!!!) that wait for evaluation. Note that `self.batch_size` must be a divisor of `self.num`. Then run

```bash
python eval.py
```

## Training
