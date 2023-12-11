# SMLFM

## To use the code

### Environment

The code is tested with `Python == 3.11`, `PyTorch == 2.0`, and `CUDA == 11.8`. We recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an conda environment:
```bash
conda env create -f environment.yml
conda activate SMLFM
```

### Checkpoints

Please down load the checkpoints from [iCloud](https://www.icloud.com/iclouddrive/012TPd7Lh0VcCFtAog-6d3gYQ#340) for scale up by 4 or [iCloud](https://www.icloud.com/iclouddrive/027xviawbF_oLcFHSc6LvUQFQ#450) for 8. 
Put the checkpoints under the folder `ckpt/e08/` or `ckpt/e10/`.
Note that `e08` and `e10` match the name in [config/e.py](https://github.com/tianrui-qi/SMLFM/blob/main/config/e.py) so that you can check the training configuration for each checkpoint.

### Run

You can check the parameters that must be specified by:
```bash
python main.py --help
```
- `-s`: Scale up factor, 4 or 8. When scale up by 4 or 8, the code will automatically load the corresponding checkpoint from `ckpt/e08/340.ckpt` or `ckpt/e10/450.ckpt`.
- `-L`: Path to the frames load folder.
- `-S`: Path to the data save folder.
- `-b`: Batch size. Set this value according to your GPU memory.

For example, for scale up by 4:
```bash
python main.py -s 4 -L data/frames -S data/dl-444 -b 5
```
or for scale up by 8:
```bash
python main.py -s 8 -L data/frames -S data/dl-488 -b 4
```

Note that the code will predict all the frames under the folder you specified by `-L`.
Thus, if you want to predict portion of the frames, please copy them to a new folder and specify the new folder by `-L`.
