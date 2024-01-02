# SMLFM

## To use the code

### Environment

The code is tested with `Python == 3.11`, `PyTorch == 2.0`, and `CUDA == 11.8`. We recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an conda environment:
```bash
conda env create -f environment.yml
conda activate SMLFM
```

### Checkpoints

Please down load the checkpoints from [iCloud](https://www.icloud.com/iclouddrive/05cFlVujbb2TkrWANiT04tdgQ#340) for scale up by 4 or [iCloud](https://www.icloud.com/iclouddrive/0e6maAxyFbHaA3MIGSYuivcOw#450) for 8. 
Put the checkpoints under the folder `ckpt/e08/` or `ckpt/e10/`.
Note that `e08` and `e10` match the name in [config/e.py](https://github.com/tianrui-qi/SMLFM/blob/main/config/e.py) so that you can check the training configuration for each checkpoint.

### Run

You can check the parameters that must be specified by:
```bash
python main.py --help
```
- `-s`: Scale up factor, 4 or 8. Default: 4.
- `-L`: Path to the frames load folder.
- `-S`: Path to the data save folder.
- `-C`: Path to the checkpoint load file without .ckpt. Default: `ckpt/e08/340` or `ckpt/e10/450` when scale up factor is 4 or 8.
- `-T`: Path to the temporary save folder for drifting analysis. Must be specified when drift correction will be performed, i.e., stride or window is set as non-zero. Default: None.
- `-stride`: Step size of the drift corrector, unit frames. Shall not be set with -window at the same time. Default: 0.
- `-window`: Number of frames in each window, unit frames. Sall not be set with -stride at the same time. Default: 0.
- `-b`: Batch size. Set this value according to your GPU memory.

For example, for scale up by 4 without drift correction:
```bash
python main.py -s 4 -L data/frames -S data/dl-444 -b 5
```
for scale up by 8 without drift correction:
```bash
python main.py -s 8 -L data/frames -S data/dl-488 -b 4
```

Note that the code will predict all the frames under the folder you specified by `-L`.
Thus, if you want to predict portion of the frames, please copy them to a new folder and specify the new folder by `-L`.
