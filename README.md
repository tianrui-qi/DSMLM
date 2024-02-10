# SMLFM

## Environment

The code is tested with `Python == 3.11`, `PyTorch == 2.0`, and `CUDA == 11.8`. We recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an conda environment:
```bash
conda env create -f environment.yml
conda activate SMLFM
```

## Checkpoints

Please down load the checkpoints from [iCloud](https://www.icloud.com/iclouddrive/05cFlVujbb2TkrWANiT04tdgQ#340) for scale up by 4 or [iCloud](https://www.icloud.com/iclouddrive/0e6maAxyFbHaA3MIGSYuivcOw#450) for 8. 
Put the checkpoints under the folder `ckpt/e08/` or `ckpt/e10/`.
Note that `e08` and `e10` match the name in [config/e.py](https://github.com/tianrui-qi/SMLFM/blob/main/config/e.py) so that you can check the training configuration for each checkpoint.

## Evaluation

You can check the parameters that must be specified by:
```bash
python main.py --help
```
usage:
```bash
python main.py [-s {4,8}] -L FRAMES_LOAD_FOLD [-S DATA_SAVE_FOLD] [-C CKPT_LOAD_PATH] [-T TEMP_SAVE_FOLD]  [-stride STRIDE] [-window WINDOW] -b BATCH_SIZE
```
options:
- `-s`: Scale up factor, 4 or 8. Default: 4.
- `-L`: Path to the frames load folder. Note that the code will predict all the frames under this folder. Thus, if you want to predict portion of the frames, please copy them to a new folder and specify this parameter with that new folder.
- `-S`: Path to the data save folder. No need to specify when stride or window is set as non-zero. Default: None.
- `-C`: Path to the checkpoint load file without .ckpt. Default: `ckpt/e08/340` or `ckpt/e10/450` when scale up factor is 4 or 8.
- `-T`: Path to the temporary save folder for drifting analysis. Must be specified when drift correction will be performed. Default: None.
- `-stride`: Step size of the drift corrector, unit frames. Should set with window at the same time. Default: 0.
- `-window`: Number of frames in each window, unit frames. Should set with stride at the same time. Default: 0.
- `-b`: Batch size. Set this value according to your GPU memory.

Note that for all example below, we assign the batch size as 4. You can change it according to your GPU memory and the region number you selected to predict.

### Without drift correction

For example, for scale up by 4 or 8 without drift correction:
```bash
python main.py -s 4 -L data/frames/ -S data/dl-444/ -b 4
python main.py -s 8 -L data/frames/ -S data/dl-488/ -b 4
```

### With drift correction

To perform drift correction, we split into two steps. 
First, we stack and save stride number of prediction results to TEMP_SAVE_FOLD `-T` then reset while predicting the frames as temp results. 
These temp results will then be used to calculate the drift, which will be saved in TEMP_SAVE_FOLD `-T` as `drift.csv`. 
Scale up factor must set to 4, i.e., the default value. 
For example, to perform drift correction with stride 250 and window size 2000,
```bash
python main.py -L data/frames/ -T data/temp/ -stride 250 -window 2000 -b 4
```

Then, in second step, since we already cached the drift result, we can directly use it and perform drift correction while predicting the frames.
For example, if we set TEMP_SAVE_FOLD `-T` as `data/temp/` in the first step, then we can perform drift correction by
```bash
python main.py -s 4 -L data/frames/ -S data/dl-444/ -T data/temp/ -b 4
python main.py -s 8 -L data/frames/ -S data/dl-488/ -T data/temp/ -b 4
```

Note that calculating the drift will use temp results. Please delete `drift.csv` in TEMP_SAVE_FOLD `-T` and re-run first step if you want to re-calculate the drift with a new window size for same dataset. However, if you change to a new dataset or change the stride size, you must delete the whole TEMP_SAVE_FOLD `-T` and re-run first step.
