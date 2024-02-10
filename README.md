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
python main.py [-h] [-s {4,8}] -L FRAMES_LOAD_FOLD [-S DATA_SAVE_FOLD] [-C CKPT_LOAD_PATH] [-T TEMP_SAVE_FOLD][-stride STRIDE] [-window WINDOW] [-method {DCC,MCC,RCC}] -b BATCH_SIZE
```
options:
- `-s {4,8}`: Scale up factor, 4 or 8. Default: 4.
- `-L FRAMES_LOAD_FOLD`: Path to the frames load folder. Note that the code will predict all the frames under this folder. Thus, if you want to predict portion of the frames, please copy them to a new folder and specify this parameter with that new folder.
- `-S DATA_SAVE_FOLD`: Path to the data save folder. No need to specify when stride or window is set as non-zero. Default: None.
- `-C CKPT_LOAD_PATH`: Path to the checkpoint load file without .ckpt. Default: `ckpt/e08/340` or `ckpt/e10/450` when scale up factor is 4 or 8.
- `-T TEMP_SAVE_FOLD`: Path to the temporary save folder for drifting analysis. Must be specified when drift correction will be performed. Default: None.
- `-stride STRIDE`: Step size of the drift corrector, unit frames. Should set with window at the same time. Default: 0.
- `-window WINDOW`: Number of frames in each window, unit frames. Should set with stride at the same time. Default: 0.
- `-method {DCC,MCC,RCC}`: Drift correction method, DCC, MCC, or RCC. DCC run very fast where MCC and RCC is more accurate. We suggest to use DCC to test the window size first and then use MCC or RCC to calculate the final drift. Optional to set when window is set. Default: DCC.
- `-b BATCH_SIZE`: Batch size. Set this value according to your GPU memory.

Note that for all example below, we assign `-b BATCH_SIZE` as 4. You can change it according to your GPU memory and the region you selected to predict.

### Without drift correction

For example, for scale up by 4 (default) or 8 without drift correction:
```bash
python main.py -s 4 -L "data/frames/" -S "data/444-dl/" -b 4
python main.py -s 8 -L "data/frames/" -S "data/488-dl/" -b 4
```

### With drift correction

To perform evaluation with drift correction, we split into two steps since temp results for calculating the drift and final prediction may use different region of the frames, i.e., a small region for getting temp result to reduce the time of calculating the drift and a large region for the final prediction.

#### Step 1: Calculate the drift

First, we predict frames in `-L FRAMES_LOAD_FOLD` as usual like without drift correction where the scale up factor `-s` must set to 4, i.e., the default value. 
However, instead of saving the final prediction results in `-S DATA_SAVE_FOLD`, we stack and save `-stride STRIDE` number of prediction results to `-T TEMP_SAVE_FOLD` as temp result then reset.
For example, if `-stride STRIDE` is 250, temp result `TEMP_SAVE_FOLD/250.tif` will be the stack of frames 1-250 prediction results, `TEMP_SAVE_FOLD/500.tif` will be the stack of frames 251-500 prediction results, and so on. 
As a comparison, when predicting without drift correction, `DATA_SAVE_FOLD/500.tif` will be the stack of frames 1-500 prediction results.
Then, these temp results will be used to calculate the drift, and the final drift of each frames in all dimension will be saved in `TEMP_SAVE_FOLD/drift.csv`. 

We highly rely on cached temp result and drift here: 
please delete `TEMP_SAVE_FOLD/drift.csv` before running if you want to re-calculate the drift for same dataset with new window size or method; 
please delete whole `-T TEMP_SAVE_FOLD` before running if you want to re-calculate the drift for same dataset with new stride size or for a new dataset.

It will be time comsuming to change `-stride STRIDE` since we need to delete whole `-T TEMP_SAVE_FOLD` and re-predict frames. 
A relatively small `-stride STRIDE`, i.e., 250, is recommended to leave the room for test `-window WINDOW` since window size must larger or equal to stride and divisible by stride.
Smaller stride size will be more accurate but time comsuming since we have more windows; big O of DCC is linear to number of windows and MCC and RCC are quadratic to number of windows. 
We suggest to use DCC (default) to test the window size first and then use MCC or RCC to calculate the final drift.

For example, test the window size 1000, 2000, or 3000 with DCC (default) method,
```bash
python main.py -L "data/frames/" -T "data/temp/" -stride 250 -window 1000 -b 4
python main.py -L "data/frames/" -T "data/temp/" -stride 250 -window 2000 -b 4
python main.py -L "data/frames/" -T "data/temp/" -stride 250 -window 3000 -b 4
```
and then use MCC or RCC to calculate the final drift with the best window size, 2000 as a example,
```bash
python main.py -L "data/frames/" -T "data/temp/" -stride 250 -window 2000 -method MCC -b 4
python main.py -L "data/frames/" -T "data/temp/" -stride 250 -window 2000 -method RCC -b 4
```
Remember to delete `TEMP_SAVE_FOLD/drift.csv` every time you change `-window WINDOW` or `-method` in order to re-calculate the drift instead of using the cached drift.

#### Step 2: Perform drift correction

Since we already cached the drift result, we can directly use it and perform drift correction while predicting the frames.
For example, if we set `-T TEMP_SAVE_FOLD` as `data/temp/` in the first step, then we can perform drift correction by
```bash
python main.py -s 4 -L "data/frames/" -S "data/444-dl/" -T "data/temp/" -b 4
python main.py -s 8 -L "data/frames/" -S "data/488-dl/" -T "data/temp/" -b 4
```
