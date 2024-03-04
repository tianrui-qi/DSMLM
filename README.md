# DSMLM

## Environment

The code is tested with `Python == 3.11`, `PyTorch == 2.1`, and `CUDA == 12.1`. 
We recommend you to use 
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or 
[Anaconda](https://www.anaconda.com/) to make sure that all dependencies are in 
place. To create an conda environment:
```bash
# clone the repository
git clone git@github.com:tianrui-qi/DSMLM.git
cd DSMLM
# create the conda environment
conda env create -f environment.yml
conda activate DSMLM
```

## Checkpoints

Please download the checkpoints from 
[iCloud](https://www.icloud.com/iclouddrive/05cFlVujbb2TkrWANiT04tdgQ#340) for 
scale up by 4 or 
[iCloud](https://www.icloud.com/iclouddrive/0e6maAxyFbHaA3MIGSYuivcOw#450) 
for 8. Put the checkpoints under the folder `ckpt/e08/` or `ckpt/e10/`.
Note that `e08` and `e10` match the name in 
[src/cfg/e.py](https://github.com/tianrui-qi/DSMLM/blob/main/src/cfg/e.py) so 
that you can check the training configuration for each checkpoint.

## Evaluation

You can check the parameters that must be specified for evaluation mode by:
```bash
python main.py evalu --help
```
usage:
```bash
python main.py evalu [-h] [-s {4,8}] [-r RNG_SUB_USER [RNG_SUB_USER ...]] -L FRAMES_LOAD_FOLD [-S DATA_SAVE_FOLD] [-C CKPT_LOAD_PATH] [-T TEMP_SAVE_FOLD] [-stride STRIDE] [-window WINDOW] [-m {DCC,MCC,RCC}] -b BATCH_SIZE
```
options:
-   `-s {4,8}`: Scale up factor, 4 or 8. Default: 4.
-   `-r RNG_SUB_USER`: Range of the sub-region of the frames to predict. Due to 
    limited memory, we cut whole frames into patches, i.e., sub-regions and 
    predict them separately. Please type six int separated by space as the 
    subframe start (inclusive) and end (exclusive) index for each dimension, 
    i.e., `-r 0 1 8 12 9 13`. If you not sure about the number of subframe for 
    each dimension you can select, do not specify this parameter; the code will 
    print the range you can select and ask you to type the range. Default: None.
-   `-L FRAMES_LOAD_FOLD`: Path to the frames load folder. Note that the code 
    will predict all the frames under this folder. Thus, if you want to predict 
    portion of the frames, please copy them to a new folder and specify this 
    parameter with that new folder.
-   `-S DATA_SAVE_FOLD`: Path to the data save folder. No need to specify when 
    stride or window is set as non-zero. Default: None.
-   `-C CKPT_LOAD_PATH`: Path to the checkpoint load file without .ckpt. 
    Default: `ckpt/e08/340` or `ckpt/e10/450` when scale up factor is 4 or 8.
-   `-T TEMP_SAVE_FOLD`: Path to the temporary save folder for drifting 
    analysis. Must be specified when drift correction will be performed. 
    Recomend to specify different path for different dataset.
    Default: `os.path.dirname(FRAMES_LOAD_FOLD)/temp/`.
-   `-stride STRIDE`: Step size of the drift corrector, unit frames. Window size
    must larger or equal to stride and divisible by stride. Should set with 
    window at the same time. Default: 0.
-   `-window WINDOW`: Number of frames in each window, unit frames. Window size 
    must larger or equal to stride and divisible by stride. Should set with 
    stride at the same time. Default: 0.
-   `-m {DCC,MCC,RCC}`: Drift correction method, DCC, MCC, or RCC. Must be set 
    if you want to evaluate with drift correction. DCC run very fast where MCC 
    and RCC is more accurate. We suggest to use DCC to test the window size 
    first and then use MCC or RCC to calculate the final drift. Default: None.
-   `-b BATCH_SIZE`: Batch size. Set this value according to your GPU memory. 
    Note that the product of rng_sub_user must divisible by batch_size.

For examples below, we assign `-b BATCH_SIZE` as 4. Change it according to your 
GPU memory and the region `-r RNG_SUB_USER` you selected to predict.

### Without drift correction

For example, for scale up by 4 (default) or 8 without drift correction, if you
not sure about the number of subframe for each dimension you can select, run 
command below and follow the instruction of the code to type the range of 
sub-region you want to predict.
```bash
python main.py evalu -s 4 -L "data/frames/" -S "data/dl-444/" -b 4
python main.py evalu -s 8 -L "data/frames/" -S "data/dl-488/" -b 4
```

If you already know the sub-region you want to predict, for example, patch 
`[0, 1)` in Z, `[8, 12)` in Y, and `[9, 13)` in X, pass the range to 
`-r RNG_SUB_USER` as below.
```bash
python main.py evalu -s 4 -r 0 1 8 12 9 13 -L "data/frames/" -S "data/dl-444/" -b 4
python main.py evalu -s 8 -r 0 1 8 12 9 13 -L "data/frames/" -S "data/dl-488/" -b 4
```

### With drift correction

To perform evaluation with drift correction, we split into two steps since temp 
results for calculating the drift and final prediction may use different region
`-r RNG_SUB_USER` of the frames, i.e., a small region for getting temp result to 
reduce the time of calculating the drift and a large region for the final 
prediction. We skip examples of passing `-r RNG_SUB_USER` to the command below; 
it is the same as without drift correction.

#### Step 1: Calculate the drift

First, we predict frames in `-L FRAMES_LOAD_FOLD` like without drift correction 
where the scale up factor `-s` must set to 4 (default). However, instead of 
saving the final prediction results in `-S DATA_SAVE_FOLD`, we stack and save 
`-stride STRIDE` number of prediction results to `-T TEMP_SAVE_FOLD` as temp 
result then reset. For example, if `-stride STRIDE` is 250, temp result 
`TEMP_SAVE_FOLD/00250.tif` will be the stack of frames 1-250 prediction results,
`TEMP_SAVE_FOLD/00500.tif` will be the stack of frames 251-500 prediction 
results, and so on. As a comparison, when predicting without drift correction, 
`DATA_SAVE_FOLD/00500.tif` will be the stack of frames 1-500 prediction results.
These temp results will be used to calculate the drift, and the final drift of 
each frames in all dimension will be saved in `TEMP_SAVE_FOLD/{DCC,MCC,RCC}.csv`
depend on the method you choose. 

We highly rely on cached temp result and drift value here:
-   Delete `TEMP_SAVE_FOLD/{DCC,MCC,RCC}.csv` if you want to re-calculate the 
    drift for same dataset with new window size; 
-   In addition, for MCC and RCC, delete `TEMP_SAVE_FOLD/r.csv`, temp result 
    shared between MCC and RCC method. If you have run one of MCC or RCC method 
    and want to try the other, you can keep `TEMP_SAVE_FOLD/r.csv` to save time. 
-   Delete whole `-T TEMP_SAVE_FOLD` if you want to re-calculate the drift for 
    same dataset with new stride size.
-   Delete whole `-T TEMP_SAVE_FOLD` or specify a new path (recommend) if you 
    want to re-calculate the drift for different dataset.

Smaller stride size means more window number, leading to more accurate drift
calculation but more time comsuming; big O of DCC is linear to number of windows
and MCC and RCC are quadratic to number of windows. We suggest to use DCC to 
test the window size first and then use MCC or RCC to calculate the final drift.

For example, test the window size 1000, 2000, or 3000 with DCC method
```bash
python main.py evalu -L "data/frames/" -stride 250 -window 1000 -m DCC -b 4
python main.py evalu -L "data/frames/" -stride 250 -window 2000 -m DCC -b 4
python main.py evalu -L "data/frames/" -stride 250 -window 3000 -m DCC -b 4
```
and then use MCC or RCC to calculate the final drift with the best window size, 
2000 as a example,
```bash
python main.py evalu -L "data/frames/" -stride 250 -window 2000 -m MCC -b 4
python main.py evalu -L "data/frames/" -stride 250 -window 2000 -m RCC -b 4
```
Note that we use default `-T TEMP_SAVE_FOLD` here, 
`os.path.dirname(FRAMES_LOAD_FOLD)/temp/`, i.e., `data/temp/`.

#### Step 2: Perform drift correction

Since we already cached the drift result, we can directly use it and perform 
drift correction while predicting the frames. Please make sure that 
`-T TEMP_SAVE_FOLD` and `-m {DCC,MCC,RCC}` match the first step. For 
example, if we use default `-T TEMP_SAVE_FOLD` and set `-m {DCC,MCC,RCC}` 
as RCC in the first step, we can perform drift correction by
```bash
python main.py evalu -s 4 -L "data/frames/" -S "data/dl-444-RCC/" -m RCC -b 4
python main.py evalu -s 8 -L "data/frames/" -S "data/dl-488-RCC/" -m RCC -b 4
```

### Scale Up

We provide the argument `-r RNG_SUB_USER` in purpose; if your whole frames patch 
into `(1, 32, 32)` sub-regions in `(Z, Y, X)` but your GPU memeory can only 
predict 4 sub-regions at a time, you can easily predict the whole frames by 
a loop script. Here is a python example that perform evaluation without drift 
correction, scale up by 8, 4 sub-regions at a time:
```python
import subprocess
for y in range(0, 32):          # 0, 1, 2, ..., 31
    for x in range(0, 32, 4):   # 0, 4, 8, ..., 28
        s =  "-s 8"
        r = f"-r 0 1 {y} {y+1} {x} {x+4}"
        L =  "-L data/frames/"
        S = f"-S data/dl-488-RCC-({y:02d}-{y+1:02d}-{x:02d}-{x+4:02d})/"
        m =  "-m RCC"   # set to None for without drift correction
        b =  "-b 4"
        subprocess.run(
            f"python main.py evalu {s} {r} {L} {S} {m} {b}", 
        check=True, shell=True)
```
Remember to provide unique `-S DATA_SAVE_FOLD` for each loop, like the example 
we show above, to avoid overwriting the results.

Then, you can concatenate results together to get the prediction for whole 
frames. We provide a simple script to do this; please check code cell 4 
"concatenate two 3D subframes into a 3D frame" in 
[utils.py](https://github.com/tianrui-qi/DSMLM/blob/main/util.ipynb) for more
detail.
