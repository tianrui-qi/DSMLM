# DL-SMLFM
This branch has been achieve and no longer update; we implement the data generation and networks by MATLAB in this branch and have switch all the code to python.

## File Structure
- `setParas.m`: This file contains parameters to control all other files (except train option in `trainer.m`). All other functions will call `paras = setParas.m();` at beginning to load parameters they will use so that we do not need to run this file seperatly before run other files.  
- `dataGenerator.m`: 
  - All the datas will store in fold `paras.DataDir`. Since in `dataGeneratorHelper.m` we implement two kinds of samples (clean samples and noised samples) and two kinds of labels (labels with luminance information for regression and binary labels for classification), we will create four subfolds to store them seperatly. In each fold, each frame will store as `.mat` file and name by number index, i.e., `1.mat`, `2.mat` so we can match four kinds of data by name. Note that this files saving rule is dependent of `dataLoader.m`.
  - This file will generate `paras.NumSample` number of datas set in `setParas.m`. We will first check number of existing datas by checking number of files in each subfold and then generate number of datas we still need.
  - Due to limit amount of memory, we will not generate `paras.NumSample` at the same time. Instead, we will keep calling `dataGeneratorHelper.m` in while loop untill we get enough amount of datas where each time we generate `paras.NumFrame` amount of data. 
- `dataLoader.m`: 
  - This is a standard datastore for network training in MATLAB. Please check MATLAB documentation for more detail. 
  - In this file, we generate training datastore and validation datastore. More specificly, we read files from index 1 (`1.mat`) to `paras.NumTrain` as training datas and `paras.NumTrain`+1 to `paras.NumSample` as validation datas. 
  - Note that we will not perform dictionary/file checking in this function, so please generate enough number of datas by calling `dataGenerator` function in previous file before call this function.
- `netLoader.m`: This file will first check if we have valid checkpoint file to load by checking `paras.CheckpointDir` and `paras.Checkpoint`. If not, it will init a new network we choose: `cnn.m`, `unet.m`, or `cnnFocal`.
