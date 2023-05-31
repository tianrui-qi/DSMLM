# DL-SMLFM
This branch has been achieved and is no longer updated; we implemented the data generation and networks by MATLAB in this branch and have switched all the code to Python.

## File Structure
- `setParas.m`: This file contains parameters to control all other files (except the train option in `trainer.m`). All other functions will call `paras = setParas.m();` at the beginning to load parameters they will use so that we do not need to run this file separately before running other files.  
- `dataGenerator.m`: 
	- All the data will store in fold `paras.DataDir`. Since in `dataGeneratorHelper.m` we implement two kinds of samples (clean samples and noised samples) and two kinds of labels (labels with luminance information for regression and binary labels for classification), we will create four subfolders to store them separately. In each fold, each frame will store as `.mat` file and name by number index, i.e., `1.mat`, `2.mat` so we can match four kinds of data by name. Note that this files saving rule is dependent of `dataLoader.m`.
	- This file will generate `paras.NumSample` number of data set in `setParas.m`. We will first check the number of existing data by checking the number of files in each subfolder and then generate the number of data we still need.
	- Due to limited memory, we will not generate `paras.NumSample` at the same time. Instead, we will keep calling `dataGeneratorHelper.m` in the while loop until we get enough amount of data where each time we generate `paras.NumFrame` amount of data. 
- `dataLoader.m`: 
	- This is a standard datastore for network training in MATLAB. Please check MATLAB documentation for more detail. 
	- In this file, we generate a training datastore and validation datastore. More specifically, we read files from index 1 (`1.mat`) to `paras.NumTrain` as training data and `paras.NumTrain`+1 to `paras.NumSample` as validation data. 
	- Note that we will not perform dictionary/file checking in this function, so please generate enough data by calling the `dataGenerator` function in the previous file before calling this function.
- `netLoader.m`: This file will first check if we have a valid checkpoint file to load by checking `paras.CheckpointDir` and `paras.Checkpoint`. If not, it will init a new network we choose: [cnn.m](https://github.com/tianrui-qi/DL-SMLFM/blob/3329956e9a1f08b2940c3de1007bd1e2c00f9efa/cnn.m), [unet.m](https://github.com/tianrui-qi/DL-SMLFM/blob/048312dc0367812bcfc51545ffdf9a8746cb5e32/cnnFocal.m), or `cnnFocal`.
