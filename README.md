# DL-SMLFM
This branch has been achieved and is no longer updated; we implemented the data generation and networks by MATLAB in this branch and have switched all the code to Python.

## [setParas.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/setParas.m) 

This file contains parameters to control all other files (except the train option in [trainer.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/trainer.m)). All other functions will call 
```matlab
paras = setParas;
``` 
at the beginning to load parameters they will use so that we do not need to run this file separately before running other files. Make sure to setup this file properly before run other file.

## [trainer.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/trainer.m)

This is the main file that contain all the pipeline: generate data by [dataGenerator.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGenerator.m), load data by [dataLoader.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataLoader.m), load network by [netLoader.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/netLoader.m), and then start to train. Thus, to train a network, direct run 
```matlab
trainer
```
after setup parameters in [setParas.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/setParas.m) and `trainingOptions` in [trainer.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/trainer.m). Still, each file can run separately and is necessary to run by [trainer.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/trainer.m).

## [dataGenerator.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGenerator.m)

- All the data will store in fold `paras.DataDir`. Since in [dataGeneratorHelper.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGeneratorHelper.m) we implement two kinds of samples (clean samples and noised samples) and two kinds of labels (labels with luminance information for regression and binary labels for classification), we will create four subfolders to store them separately. In each fold, each frame will store as `.mat` file and name by number index, i.e., `1.mat`, `2.mat` so we can match four kinds of data by name. Note that this files saving rule is dependent of [dataLoader.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataLoader.m).

- This file will generate `paras.NumSample` number of data. We will first check the number of existing data by checking the number of files in each subfolder and then generate the number of data we still need.

- Due to limited memory, we will not generate `paras.NumSample` at the same time. Instead, we will keep calling [dataGeneratorHelper.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGeneratorHelper.m) in the while loop until we get enough amount of data where each time we generate `paras.NumFrame` amount of data. 

## [dataLoader.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataLoader.m)

- This is a standard datastore for network training in MATLAB. Please check MATLAB documentation for more detail. 

- In this file, we generate a training datastore and validation datastore. More specifically, we read files from index 1 (`1.mat`) to `paras.NumTrain` as training data and `paras.NumTrain+1` to `paras.NumSample` as validation data. 

- Note that we will not perform dictionary or file checking in this function, so please generate enough data by calling the [dataGenerator](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGenerator.m) function in the previous file before calling this function.

## [netLoader.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/netLoader.m)

This file will first check if we have a valid checkpoint file to load by checking `paras.CheckpointDir` and `paras.Checkpoint`. If not, it will init a new network we choose( [cnn.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/cnn.m), [unet.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/unet.m), or [cnnFocal](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/cnnFocal.m)) that control by `paras.WhichNet`. For architecture of each network, please check each network's file or run the file in MATLAB to add them in workspace first and then use [Deep Network Designer](https://www.mathworks.com/help/deeplearning/gs/get-started-with-deep-network-designer.html) to check them. 
