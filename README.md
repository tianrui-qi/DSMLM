# DL-SMLFM

This branch has been achieved and is no longer updated; we implemented the data generation and networks by MATLAB in this branch and have switched all the code to Python. Note that this documentation is only for this achieved MATLAB code. We may change the logic of data generation, file saving, and architecture of networks in main or other branches. 

The problem we are facing with this MATLAB pipeline:

- In [cnn.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/cnn.m) and [unet.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/unet.m), gaussian fitting problem is treated as regression problem where they use [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) as loss function. Both network converge at `loss = 0.7` and predict too many pixel compare to label, i.e., 600 instead of 8.
- In [cnnFocal](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/cnnFocal.m), gaussian fitting problem is treated as binary classification problem where it use [Focal Loss]([https://en.wikipedia.org/wiki/Mean_squared_error](https://arxiv.org/abs/1708.02002)) as loss function. Trained network alwayes output a frame that only the last pixel is lighted up.

We give brief documentation for each file to introduce their function, dependent, and other information that may have to know.

## [setParas.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/setParas.m) 

This file contains parameters to control all other files (except the train option in [trainer.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/trainer.m)). All other functions will call 
```matlab
paras = setParas;
``` 
at the beginning to load parameters they will use so that we do not need to run this file separately before running other files. Make sure to set up this file properly before running other files.

## [trainer.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/trainer.m)

This is the main file that contains all the pipeline: generate data by [dataGenerator.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGenerator.m), load data by [dataLoader.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataLoader.m), load network by [netLoader.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/netLoader.m), and then start to train. Thus, to train a network, direct run 
```matlab
trainer
```
after setup parameters in [setParas.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/setParas.m) and `trainingOptions` in [trainer.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/trainer.m). Still, each file can run separately and is necessary to run by [trainer.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/trainer.m).

## [dataGenerator.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGenerator.m)

- This file is mainly use to store all the data we generated as file, and [dataGeneratorHelper.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGeneratorHelper.m) will be used to generate all the samples and labels.

- All the data will store in fold `paras.DataDir`. We will create four subfolders to store them separately since in [dataGeneratorHelper.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGeneratorHelper.m) we implement two kinds of samples (clean samples and noised samples) and two kinds of labels (labels with luminance information for regression and binary labels for classification). In each fold, each frame will store as `.mat` file and name by number index, i.e., `1.mat`, `2.mat` so we can match four kinds of data by name. Note that this files saving rule is dependent of [dataLoader.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataLoader.m).

- This file will generate `paras.NumSample` number of data. We will first check the number of existing data by checking the number of files in each subfolder and then generate the number of data we still need.

- Due to limited memory, we will not generate `paras.NumSample` at the same time. Instead, we will keep calling [dataGeneratorHelper.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGeneratorHelper.m) in the while loop until we get enough amount of data where each time we generate `paras.NumFrame` amount of data. 

## [dataGeneratorHelper.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGeneratorHelper.m)

- This file is the most important file that the define the logic / distribution of the samples and labels we generate. We may change the logic of samples and labels generation in other branches. 

- By running this file, we will generate `paras.NumFrame` number of samples and labels pair. Note that we will first generate `paras.NumMolecular` number of molecular where each follows gaussian distribution and has random mu and variance (not covariance). Then, we generate each sample by random pick part of molecular and sum them up. Thus, although we generate `paras.NumFrame` number of data, they share same molecular set. 

- It's better to set a larger `paras.NumMolecular`, i.e., 256, if you want to generate `paras.NumFrame` number of data that share same molecular list, i.e., simulate the STORM since all the frames in STORM also share same molecular list. However, this may cost lot's of memory, need longer running time, and the samples we generate are not generalize enought for network training since they all share the same molecular list. During training, we may set a small `paras.NumMolecular`, i.e., 16 or 32, and small `paras.NumFrame`, i.e., 20 or 100, and calling this file multiple time. 

## [dataLoader.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataLoader.m)

- This is a standard datastore for network training in MATLAB. Please check MATLAB documentation for more detail. 

- In this file, we generate a training datastore and validation datastore. More specifically, we read files from index 1 (`1.mat`) to `paras.NumTrain` as training data and `paras.NumTrain+1` to `paras.NumSample` as validation data. 

- Note that we will not perform dictionary or file checking in this function, so please generate enough data by calling the [dataGenerator](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/dataGenerator.m) function in the previous file before calling this function.

## [netLoader.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/netLoader.m)

This file will first check if we have a valid checkpoint file to load by checking `paras.CheckpointDir` and `paras.Checkpoint`. If not, it will init a new network we choose( [cnn.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/cnn.m), [unet.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/unet.m), or [cnnFocal](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/cnnFocal.m)) that control by `paras.WhichNet`. For the architecture of each network, please check each network's file or run the file in MATLAB to add them in the workspace first and then use [Deep Network Designer](https://www.mathworks.com/help/deeplearning/gs/get-started-with-deep-network-designer.html) to check them. 

## [inferencer.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/inferencer.m), [gmmFit.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/gmmFit.m)

These two files are not well defined, i.e., they are still temp scripts and have not been written into function. The parameters in these two files need to be set up manually and not controlled by [setParas.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/setParas.m). 

- [inferencer.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/inferencer.m) will cut a `512 512 64` raw sample into 121 number of `64 64 64` samples and predict each of them by network store in the checkpoint. 

- [gmmFit.m](https://github.com/tianrui-qi/DL-SMLFM/blob/matlab-achieve/gmmFit.m) implements a traditional method for Gaussian fitting where we transfer the pixel map to point cloud space first and then fit the mu and covariance of each Gaussian by Gaussian Mixture Model (GMM). This method is not functioning correctly and still has a large error due to the transferring error from pixel map to point cloud space. 
