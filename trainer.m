function [] = trainer()
    % set parameters
    paras = setParas;
    
    % generating samples and labels
    dataGenerator(paras);

    % load generated samples and labels and split to train and validation
    [trainData, valData] = dataLoader(paras);

    % load the checkpoint net or init a new net
    net = netLoader(paras);

    % options for training
    options = trainingOptions('adam', ...
        'Plots', 'training-progress', ...
        'Verbose', true, ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 2, ...  
        'Shuffle','every-epoch', ...
        'ValidationData', valData, ...
        'ValidationFrequency', 600, ...  % iteration
        'ValidationPatience', 6, ...    
        'InitialLearnRate', 1e-5, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1000, ...
        'LearnRateDropPeriod', 2, ...    % epoch
        'CheckpointPath', paras.CheckpointDir, ...
        'CheckpointFrequency', 600, ...  % iteration
        'CheckpointFrequencyUnit', 'iteration', ...
        'ResetInputNormalization', 0 );
    
    % start to train
    trainNetwork(trainData, net, options);
end

function paras = setParas()
    paras = [];

    % Parameters for netLoader
    paras.CheckpointDir = "checkpoints";
    paras.Checkpoint    = "net_checkpoint__9000__2023_05_20__04_22_03.mat";
    
    % Parameters for dataloader and dataGenerator
    paras.SampleDir     = "generated\samples";
    paras.LabelDir      = "generated\labels";
    paras.NumSample     = 4000;
    paras.NumTrain      = 3600;
    
    % Parameters for dataGeneratorHelper
    % dimensional parameters that need to consider memory
    paras.NumMolecule   = 32;             % big affect on running time
    paras.NumFrame      = 100;            % generate NumFrame each time
    paras.DimFrame      = [64, 64, 64];   % row-column-(depth); yx(z)
    paras.UpSampling    = [8, 8, 4];
    % parameters that adjust distribution of sample
    paras.PixelSize     = [65, 65, 100];  % use to correct the covariance
    paras.StdRange      = [0.5, 2];       % adjust covariance of moleculars
    paras.LumRange      = [1/512, 1];
    paras.AppearRange   = [1/16, 4/16];   % min, max % of moleculars/frame
    % parameters for adding noise
    paras.noise_mu      = 0;
    paras.noise_var     = 1/512;

    % generate in function dataGeneratorHelper.m\generateDataParas
    paras.mu_set        = NaN;
    paras.cov_set       = NaN;
    paras.lum_set       = NaN;
    paras.mask_set      = NaN;
end
