paras = [];
paras = setBasicParas(paras);               % set parameters

sampleGenerator(paras);                     % generating samples and labels
[trainData, valData] = dataloader(paras);   % load generated samples

if exist(paras.CheckpointDir, 'dir') == false
    % no checkpoint dir
    mkdir(paras.CheckpointDir);
    model = unet;
elseif length(dir(paras.CheckpointDir)) == 2
    % have checkpoint dir but empty
    model = unet;
else
    model = load("checkpoint\net_checkpoint__10800__2023_05_19__15_44_04.mat");
end

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 2, ...
    'InitialLearnRate', 1e-5, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1000, ...
    'LearnRateDropPeriod', 2, ...
    'Shuffle','every-epoch', ...
    'ValidationData', valData, ...
    'ValidationFrequency', 600, ...
    'ValidationPatience', 6, ...
    'CheckpointPath', paras.CheckpointDir, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

net = trainNetwork(trainData, model, options);


%% Set parameters

function paras = setBasicParas(paras)
    % Parameters for dataloader and sampleGenerator
    % Path
    paras.SampleDir     = "sampleGenerator\samples";
    paras.LabelDir      = "sampleGenerator\labels";
    paras.CheckpointDir = "checkpoint";
    % sample info
    paras.NumSample     = 4000;
    paras.NumTrain      = 3600;
    
    % Parameters for sampleGeneratorHelper
    % dimensional parameters that need to consider memory
    paras.NumMolecule = 32;             % big affect on running time
    paras.NumFrame    = 100;            % generate NumFrame each time
    paras.DimFrame    = [64, 64, 64];   % row-column-(depth); yx(z)
    paras.UpSampling  = [8, 8, 4];
    % parameters that adjust distribution of sample parameters
    paras.PixelSize   = [65, 65, 100];  % use to correct the covariance
    paras.StdRange    = [0.5, 2];       % adjust covariance of moleculars
    paras.LumRange    = [1/512, 1];
    paras.AppearRange = [1/16, 4/16];   % min, max % of moleculars/frame
    % parameters for adding noise
    paras.noise_mu  = 0;
    paras.noise_var = 1/512;
end