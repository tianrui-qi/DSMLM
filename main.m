%% Set parameters for all files

paras = setParas();

%% generating samples and labels

sampleGenerator(paras);                     

%% load generated samples and labels and split to train and validation data

[trainData, valData] = dataLoader(paras);   

%% load the checkpoint net or init a new net

net = netLoader(paras);

%% train the net

trainer(paras, net, trainData, valData)

%% Help function for set parameters

function paras = setParas()
    paras = [];

    % Parameters for netLoader
    paras.CheckpointDir = "checkpoints";
    paras.Checkpoint    = "net_checkpoint__9000__2023_05_20__04_22_03.mat";
    
    % Parameters for dataloader and sampleGenerator
    paras.SampleDir     = "sampleGenerator\samples";
    paras.LabelDir      = "sampleGenerator\labels";
    paras.NumSample     = 4000;
    paras.NumTrain      = 3600;
    
    % Parameters for sampleGeneratorHelper
    % dimensional parameters that need to consider memory
    paras.NumMolecule   = 32;             % big affect on running time
    paras.NumFrame      = 100;            % generate NumFrame each time
    paras.DimFrame      = [64, 64, 64];   % row-column-(depth); yx(z)
    paras.UpSampling    = [8, 8, 4];
    % parameters that adjust distribution of sample parameters
    paras.PixelSize     = [65, 65, 100];  % use to correct the covariance
    paras.StdRange      = [0.5, 2];       % adjust covariance of moleculars
    paras.LumRange      = [1/512, 1];
    paras.AppearRange   = [1/16, 4/16];   % min, max % of moleculars/frame
    % parameters for adding noise
    paras.noise_mu      = 0;
    paras.noise_var     = 1/512;

    % generate in function sampleGeneratorHelper.m\generateSampleParas
    paras.mu_set        = NaN;
    paras.cov_set       = NaN;
    paras.lum_set       = NaN;
    paras.mask_set      = NaN;
end
