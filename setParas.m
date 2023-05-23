function paras = setParas()
    paras = [];

    % Parameters for netLoader
    paras.CheckpointDir = "checkpoints_unet_noised";
    paras.Checkpoint    = "net_checkpoint_2800.mat";
    
    % Parameters for dataloader and dataGenerator
    paras.Noised        = True;  % generate clean sample or noised sample
    paras.DataDir       = "generated_noised";  % _noised or _clean
    paras.SampleDir     = fullfile(paras.DataDir, "samples");
    paras.LabelDir      = fullfile(paras.DataDir, "labels");
    paras.NumSample     = 6000;
    paras.NumTrain      = 5600;

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
    paras.AppearRange   = [0/16, 5/16];   % min, max % of moleculars/frame
    % parameters for adding noise
    paras.noise_mu      = 0;
    paras.noise_var     = 1/512;
end
