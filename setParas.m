function paras = setParas()
    paras = [];

    % Parameters for netLoader
    paras.CheckpointDir = "checkpoints";
    paras.Checkpoint    = false;
    paras.WhichNet      = "unet";   % "cnn" / "unet" / "cnnFocal"
    
    % Parameters for dataLoader and dataGenerator
    paras.DataDir       = "generated";
    paras.NumSample     = 6000;
    paras.NumTrain      = 5600;
    paras.Noised        = 1;  % Noised ? noised sample : clean sample
    paras.Binary        = 0;  % Binary ? classification : regression

    % Parameters for dataGeneratorHelper
    % dimensional parameters that need to consider memory
    paras.NumMolecule   = 32;             % big effect on running time
    paras.NumFrame      = 20;             % generate NumFrame each time
    paras.DimFrame      = [64, 64, 64];   % row-column-(depth); yx(z)
    paras.UpSampling    = [8,  8,  4];
    % parameters that adjust distribution of sample
    paras.PixelSize     = [65, 65, 100];  % use to correct the covariance
    paras.StdRange      = [0.5,   2   ];  % adjust covariance of moleculars
    paras.LumRange      = [1/512, 1   ];
    paras.AppearRange   = [0/16,  5/16];  % min, max % of moleculars/frame
    % parameters for adding noise
    paras.noise_mu      = 0;
    paras.noise_var     = 1/512;
end
