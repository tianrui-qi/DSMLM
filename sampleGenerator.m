function [] = sampleGenerator()
    paras = [];
    paras = setBasicParas(paras);
    
    if exist(mfilename, 'dir'), rmdir(mfilename, 's'); end
    mkdir(mfilename);
    mkdir(fullfile(mfilename, "samples_noised"));
    mkdir(fullfile(mfilename, "labels_up"));
    
    current_idx = 1;
    for i = 1:10
        [samples_noised, labels_up] = sampleGeneratorHelper(paras);
        for f = 1:paras.NumFrame
            shape = size(samples_noised);
            sample = reshape(samples_noised(f, :), shape(2:end));
            path = mfilename + "/samples_noised/" + current_idx;
            save(path, "sample");
            
            shape = size(labels_up);
            label = reshape(labels_up(f, :), shape(2:end));
            path = mfilename + "/labels_up/" + current_idx;
            save(path, "label");
    
            current_idx = current_idx + 1;
        end
    end
end

%% Set parameters

function paras = setBasicParas(paras)
    % dimensional parameters that need to consider memory
    paras.NumMolecule = 16;             % big affect on running time
    paras.NumFrame    = 100;
    paras.DimFrame    = [64, 64, 64]; % row-column-(depth); yx(z)
    paras.UpSampling  = [8, 8, 4];

    % parameters that adjust distribution of sample parameters
    paras.PixelSize   = [65, 65, 100];  % use to correct the covariance
    paras.StdRange    = [0.5, 2];       % adjust covariance of moleculars
    paras.LumRange    = [1/512, 1];
    paras.AppearRange = [1/8, 1/2];     % min, max % of moleculars/frame

    % parameters for adding noise
    paras.noise_mu  = 0;
    paras.noise_var = 1/512;
end