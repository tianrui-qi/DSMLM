paras = [];
paras = setBasicParas(paras);

current_idx = 1;
for i = 1:1
    [samples_noised, labels_up] = sampleGenerator(paras);
    for f = 1:paras.NumFrame
        sample = reshape(sample_noised(f, :), paras.DimFrame);
        path = "generated/sample_noised/" + current_idx;
        save(path, "sample");
        
        label = reshape(label_up(f, :), paras.DimFrame);
        path = "generated/label_up/" + current_idx;
        save(path, "label");

        current_idx = current_idx + 1;
    end
end

%% Set parameters

function paras = setBasicParas(paras)
    % dimensional parameters that need to consider memory
    paras.NumMolecule = 256;             % big affect on running time
    paras.NumFrame    = 100;
    paras.DimFrame    = [64, 64, 64]; % row-column-(depth); yx(z)
    paras.UpSampling  = [8, 8, 4];

    % parameters that adjust distribution of sample parameters
    paras.PixelSize   = [65, 65, 100];  % use to correct the covariance
    paras.StdRange    = [0.5, 2];       % adjust covariance of moleculars
    paras.LumRange    = [1/512, 1];
    paras.AppearRange = [1/2, 1/1];     % min, max % of moleculars/frame

    % parameters for adding noise
    paras.noise_mu  = 0;
    paras.noise_var = 1/512;
end