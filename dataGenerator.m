function [] = dataGenerator()
    % load parameters we will use
    paras       = setParas();
    DataDir     = paras.DataDir;
    NumSample   = paras.NumSample;  % total number of datas we want
    
    samples_dir         = fullfile(DataDir, "samples");
    samples_noised_dir  = fullfile(DataDir, "samples_noised");
    labels_dir          = fullfile(DataDir, "labels");
    labels_binary_dir   = fullfile(DataDir, "labels_binary");

    % Get the index of data we want to generate next, 'current_idx,' by
    % check 'SampleDir' and 'LabelDir'
    if exist(DataDir, 'dir') == 0
        mkdir(samples_dir);
        mkdir(samples_noised_dir);
        mkdir(labels_dir);
        mkdir(labels_binary_dir);
        current_idx = 1;
        fprintf("dataGenerator: Create dictionary for data storing\n");
    else
        num_samples         = length(dir(samples_dir))        - 2;
        num_samples_noised  = length(dir(samples_noised_dir)) - 2;
        num_labels          = length(dir(labels_dir))         - 2;
        num_labels_binary   = length(dir(labels_binary_dir))  - 2;
        current_idx = min( ...
            min(num_samples, num_samples_noised), ...
            min(num_labels, num_labels_binary) ...
            ) + 1;
    end
    
    fprintf("dataGenerator: Start; " + ...
       "Existing datas: " + (current_idx-1) + "\n");

    % If we do not have enough datas
    % We may generate more than paras.NumSample since in each while loop,
    % we generate paras.NumFrame number of datas instead of number of
    % datas still need to generate. It's OK since we will just read 
    % paras.NumSample number of data in dataLoader when training.
    while current_idx <= NumSample
        generated = dataGeneratorHelper();
        for f = 1:size(generated.samples, 1)
            % samples and samples noised
            shape  = size(generated.samples); 
            sample = reshape(generated.samples(f, :), shape(2:end));
            save(samples_dir        + '\' + current_idx, "sample");
            sample = reshape(generated.samples_noised(f, :), shape(2:end));
            save(samples_noised_dir + '\' + current_idx, "sample");
            % labels and labels binary
            shape  = size(generated.labels);
            label  = reshape(generated.labels(f, :), shape(2:end));
            save(labels_dir        + '\' + current_idx, "label");
            label  = reshape(generated.labels_binary(f, :), shape(2:end));
            save(labels_binary_dir + '\' + current_idx, "label");
            % update the index of next data
            current_idx = current_idx + 1;
        end
        fprintf("dataGenerator: In progress; " + ...
            "Existing datas: " + (current_idx-1) + "\n");
    end

    fprintf("dataGenerator: Finish; " + ...
        "Existing datas: " + (current_idx-1) + "\n")
end
