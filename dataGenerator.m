function [] = dataGenerator()
    % load parameters we will use
    paras       = setParas;
    Noised      = paras.Noised;
    SampleDir   = paras.SampleDir;
    LabelDir    = paras.LabelDir;
    NumSample   = paras.NumSample;   % total number of datas we want

    % Get the index of data we want to generate next, 'current_idx,' by
    % check 'SampleDir' and 'LabelDir'
    if exist(SampleDir, 'dir') == 0 || exist(LabelDir, 'dir') == 0
        % dir do not exist, we do not have any data
        mkdir(SampleDir);
        mkdir(LabelDir);
        current_idx = 1;
        fprintf("dataGenerator: Create dictionary SampleDir & LabelDir\n");
    else
        % if exist, get the number of samples and labels we have
        num_sample  = length(dir(SampleDir)) - 2;
        num_label   = length(dir(LabelDir))  - 2;
        current_idx = min(num_sample, num_label) + 1;
    end
    
    fprintf("dataGenerator: Start; " + ...
        "Existing datas: " + (current_idx-1) + "\n");
    
    % If we do not have enough datas
    % We may generate more than paras.NumSample since in each while loop,
    % we generate paras.NumFrame number of datas instead of number of
    % datas still need to generate. It's OK since we will just read 
    % paras.NumSample number of data in dataLoader when training.
    while current_idx <= NumSample
        generated = dataGeneratorHelper(paras);
        if Noised, samples = generated.samples_noised; end
        if ~Noised, samples = generated.samples; end
        labels = generated.labels_up;
        for f = 1:size(samples, 1)
            % samples
            shape   = size(samples);
            sample  = reshape(samples(f, :), shape(2:end));
            save(SampleDir + '\' + current_idx, "sample");
            % labels
            shape   = size(labels);
            label   = reshape(labels(f, :), shape(2:end));
            save(LabelDir + '\' + current_idx, "label");
            % update the index of next data
            current_idx = current_idx + 1;
        end
    end

    fprintf("dataGenerator: Finish; " + ...
        "Existing datas: " + (current_idx-1) + "\n")
end
