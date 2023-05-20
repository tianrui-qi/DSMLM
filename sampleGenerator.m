function [] = sampleGenerator(paras)
    % load parameters we will use
    SampleDir  = paras.SampleDir;
    LabelDir   = paras.LabelDir;
    NumSample  = paras.NumSample; % total # samples we want
    
    % Index of sample we want to generate next
    if exist(SampleDir, 'dir') == 0 || exist(LabelDir, 'dir') == 0
        % if mfilename not exist, means we do not have any sample
        mkdir(SampleDir);
        mkdir(LabelDir);
        current_idx = 1;
        fprintf("sampleGenerator: Create dictionary SampleDir & LabelDir");
    else
        % if exist, get the number of sample we have 
        num_sample  = length(dir(SampleDir)) - 2;
        num_label   = length(dir(LabelDir))  - 2;
        current_idx = min(num_sample, num_label) + 1;
    end
    
    fprintf("sampleGenerator: Start; " + ...
        "Existing sample/label pairs: " + current_idx + "\n");
    
    % If we do not have enough sample, generate samples
    % We may generate more than paras.NumSample since for in while loop,
    % we generate paras.NumFrame number of frames instead of number of
    % samples still need to generate. It's OK since we will just read 
    % paras.NumSample number of files in dataloader when training.
    while current_idx <= NumSample
        generated       = sampleGeneratorHelper(paras);
        samples_noised  = generated.samples_noised;
        labels_up       = generated.labels_up;
        for f = 1:size(samples_noised, 1)
            shape   = size(samples_noised);
            sample  = reshape(samples_noised(f, :), shape(2:end));
            save(fullfile(SampleDir, current_idx), "sample");
            
            shape   = size(labels_up);
            label   = reshape(labels_up(f, :), shape(2:end));
            save(fullfile(LabelDir, current_idx), "label");
    
            current_idx = current_idx + 1;
        end
    end

    fprintf("sampleGenerator: Finish; " + ...
        "Existing sample/label pairs: " + current_idx + "\n")
end
