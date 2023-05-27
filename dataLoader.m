function [trainData, valData] = dataLoader()
    % load parameters we will use
    paras       = setParas();
    DataDir     = paras.DataDir;
    NumSample   = paras.NumSample;  % total number of datas we want
    NumTrain    = paras.NumTrain;   % number of data splite to train
    Noised      = paras.Noised;     % Noised ? noised sample : clean sample
    Binary      = paras.Binary;     % Binary ? classification : regression
    
    % dictionary to samples and labels
    % note that in dataGenerator.m, we use different variable name for
    % different data, i.e., we use "samples_dir" for clean samples 
    % dictionary and "samples_noised_dir" for noised data dictionary. But
    % here we use same name "samples_dir" since we just use one of it.
    % Same rule apply to labels as well.
    if Noised
        samples_dir = fullfile(DataDir, "samples");
    else
        samples_dir = fullfile(DataDir, "samples_noised");
    end
    if Binary
        labels_dir = fullfile(DataDir, "labels");
    else
        labels_dir = fullfile(DataDir, "labels_binary");
    end

    % reading function
    % in dataGenerator.m, we name each sample in samples and samples_noised 
    % as "sample" and each label in labels and labels_bianry as "label". 
    sampleReader = @(filename) load(filename).sample;
    labelReader  = @(filename) load(filename).label;

    % training datastore
    % file name, idx from 1 to NumTrain
    sampleFiles = fullfile(samples_dir, ...
        arrayfun(@(n) sprintf('%d.mat', n), ...
        1:NumTrain, 'UniformOutput', false));
    labelFiles  = fullfile(labels_dir, ...
        arrayfun(@(n) sprintf('%d.mat', n), ...
        1:NumTrain, 'UniformOutput', false));
    % build the sample datastore and label datastore
    sampleDS = fileDatastore(sampleFiles, 'ReadFcn', sampleReader);
    labelDS = fileDatastore(labelFiles, 'ReadFcn', labelReader);
    % combine sample and label datastore to form train datastore
    trainData = combine(sampleDS, labelDS);
    
    % validation datastore
    % file name, idx from (NumTrain+1) to NumSample
    sampleFiles = fullfile(samples_dir, ...
        arrayfun(@(n) sprintf('%d.mat', n), ...
        (NumTrain+1):NumSample, 'UniformOutput', false));
    labelFiles  = fullfile(labels_dir, ...
        arrayfun(@(n) sprintf('%d.mat', n), ...
        (NumTrain+1):NumSample, 'UniformOutput', false));
    % build the sample datastore and label datastore
    sampleDS = fileDatastore(sampleFiles, 'ReadFcn', sampleReader);
    labelDS = fileDatastore(labelFiles, 'ReadFcn', labelReader);
    % combine sample and label datastore to form validation datastore
    valData = combine(sampleDS, labelDS);
end
