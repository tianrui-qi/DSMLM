function [trainData, valData] = dataLoader()
    % load parameters we will use
    paras       = setParas;
    SampleDir   = paras.SampleDir;
    LabelDir    = paras.LabelDir;
    NumSample   = paras.NumSample;  % total number of datas we want
    NumTrain    = paras.NumTrain;   % number of data splite to train
    
    % reading function
    sampleReader    = @(filename) load(filename).sample;
    labelReader     = @(filename) load(filename).label;

    % training datastore
    % file name, idx from 1 to NumTrain
    sampleFiles = fullfile(SampleDir, ...
        arrayfun(@(n) sprintf('%d.mat', n), ...
        1:NumTrain, 'UniformOutput', false));
    labelFiles  = fullfile(LabelDir, ...
        arrayfun(@(n) sprintf('%d.mat', n), ...
        1:NumTrain, 'UniformOutput', false));
    % build the sample datastore and label datastore
    sampleDS = fileDatastore(sampleFiles, 'ReadFcn', sampleReader);
    labelDS = fileDatastore(labelFiles, 'ReadFcn', labelReader);
    % combine sample and label datastore to form train datastore
    trainData = combine(sampleDS, labelDS);
    
    % validation datastore
    % file name, idx from (NumTrain+1) to NumSample
    sampleFiles = fullfile(SampleDir, ...
        arrayfun(@(n) sprintf('%d.mat', n), ...
        (NumTrain+1):NumSample, 'UniformOutput', false));
    labelFiles  = fullfile(LabelDir, ...
        arrayfun(@(n) sprintf('%d.mat', n), ...
        (NumTrain+1):NumSample, 'UniformOutput', false));
    % build the sample datastore and label datastore
    sampleDS = fileDatastore(sampleFiles, 'ReadFcn', sampleReader);
    labelDS = fileDatastore(labelFiles, 'ReadFcn', labelReader);
    % combine sample and label datastore to form validation datastore
    valData = combine(sampleDS, labelDS);
end
