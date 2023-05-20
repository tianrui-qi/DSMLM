function [trainData, valData] = dataloader(paras)
    % load parameters we will use
    SampleDir   = paras.SampleDir;
    LabelDir    = paras.LabelDir;
    NumSample   = paras.NumSample;
    NumTrain    = paras.NumTrain;
    
    % reading function
    sampleReader    = @(filename) load(filename).sample;
    labelReader     = @(filename) load(filename).label;

    % training 
    % file name
    sampleFiles = fullfile(SampleDir, arrayfun(@(n) sprintf('%d.mat', n), ...
        1:NumTrain, 'UniformOutput', false));
    labelFiles  = fullfile(LabelDir, arrayfun(@(n) sprintf('%d.mat', n), ...
        1:NumTrain, 'UniformOutput', false));
    % build the sample ds and label ds
    sampleDatastore = fileDatastore(sampleFiles, 'ReadFcn', sampleReader);
    labelDatastore = fileDatastore(labelFiles, 'ReadFcn', labelReader);
    % combine the sample ds and label ds
    trainData = combine(sampleDatastore, labelDatastore);
    
    % validation
    % file name
    sampleFiles = fullfile(SampleDir, arrayfun(@(n) sprintf('%d.mat', n), ...
        (NumTrain+1):NumSample, 'UniformOutput', false));
    labelFiles  = fullfile(LabelDir, arrayfun(@(n) sprintf('%d.mat', n), ...
        (NumTrain+1):NumSample, 'UniformOutput', false));
    % build the sample ds and label ds
    sampleDatastore = fileDatastore(sampleFiles, 'ReadFcn', sampleReader);
    labelDatastore = fileDatastore(labelFiles, 'ReadFcn', labelReader);
    % combine the sample ds and label ds
    valData = combine(sampleDatastore, labelDatastore);
end
