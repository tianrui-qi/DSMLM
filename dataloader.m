function [trainData, valData] = dataloader()
    %% file path to sample and label
    sampleDir = 'C:\Users\tianrui\OneDrive - Georgia Institute of Technology\Research\Jia-Lab\DL-SMLFM\sampleGenerator\samples_noised';
    labelDir = 'C:\Users\tianrui\OneDrive - Georgia Institute of Technology\Research\Jia-Lab\DL-SMLFM\sampleGenerator\labels_up';
    
    %% training data
    % file name
    sampleFiles = fullfile(sampleDir, arrayfun(@(n) sprintf('%d.mat', n), 1:2100, 'UniformOutput', false));
    labelFiles  = fullfile(labelDir, arrayfun(@(n) sprintf('%d.mat', n), 1:2100, 'UniformOutput', false));
    % reading function
    sampleReader = @(filename) load(filename).sample;
    labelReader = @(filename) load(filename).label;
    % build the sample ds and label ds
    sampleDatastore = fileDatastore(sampleFiles, 'ReadFcn', sampleReader);
    labelDatastore = fileDatastore(labelFiles, 'ReadFcn', labelReader);
    % combine the sample ds and label ds
    trainData = combine(sampleDatastore, labelDatastore);
    
    %% validation data
    % file name
    sampleFiles = fullfile(sampleDir, arrayfun(@(n) sprintf('%d.mat', n), 2101:3000, 'UniformOutput', false));
    labelFiles  = fullfile(labelDir, arrayfun(@(n) sprintf('%d.mat', n), 2101:3000, 'UniformOutput', false));
    % reading function
    sampleReader = @(filename) load(filename).sample;
    labelReader = @(filename) load(filename).label;
    % build the sample ds and label ds
    sampleDatastore = fileDatastore(sampleFiles, 'ReadFcn', sampleReader);
    labelDatastore = fileDatastore(labelFiles, 'ReadFcn', labelReader);
    % combine the sample ds and label ds
    valData = combine(sampleDatastore, labelDatastore);
end
