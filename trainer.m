function [] = trainer(paras)
    % prepare train and validation dataset including samples and labels
    sampleGenerator(paras);                     % generating samples/labels
    [trainData, valData] = dataloader(paras);   % load generated samples
    
    % Load a checkpoint net or create a new net
    if exist(paras.CheckpointDir, 'dir') == false
        mkdir(paras.CheckpointDir);
        net = unet;
        fprintf("trainer: Checkpoint dictionary does not exist\n");
    elseif isstring(paras.Checkpoint) == false
        net = unet;
        fprintf("trainer: Choose to not load checkpoint file\n");
    elseif any(strcmp({dir(paras.CheckpointDir).name}, paras.Checkpoint))
        load(fullfile(paras.CheckpointDir, paras.Checkpoint), "net");
        net = layerGraph(net);
        fprintf("trainer: Load checkpoint file success\n");
    else
        net = unet;
        fprintf("trainer: Checkpoint file does not exist\n");
    end
    
    % Options for training
    options = trainingOptions('adam', ...
        'Plots', 'training-progress', ...
        'Verbose', true, ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 2, ...  
        'Shuffle','every-epoch', ...
        'ValidationData', valData, ...
        'ValidationFrequency', 600, ...  % iteration
        'ValidationPatience', 6, ...    
        'InitialLearnRate', 1e-5, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1000, ...
        'LearnRateDropPeriod', 2, ...    % epoch
        'CheckpointPath', paras.CheckpointDir, ...
        'CheckpointFrequency', 600, ...  % iteration
        'CheckpointFrequencyUnit', 'iteration', ...
        'ResetInputNormalization', 0 );
    
    % Start to train
    trainNetwork(trainData, net, options);
end
