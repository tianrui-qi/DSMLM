function net = netLoader()
    % load parameters we will use
    paras           = setParas;
    CheckpointDir   = paras.CheckpointDir;  % dir to store checkpoint
    Checkpoint      = paras.Checkpoint;     % checkpoint file name
    
    % load the checkpoint net or init a new net
    if exist(CheckpointDir, 'dir') == false
        mkdir(CheckpointDir);
        fprintf("netLoader: Creat checkpoint dictionary\n");
        net = unet;
        fprintf("netLoader: Init a new net\n")
    elseif isstring(Checkpoint) == false
        fprintf("netLoader: Given checkpoint file name is not string\n");
        net = unet;
        fprintf("netLoader: Init a new net\n")
    elseif any(strcmp({dir(CheckpointDir).name}, Checkpoint))
        load(fullfile(CheckpointDir, Checkpoint), "net");
        fprintf("netLoader: Load checkpoint file success\n");
        net = layerGraph(net);
        fprintf("netLoader: Init the net using checkpoint file\n");
    else
        fprintf("netLoader: File to load the given checkpoint file\n");
        net = unet;
        fprintf("netLoader: Init a new net\n")
    end
end

function layers = unet()
    layers = [
        imageInputLayer([64 64 64],"Name","imageinput")
        resize3dLayer("Name","resize3d-scale","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[8 8 4])
        convolution2dLayer([3 3],256,"Name","conv","Padding","same")
        batchNormalizationLayer("Name","batchnorm")
        reluLayer("Name","relu")
        maxPooling2dLayer([5 5],"Name","maxpool","Padding","same","Stride",[4 4])
        convolution2dLayer([3 3],512,"Name","conv_1","Padding","same")
        batchNormalizationLayer("Name","batchnorm_1")
        reluLayer("Name","relu_1")
        maxPooling2dLayer([5 5],"Name","maxpool_1","Padding","same","Stride",[4 4])
        convolution2dLayer([3 3],1024,"Name","conv_2","Padding","same")
        batchNormalizationLayer("Name","batchnorm_2")
        reluLayer("Name","relu_2")
        resize2dLayer("Name","resize-scale","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[4 4])
        convolution2dLayer([3 3],1024,"Name","conv_3","Padding","same")
        batchNormalizationLayer("Name","batchnorm_3")
        reluLayer("Name","relu_3")
        resize2dLayer("Name","resize-scale_1","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[4 4])
        convolution2dLayer([3 3],512,"Name","conv_4","Padding","same")
        batchNormalizationLayer("Name","batchnorm_4")
        reluLayer("Name","relu_4")
        convolution2dLayer([3 3],256,"Name","conv_5","Padding","same")
        reluLayer("Name","relu_5")
        regressionLayer("Name","regressionoutput")];
end
