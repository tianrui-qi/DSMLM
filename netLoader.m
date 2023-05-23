function net = netLoader()
    % load parameters we will use
    paras           = setParas();
    WhichNet        = paras.WhichNet;       % "unet" or "cnn"
    CheckpointDir   = paras.CheckpointDir;  % dir to store checkpoint
    Checkpoint      = paras.Checkpoint;     % checkpoint file name
    
    % load the checkpoint net or init a new net
    if exist(CheckpointDir, 'dir') == false
        mkdir(CheckpointDir);
        fprintf("netLoader: Creat checkpoint dictionary\n");
        if WhichNet == "cnn", net = cnn(); else, net = unet(); end
        fprintf("netLoader: Init a new " + WhichNet + "\n")
    elseif isstring(Checkpoint) == false
        fprintf("netLoader: Given checkpoint file name is not string\n");
        if WhichNet == "cnn", net = cnn(); else, net = unet(); end
        fprintf("netLoader: Init a new " + WhichNet + "\n")
    elseif any(strcmp({dir(CheckpointDir).name}, Checkpoint))
        load(fullfile(CheckpointDir, Checkpoint), "net");
        fprintf("netLoader: Load checkpoint file success\n");
        net = layerGraph(net);
        fprintf("netLoader: Init the net using checkpoint file\n");
    else
        fprintf("netLoader: File to load the given checkpoint file\n");
        if WhichNet == "cnn", net = cnn(); else, net = unet(); end
        fprintf("netLoader: Init a new " + WhichNet + "\n")
    end
end
