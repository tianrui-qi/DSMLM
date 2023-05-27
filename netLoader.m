function net = netLoader()
    % load parameters we will use
    paras           = setParas();
    CheckpointDir   = paras.CheckpointDir;  % dir to store checkpoint
    Checkpoint      = paras.Checkpoint;     % checkpoint file name
    WhichNet        = paras.WhichNet;       % "cnn" / "unet" / "cnnFocal"
    
    % choose the network if no checkpoint
    if WhichNet == "cnn",       net = cnn();        end
    if WhichNet == "unet",      net = unet();       end
    if WhichNet == "cnnFocal",  net = cnnFocal();  end

    % load the checkpoint net or init a new net
    if exist(CheckpointDir, 'dir') == false
        mkdir(CheckpointDir);
        fprintf("netLoader: Creat checkpoint dictionary\n");
        fprintf("netLoader: Init a new " + WhichNet + "\n")
    elseif isstring(Checkpoint) == false
        fprintf("netLoader: Given checkpoint file name is not string\n");
        fprintf("netLoader: Init a new " + WhichNet + "\n")
    elseif any(strcmp({dir(CheckpointDir).name}, Checkpoint))
        load(fullfile(CheckpointDir, Checkpoint), "net");
        fprintf("netLoader: Load checkpoint file success\n");
        net = layerGraph(net);
        fprintf("netLoader: Init the net using checkpoint file\n");
    else
        fprintf("netLoader: File to load the given checkpoint file\n");
        fprintf("netLoader: Init a new " + WhichNet + "\n")
    end
end
