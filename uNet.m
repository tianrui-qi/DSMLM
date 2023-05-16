function lgraph = uNet()
lgraph = layerGraph();

% Add Layer Branches

tempLayers = imageInputLayer([128 128 64],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    convolution2dLayer([3 3],128,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([5 5],"Name","maxpool","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    convolution2dLayer([3 3],256,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([5 5],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")
    convolution2dLayer([3 3],512,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    transposedConv2dLayer([3 3],256,"Name","transposed-conv","Cropping","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat")
    convolution2dLayer([3 3],512,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_6")
    convolution2dLayer([3 3],256,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_7")
    transposedConv2dLayer([3 3],128,"Name","transposed-conv_1","Cropping","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_1")
    convolution2dLayer([3 3],256,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_8")
    convolution2dLayer([3 3],128,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_9")
    transposedConv2dLayer([3 3],256,"Name","transposed-conv_3","Cropping","same","Stride",[8 8])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = transposedConv2dLayer([3 3],256,"Name","transposed-conv_2","Cropping","same","Stride",[8 8]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_2")
    convolution2dLayer([3 3],512,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_10")
    convolution2dLayer([3 3],256,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_11")
    convolution2dLayer([1 1],256,"Name","conv_11_1","Padding","same")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

% Connect Layer Branches

lgraph = connectLayers(lgraph,"imageinput","conv");
lgraph = connectLayers(lgraph,"imageinput","transposed-conv_2");
lgraph = connectLayers(lgraph,"relu_1","maxpool");
lgraph = connectLayers(lgraph,"relu_1","depthcat_1/in2");
lgraph = connectLayers(lgraph,"relu_3","maxpool_1");
lgraph = connectLayers(lgraph,"relu_3","depthcat/in1");
lgraph = connectLayers(lgraph,"transposed-conv","depthcat/in2");
lgraph = connectLayers(lgraph,"transposed-conv_1","depthcat_1/in1");
lgraph = connectLayers(lgraph,"transposed-conv_3","depthcat_2/in1");
lgraph = connectLayers(lgraph,"transposed-conv_2","depthcat_2/in2");

% plot(lgraph);
end
