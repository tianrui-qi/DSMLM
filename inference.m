%% load network
load(fullfile("checkpoints_cnn_noised", ...
    "net_checkpoint__7000__2023_05_22__19_38_02.mat"), "net");

%% get a sample 512 512 64
filename = "raw\sample\FLFM_stack_00001.tif";
info = imfinfo(filename);  % file info
num_images = numel(info);  % depth
image = imread(filename, 1, 'Info', info);  % first layer
stack = zeros([size(image), num_images], class(image));
stack(:, :, 1) = image;
for k = 2:num_images
    stack(:, :, k) = imread(filename, k, 'Info', info);
end
stack = double(stack);
stack = stack / 255;

%% pad the sample to 544 544 64
pad_size = [16, 16, 0];
sample_pad = padarray(stack, pad_size, 'both');

%% begin to predict 
sample_cut = cell(11, 11);
predict_cut = cell(11, 11);

for i = 1:11  % row idx
    for j = 1:11    % col idx
        % cut the sample to 121 x 64 64 64
        sample_cut{i, j} = sample_pad( ...
            (i-1)*48 + 1 : (i-1)*48 + 64, ... % row up to down
            (j-1)*48 + 1 : (j-1)*48 + 64,  ... % col left to right
            : );
        
        % predict each subsample in sample_cut, 121 x 512 512 256
        output = predict(net, sample_cut{i, j});
        predict_cut{i, j} = output(64:448, 64:448, :);
    end
end
