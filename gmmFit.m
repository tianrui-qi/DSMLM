%%
[paras, generated] = sampleGenerator;
s_sample = reshape(generated.sample(1, :), paras.DimFrame);

% ground truth, low resolution and high resolution
truMuHi = paras.mu_set .* paras.mask_set(1, :);
truMuHi = truMuHi(:, any(truMuHi, 1));  % remve [0; 0] coloum
truMuHi = sortrows(truMuHi');
truMuHi = truMuHi';
truMuLo = round(truMuHi);

%% Cut the moleculars in frame according to low resolution mu
tic
cut_radius = 4 * 2;
[preMuLo, molecular_cut] = cutMolecular(s_sample, cut_radius);
toc
saveFrames("predict", "cut", molecular_cut)

%%
tic
preMuHi = zeros(size(preMuLo));
for m = 1:length(preMuLo)
    shape = size(molecular_cut);
    molecular = reshape(molecular_cut(m, :), shape(2:end));
    molecular = 2^13 * molecular;
    pc = frame2pc(molecular);
    
    AIC = zeros(1,3);
    GMModels = cell(1,3);
    for k = 1:3
        GMModels{k} = fitgmdist(pc, k, 'Regularize', 1e-10);
        AIC(k)= GMModels{k}.AIC;
    end
    [minAIC,numComponents] = min(AIC);
    BestModel_mu = GMModels{numComponents}.mu;  % [NumComponents, D]
    BestModel_mu = BestModel_mu - (cut_radius+1);

    distances = sqrt(sum(BestModel_mu .^ 2, 2));
    [~, idx] = min(distances);

    preMuHi(:, m) = BestModel_mu(idx, :)' + preMuLo(:, m);
end
toc

%%
diff = abs(preMuHi - truMuHi);
mu = mean(diff(:));
variance = var(diff(:));
disp(['Mean: ', num2str(mu)])
disp(['Variance: ', num2str(variance)])

list_error = [];
for m = 1:length(preMuHi)
    if diff(1, m) > 0.125 || diff(2, m) > 0.125 || diff(3, m) > 0.25
        list_error = [list_error, m];
    end
end
length(list_error)

%% Help function

function pc = im2pc(im)
    pc = [];
    [x, y] = size(im);
    for i = 1:x
        for j = 1:y
            number = im(i, j);
            for n = 1:number
                pc = [pc; [i, j]];
            end
        end
    end
end

function pc = frame2pc(im)
    pc = [];
    [x, y, z] = size(im);
    for i = 1:x
        for j = 1:y
            for k = 1:z
                number = im(i, j, k);
                for n = 1:number
                    pc = [pc; [i, j, k]];
                end
            end   
        end
    end
end

function [mu_low, molecular_cut] = cutMolecular(frame, cut_radius)
    D = ndims(frame);       % number of dimension
    DimFrame = size(frame); % dimension of frame
    
    % detect all the pixel that are mu (center of molecular)
    BW = imregionalmax(frame);                    % DimFrame, bool
    
    % extract coordinate of all '1' in BW as mu with low resolution
    linear_idx = find(BW == 1);
    sub_idx = cell(1, D);
    [sub_idx{:}] = ind2sub(DimFrame, linear_idx); % D * [Nummolecular * 1]
    mu_low = sortrows([sub_idx{:}]);              % [NumMolecular * D]
    mu_low = mu_low';                             % [D * NumMolecular]
    
    % we pad the frame with cut_radius
    % prevent cutting out of boundary of frame
    frame_pad = padarray(frame, ones([1, D]) * cut_radius);
    mu_low_pad = mu_low + (ones([D, 1]) * cut_radius);
    
    NumMolecular = length(mu_low_pad);
    cutDimFrame = ones([1, D]) * (2 * cut_radius + 1);

    molecular_cut = zeros([NumMolecular, cutDimFrame]);
    for m = 1:NumMolecular
        lower = mu_low_pad(:, m) - cut_radius;
        upper = mu_low_pad(:, m) + cut_radius;
        index = arrayfun(@(l, u) l:u, lower, upper, 'UniformOutput', false);
        molecular_cut(m, :) = reshape(frame_pad(index{:}), [], 1);
    end
end

function [] = saveTif(path, frame)
    % input frame is normalized float
    frame = uint8(round(frame*255));
    DimFrame = size(frame);
    if length(DimFrame) == 2
        imwrite(frame, path+".tif", ...
            'WriteMode', 'overwrite',  'Compression','none');
    end
    if length(DimFrame) == 3
        imwrite(frame(:, :, 1), path+".tif", ...
            'WriteMode', 'overwrite',  'Compression','none');
        for d = 2:DimFrame(3)
            imwrite(frame(:, :, d), path+".tif", ...
                'WriteMode', 'append',  'Compression','none');
        end
    end
end

function [] = saveFrames(fold, subfold, frames)
    % .tif is for illustration purposes only, the files saved are not
    % dependence of other function
    shape = size(frames);
    NumFrame = shape(1);
    DimFrame = shape(2:end);

    % creat the subfold to store tif
    mkdir(fullfile(fold, subfold));
    for f = 1:NumFrame
        filename = f + "_" + subfold + "_" + fold;
        path = fullfile(fold, subfold, filename);
        saveTif(path, reshape(frames(f, :), DimFrame));
    end
end