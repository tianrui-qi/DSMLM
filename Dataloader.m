classdef Dataloader < matlab.io.Datastore
    properties
        paras;
        moleculars;
        CurrentFrameIdx;
    end
    
    methods
        function ds = Dataloader(paras)
            ds.paras = paras;
            generateSampleParas(ds);
            ds.moleculars = generateMoleculars(ds);

            reset(ds)
        end
        
        function data = read(ds)
            sample = reshape(generateSample(ds, ds.CurrentFrameIdx), [64, 64, 64]);
            label = reshape(generateLabelUp(ds, ds.CurrentFrameIdx), [512, 512, 256]);
            data = [];
            data.sample = sample;
            data.label = label;

            ds.CurrentFrameIdx = ds.CurrentFrameIdx + 1;
        end

        function tf = hasdata(ds)
            tf = ds.CurrentFrameIdx <= ds.paras.NumFrame;
        end

        function [] = reset(ds)
            ds.CurrentFrameIdx = 1;
        end

        function p = progress(ds)
            p = ds.CurrentFrameIdx / ds.paras.NumFrame;
        end
        
        %% Set parameters

        function [] = generateSampleParas(ds)    
            % load the basic parameters we will use
            NumMolecule = ds.paras.NumMolecule;
            NumFrame    = ds.paras.NumFrame;
            DimFrame    = ds.paras.DimFrame;
            PixelSize   = ds.paras.PixelSize;
            StdRange    = ds.paras.StdRange;
            LumRange    = ds.paras.LumRange;
            AppearRange = ds.paras.AppearRange;
        
            D = length(DimFrame);  % number of dimensions
        
            % 1. mu set, [D * NumMolecule]
            mu_set = (DimFrame-1)' .* rand([D, NumMolecule]) + 1;
            mu_set = sortrows(mu_set');
            mu_set = mu_set';
        
            % 2. covariance set, [D * D * NumMolecule]
            cov_set = zeros(D, NumMolecule);
            % pixel size correction
            StdRange = StdRange .* (PixelSize(1) ./ PixelSize');  % [D * 2]
            for n = 1:NumMolecule
                cov_set(:, n) = (StdRange(:, 2) - StdRange(:, 1)) .* rand([D, 1]);
                cov_set(:, n) = cov_set(:, n) + StdRange(:, 1);
            end
        
            % 3. luminance set, [NumMolecule]
            lum_set = (LumRange(2) - LumRange(1)) * rand([1, NumMolecule]);
            lum_set = lum_set + LumRange(1);
        
            % 4. mask set, [NumFrame, NumMolecule]
            % determine how many moleculars will appear in each frame
            NumRange = round(NumMolecule * AppearRange);
            num_appear_set = randi(NumRange, [NumFrame, 1]);
            % create a mask stands which moleculars will appear in each frame
            mask_set = zeros(NumFrame, NumMolecule);
            for n = 1:NumFrame
                index = randperm(NumMolecule, num_appear_set(n));
                mask_set(n, index) = 1;  % mask of a molecular is 1 if appear
            end
        
            % save sample parameters in 'paras,' change dtype/rounded meantime
            ds.paras.mu_set    = mu_set;
            ds.paras.cov_set   = cov_set;
            ds.paras.lum_set   = lum_set;
            ds.paras.mask_set  = logical(mask_set);
        end
        
        %% Generate moleculars
        
        function molecular = generateMolecular(ds, m)
            % load parameters we will use
            DimFrame    = ds.paras.DimFrame;
            mu          = ds.paras.mu_set(:, m);
            cov         = ds.paras.cov_set(:, m);
            lum         = ds.paras.lum_set(m);
        
            D = length(DimFrame);  % number of dimensions
        
            % take a slice around the mu where the radia is 5 * std
            radius      = ceil(5 * sqrt(cov));
            lower       = floor(max(mu - radius, ones(D, 1)));
            upper       = ceil(min(mu + radius, DimFrame'));
            diameter    = upper - lower + 1;
        
            % build coordinate system of the slice
            % We use 'ndgrid' instead of 'meshgrid' to match the coordinate system 
            % of the mu/lower/upper above.
            % For example, if diameter is [11 13 8], meshgrid will return three
            % [13 11 8] matrix but ndgrid return three [11 13 8] matrix.
            % 'meshgrid' use Cartesian coordinate system, i.e. 11 is range of
            % x-axis/column, 13 is range of y-axis/row.
            % 'ndgrid' use matrix coordinate system, i.e. 11 is range of y-axis/row
            % and 13 is range of x-axis/column.
            % Above, for mu/lower/upper, we use matrix coordinate system, thus here
            % we use 'ndgrid'
            index = arrayfun(@(l, u) l:u, lower, upper, 'UniformOutput', false);
            grid_cell = cell(1, D);
            [grid_cell{:}] = ndgrid(index{:});
            coord = cat(D+1, grid_cell{:});                 % [y, x, (z), D]
            coord = reshape(coord, [], D);                  % [yx(z), D]
        
            % compute the probability density for each point/pixel in slice
            pdf_values = mvnpdf(coord, mu', diag(cov));           % [yx(z), D]
        
            % set the luminate
            pdf_values = pdf_values * lum / max(pdf_values(:));
        
            % put the slice back to whole frame to get single molecular frame
            molecular = zeros(DimFrame);
            for i = 1:prod(diameter)
                idx = cellfun(@(x) x(i), grid_cell, 'UniformOutput', false);
                molecular(idx{:}) = pdf_values(i);
            end
        end
        
        function moleculars = generateMoleculars(ds)
            % load parameters we will use
            NumMolecule = ds.paras.NumMolecule;
            DimFrame    = ds.paras.DimFrame;
        
            moleculars = zeros([NumMolecule, DimFrame]);
            for m = 1:NumMolecule
                molecular = generateMolecular(ds, m);
                moleculars(m, :) = molecular(:);
            end
        end
        
        %% Generate samples
        
        function sample = generateSample(ds, f)
            % load parameters we will use
            DimFrame    = ds.paras.DimFrame;
            mask_set    = ds.paras.mask_set;
            noise_mu    = ds.paras.noise_mu;
            noise_var   = ds.paras.noise_var;
            
            sample = reshape(mask_set(f, :) * ds.moleculars(:, :), DimFrame);
            sample = imnoise(sample, "gaussian", noise_mu, noise_var);
        end

        %% Generate labels/ground truth
        
        function label_up = generateLabelUp(ds, f)
            % load parameters we will use
            NumMolecule = ds.paras.NumMolecule;
            DimFrame    = ds.paras.DimFrame;
            UpSampling  = ds.paras.UpSampling;
            mu_set      = ds.paras.mu_set;
            lum_set     = ds.paras.lum_set;
            mask_set    = ds.paras.mask_set;
        
            DimFrame_up = UpSampling .* DimFrame;
            mu_set_up   = UpSampling' .* mu_set;
        
            label_up = zeros(DimFrame_up);
            for m = 1:NumMolecule
                mu_up = round(mu_set_up(:, m));
                index = arrayfun(@(x) x, mu_up, 'UniformOutput', false);
                label_up(index{:}) = lum_set(m) * mask_set(f, m);
            end
        end
  
    end
end
