classdef Dataloader < matlab.io.Datastore
    properties
        paras;
        molecular;
    end
    
    methods
        function ds = Dataloader(paras)
            ds.paras = paras;
            ds.molecular = generateMolecular(ds);
        end
        
        function [sample_noised, label] = read(ds)
            % generate samples, double normalized
            sample = generateSample(ds);
            % further processing samples, double normalized
            sample_noised = addNoise(ds, sample);
            % generate labels/ground truth, double normalized
            label = generateLabel(ds);
        end

        function tf = hasdata(ds)
            tf = true;
        end

        function [] = reset(ds)
            % Do nothing because there's nothing to reset
        end
        
        %% Generate moleculars
        
        function single_molecular = generate_single_molecular(ds, m)
            % load parameters we will use
            DimFrame    = ds.paras.DimFrame;
            mu          = ds.paras.mu_set(:, m);
            cov         = ds.paras.cov_set(:, :, m);
            lum         = ds.paras.lum_set(m);
        
            D = length(DimFrame);  % number of dimensions
        
            % take a slice around the mu where the radia is 5 * std
            radius      = ceil(5 * sqrt(diag(cov)));
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
            pdf_values = mvnpdf(coord, mu', cov);           % [yx(z), D]
        
            % set the luminate
            pdf_values = pdf_values * lum / max(pdf_values(:));
        
            % put the slice back to whole frame to get single molecular frame
            single_molecular = zeros(DimFrame);
            for i = 1:prod(diameter)
                idx = cellfun(@(x) x(i), grid_cell, 'UniformOutput', false);
                single_molecular(idx{:}) = pdf_values(i);
            end
        end

        function molecular = generateMolecular(ds)
            % load parameters we will use
            NumMolecule = ds.paras.NumMolecule;
            DimFrame    = ds.paras.DimFrame;
        
            molecular = zeros([NumMolecule, DimFrame]);
            for m = 1:NumMolecule
                single_molecular = generate_single_molecular(ds, m);
                molecular(m, :) = single_molecular(:);
            end
        end
        
        %% Generate samples
        
        function sample = generateSample(ds)
            % load parameters we will use
            NumFrame    = ds.paras.NumFrame;
            DimFrame    = ds.paras.DimFrame;
            mask_set    = ds.paras.mask_set;
        
            sample = reshape(mask_set * ds.molecular(:, :), [NumFrame, DimFrame]);
        end
        
        %% Further processing samples
        
        function sample_noised = addNoise(ds, sample)
            % load parameters we will use
            noise_mu    = ds.paras.noise_mu;
            noise_var   = ds.paras.noise_var;
            
            sample_noised = imnoise(sample, "gaussian", noise_mu, noise_var);
        end

        %% Generate labels/ground truth
        
        function label = generateLabel(ds)
            % load parameters we will use
            NumMolecule = ds.paras.NumMolecule;
            NumFrame    = ds.paras.NumFrame;
            DimFrame    = ds.paras.DimFrame;
            UpSampling  = ds.paras.UpSampling;
            mu_set      = ds.paras.mu_set;
            lum_set     = ds.paras.lum_set;
            mask_set    = ds.paras.mask_set;
        
            DimFrame_up = UpSampling .* DimFrame;
            mu_set_up   = UpSampling' .* mu_set;
        
            label = zeros([NumFrame, DimFrame_up]);
            for f = 1:NumFrame
                for m = 1:NumMolecule
                    mu_up = round(mu_set_up(:, m));
                    index = arrayfun(@(x) x, mu_up, 'UniformOutput', false);
                    label(f, index{:}) = lum_set(m) * mask_set(f, m);
                end
            end
        end
    
    end
end
