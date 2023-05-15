classdef dataloader < matlab.io.Datastore
    properties
        MiniBatchSize;
        NumObservations;
    end
    
    methods
        function ds = dataloader(numObservations, foldpath)
            load(fullfile(foldpath, "paras.mat"), "paras");
            
            ds.molecular = generate_molecular(foldpath);
            ds.foldpath = foldpath;
  
            ds.MiniBatchSize = paras.NumFrame;
            ds.NumObservations = numObservations;
            ds.reset();
        end
        
        function reset(ds)
            ds.CurrentIndex = 1;
        end
        
        function [data, info] = read(ds)
            numRead = min(ds.MiniBatchSize, ds.NumObservations - ds.CurrentIndex + 1);
            
            % Replace with data generation
            data.Data = add_noise(ds.foldpath, generate_sample(ds.foldpath, molecular));
            data.Response = generate_label(ds.foldpath);
 
            info = struct();
            ds.CurrentIndex = ds.CurrentIndex + numRead;
        end
        
        function tf = hasdata(ds)
            tf = ds.CurrentIndex <= ds.NumObservations;
        end
    end
    
    properties (Access = private)
        CurrentIndex;
    end
end
