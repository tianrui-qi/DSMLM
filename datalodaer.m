% 文件路径
sampleDir = 'C:\Users\tianrui\OneDrive - Georgia Institute of Technology\Research\Jia-Lab\DL-SMLFM\sampleGenerator\samples_noised';
labelDir = 'C:\Users\tianrui\OneDrive - Georgia Institute of Technology\Research\Jia-Lab\DL-SMLFM\sampleGenerator\labels_up';

% 生成文件名
sampleFiles = fullfile(sampleDir, arrayfun(@(n) sprintf('%d.mat', n), 1:1000, 'UniformOutput', false));
labelFiles  = fullfile(labelDir, arrayfun(@(n) sprintf('%d.mat', n), 1:1000, 'UniformOutput', false));

% 定义自定义读取函数
sampleReader = @(filename) load(filename).sample;
labelReader = @(filename) load(filename).label;

% build the sample ds and label ds
sampleDatastore = fileDatastore(sampleFiles, 'ReadFcn', sampleReader);
labelDatastore = fileDatastore(labelFiles, 'ReadFcn', labelReader);

% combine the sample ds and label ds
ds = combine(sampleDatastore, labelDatastore);
