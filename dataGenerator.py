import torch
from torch.utils.data import Dataset, DataLoader

class MyDynamicDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, index):
        # 在这里生成你的数据
        # 假设你有一个函数叫做 generate_sample 用于生成数据
        input_data, label = self.generate_sample(index)
        return input_data, label

    def __len__(self):
        return self.num_samples

    def generate_sample(self, index):
        # 在这里添加你的数据生成逻辑
        # 返回一个数据样本和对应的标签
        pass
