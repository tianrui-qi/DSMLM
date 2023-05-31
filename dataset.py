import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        # 初始化函数可以初始化一些参数和变量，例如数据的大小、类型等

    def __getitem__(self, index):
        # 在这个方法中，你可以生成并返回你的数据
        # 你可以使用你的自定义函数来生成数据
        data = your_custom_function_to_generate_data(index)
        return data

    def __len__(self):
        return
