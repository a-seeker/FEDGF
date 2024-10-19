import numpy as np
from torch.utils.data import DataLoader, Dataset
class MyDataset(Dataset):
    def __init__(self, dataset, num_per_generator):
        self.dataset = dataset
        self.num = num_per_generator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], int(index/self.num)

def saveDataset(aug_data, file):
    np.savez_compressed(file, data = aug_data)

def loadDataset(file): 
    data = np.load(file)
    aug_data = data['data']
    return aug_data