import pickle
import torch
from torch import is_tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np



def scale_and_jitter(vertices, interval=(0.9, 1.1), jitter=0.1):
    assert isinstance(interval, tuple)
    
    scale = torch.rand(1,3) * (interval[1] - interval[0]) + interval[0]
    vertices = vertices * scale
    
    jitter = (torch.rand(1,3) * 2 - 1) * jitter
    vertices = vertices + jitter

    vertices = torch.clamp(vertices, min=-0.99, max=0.99)
    
    return vertices

# dataset

class WireframeDataset(Dataset):
    def __init__(
        self,
        dataset_folder = 'data',
        dataset_file_path = '',
        is_train = True,
        transform = scale_and_jitter,
        replica=10,
    ):
        self.dataset_folder = dataset_folder
        self.data_path = dataset_file_path
        self.is_train = is_train
        self.transform = transform
        self.replica = replica

        self.data = self._load_data()

    def _load_data(self):
        data = []
            
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f'load {len(data)} data')
        
        if self.is_train:
            data = data[:int(len(data) * 0.9)]
        else:
            data = data[int(len(data) * 0.9):]
        
        return data
    
    def __len__(self):
        if self.is_train != True:
            return len(self.data)
        else:
            return len(self.data) * self.replica
        
    def __getitem__(self, idx):
        idx = idx % len(self.data)

        sample = self.data[idx]
        
        vertices = sample['vertices']
        
        segments = sample['lines'].astype(np.int32)
        
        vertices = torch.from_numpy(vertices).float()
        segments = torch.from_numpy(segments).long()

        # transform
        
        if self.transform:
            vertices = self.transform(vertices) 

        data = dict()

        data['vertices'] = vertices
        data['segments'] = segments

        return data


# custom collater

def first(it):
    return it[0]

def custom_collate(data, pad_id = -1):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output