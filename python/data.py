import numpy as np 

from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

from python.utils import get_input, get_labels


class Data(Dataset):
    def __init__(self, data):
        self.data = data
        self.data.index = np.arange(len(self.data))
 
    def __getitem__(self, i): 
        return get_input(i), get_labels(i)

    def __len__(self):
        return len(self.data)
    
     
# merges a list of samples to form a mini-batch
def collate_fn(batch):
    features, labels = zip(*batch)
    
    features = pad_sequence(features, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    
    return features, labels

