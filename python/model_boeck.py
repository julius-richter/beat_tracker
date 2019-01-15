
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

import IPython.display as ipd
from scipy.io import wavfile

import gc


# ## Global settings

# In[2]:


print('Torch version: {}'.format(torch.__version__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: %s' % (device))


# ## Load data

# In[3]:


features = pickle.load(open('../data/pickle/ballroom_features_boeck.npy', 'rb'))

labels = pickle.load(open('../data/pickle/ballroom_labels_boeck.npy', 'rb'))


# ## Create 'Dataset' and 'DataLoader' 

# In[4]:


class Data(Dataset):
    def __init__(self):
        pass
        
    def __getitem__(self, index): 
        return features[index], labels[index]
        
    def __len__(self):
        return len(features)
    
# merges a list of samples to form a mini-batch
def collate_fn(batch):
    features, labels = zip(*batch)
    
    features = pad_sequence(features, batch_first=True)
    
    labels = pad_sequence(labels, batch_first=True)
    
    return features, labels


# In[5]:


dataset = Data()

len_dataset = len(dataset)
nr_folds = 8 

len_test = int((1 / nr_folds) * len_dataset)
len_train = len_dataset - len_test

# Random split
np.random.seed(0)
perm = np.random.permutation(len_dataset)


# ## Select Fold

# In[6]:


fold = 0

indices_test = perm[len_test*fold:len_test*(fold+1)]
indices_train = np.delete(perm, np.arange(len_test*fold,len_test*(fold+1)))


# In[7]:


batch_size = 10 

trainset = Subset(dataset=dataset, indices=indices_train)
testset = Subset(dataset=dataset, indices=indices_test)

trainloader = DataLoader(dataset=trainset, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)
testloader = DataLoader(dataset=testset, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)


# ## Model

# In[8]:


class ModelBoeck(nn.Module):
    def __init__(self):
        super(ModelBoeck, self).__init__()
        
        # Model parameters
        self.input_size = 120
        self.output_size = 2
        self.num_layers = 3
        self.hidden_size = 25     
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        
        # Recurrent layer 
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, 
                            bidirectional=self.bidirectional,
                            batch_first=True)
        
        # Read out layer
        self.fc = nn.Linear(self.num_directions * self.hidden_size, self.output_size)       
        
    def forward(self, x): 
        
#         packed = pack_padded_sequence(x, lengths, batch_first=True)
        
        lstm_out, _ = self.lstm(x) 
        
#         lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        fc_out = self.fc(lstm_out)
        
        y = F.log_softmax(fc_out, dim=2)
        
        return torch.transpose(y, 1, 2)


# In[9]:


model = ModelBoeck().to(device)

loss_vec = []


# In[10]:


loss_function = nn.NLLLoss(weight=torch.tensor([1., 70.]).to(device))

optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)


# ## Load trained parameters

# In[11]:


# model.load_state_dict(torch.load('../models/model_boeck.pt'))


# ## Train the model 

# In[12]:


for epoch in range(2):  
    for i, (feature, label) in enumerate(trainloader):
        
        # Clear out accumulates gradients 
        model.zero_grad()

        # Forward pass
        out = model(feature.to(device))

        # Backward propagation
        loss = loss_function(out, label.to(device))
        loss_vec.append(loss)

        # Calculate gradients
        loss.backward()

        # Optimization 
        optimizer.step()

        print('Epoch: {:2d}   Batch: {:2d} of {:d}   Loss: {:.3e}'
              .format(epoch+1, i+1, len(trainloader), loss.item()))
        del feature, label, out
    gc.collect()





