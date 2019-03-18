import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import pickle

class wikihopDataset(Dataset):
    def __init__(self, dataJsonPath):
        with open(dataJsonPath, 'r') as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


# input: batch (a list of datasample-maps)
# output: zero-pad supports to equal length, put others in list
def wikihopBatchCollate(batch):
    batched = {}
    unknownWordId = 400000

    # zero-pad supports
    batch_support_ids = []
    maxSupportLen = max([len(item['supports']) for item in batch])
    for item in batch:
        lenDiff = maxSupportLen - len(item['supports'])
        padded_ids = item['supports'] + [unknownWordId for _ in range(lenDiff)]
        batch_support_ids.append(padded_ids)
    batched['supports'] = torch.tensor(batch_support_ids)

    # concat other features in list across batch
    def concatFeature(feature_name):
        batched[feature_name] = [item[feature_name] for item in batch]

    for feature_name in ['query', 'answer', 'candidates']:
        concatFeature(feature_name)

    return batched





