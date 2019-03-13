import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import CorefGraph

class wikihopDataset(Dataset):
    def __init__(self, dataJsonPath, emb_word2idx_path):
        with open(dataJsonPath, 'r') as f:
            self.data = json.load(f)
        self.coref_grapher = CorefGraph.CorefGraph()
        self.word2idx = pickle.load(open(emb_word2idx_path,'rb'))
        self.unknownWordId = 400000
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        concat_document = ""
        for document in sample['supports']:
            concat_document += document
        Ei, Eo, Ri, Ro, R_start, R_end, document_tokens = self.coref_grapher(concat_document)
        sample['Ei'] = Ei
        sample['Eo'] = Eo
        sample['Ri'] = Ri
        sample['Ro'] = Ro
        sample['R_start'] = R_start
        sample['R_end'] = R_end

        def simpleTokenize(sentence):
            return sentence.split().lower()

        # transform the sentence tokens to ids
        def stringListToIdList(words):
            idList = [self.word2idx(word) if word in self.word2idx else self.unknownWordId for word in words]
            return idList

        sample['supports'] = stringListToIdList(document_tokens)
        sample['query'] = stringListToIdList(simpleTokenize(sample['query']))
        sample['answer'] = stringListToIdList(simpleTokenize(sample['answer']))
        sample['candidates'] = [stringListToIdList(simpleTokenize(candidate)) for candidate in sample['candidates']]
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





