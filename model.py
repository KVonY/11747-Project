import torch
import torch.nn as nn

import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, emb_weights_path, emb_idx2word_path, emb_word2idx_path):
        super(EmbeddingLayer, self).__init__()
        #self.embedding = nn.Embedding.from_pretrained(GloVe(name="6B"), freeze=True)
        vector2d = pickle.load(open(emb_weights_path,'rb'))
        self.id2words = pickle.load(open(emb_idx2word_path,'rb'))
        self.word2idx = pickle.load(open(emb_word2idx_path,'rb'))
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(vector2d), freeze=True)
        self.numVocab = len(self.id2words)
        
    def forward(self, sentence):  # sentence is a tensor of wordID, maybe batched
        emb = self.embedding(sentence)
        return emb

class BiGRU(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, batch_size):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=1, bias=True, bidirectional=True, batch_first=True)
        self.batch_size = batch_size
        self.emb_size = emb_size
        
        numLayersTimesNumDirections = 2
        self.h0 = torch.randn(numLayersTimesNumDirections, self.batch_size, self.emb_size)
    
    def forward(self, input_seq_emb):
        seq_emb, hn = self.gru(input_seq_emb, self.h0)
        return seq_emb


class GatedAttentionLayer(torch.nn.Module):
    def __init__(self):
        super(GatedAttentionLayer, self).__init__()
    
    # compute gated-attention query-aware context sequence embeddings
    # context_emb, query_emb shape: (batch_size, seq_len, emb_dim)
    # output: query_aware_context (batch_size, context_seq_len, emb_dim)
    def forward(self, context_emb, query_emb):
        context_tr = context_emb.transpose(1,2) # (batch, emb_dim, seq)
        temp = torch.matmul(query_emb, context_tr)  # (batch, seq_query, seq_context)
        # softmax along query sequence dimension (for each context word, compute prob dist over all query words)
        querySoftmax = nn.Softmax(dim=1)
        alpha = querySoftmax(temp)  # (batch, seq_query, seq_context)
        # for each context word, compute weighted average of queries
        attention_weighted_query = torch.matmul(query_emb.transpose(1,2), alpha).transpose(1,2) # (batch, seq_context, emb_dim)
        # final element-multiplication to get new context embedding X
        query_aware_context = torch.mul(context_emb, attention_weighted_query) # (batch, seq_context, emb_dim)
        return query_aware_context


class AnswerPredictionLayer(torch.nn.Module):
    def __init__(self):
        super(AnswerPredictionLayer, self).__init__()
        
    def forward(self, sequence_emb, query_emb, clozeIdx):
        query_at_cloze = query_emb[:,clozeIdx,:]
        context_tr = context_emb.transpose(1,2) # (batch, emb_dim, seq)
        temp = torch.matmul(query_at_cloze, context_tr)  # (batch, 1, seq_context)
        # softmax along context sequence dimension (compute prob dist over all context words)
        querySoftmax = nn.Softmax(dim=2).squeeze()  # (batch, seq_context)
        return querySoftmax

class CorefQA(torch.nn.Module):
    def __init__(self, hidden_size, batch_size, K):
        super(CorefQA, self).__init__()
        self.embedding = EmbeddingLayer(emb_weights_path, emb_idx2word_path, emb_word2idx_path)
        embedding_size = 300 # GloVe 300d
        self.gru = BiGRU(embedding_size, hidden_size, batch_size)
        self.ga = GatedAttentionLayer()
        self.pred = AnswerPredictionLayer()
        self.K = K
    
    def forward(self, context, query, clozeIndex):
        context_emb = self.gru(self.embedding(context))
        query_emb = self.gru(self.embedding(query))
        
        # K layers of gated attention
        for i in range(self.K):
            context_emb = ga(context_emb, query_emb)
            
        # output layer
        return self.pred(context_emb, query_emb, clozeIndex)