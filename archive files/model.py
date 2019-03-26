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

# Do not remove! This is for query hidden representation! Need to use normal GRU
class BiGRU(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, batch_size):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=1, bias=True, bidirectional=True, batch_first=True)
        self.batch_size = batch_size
        self.emb_size = emb_size
        
        numLayersTimesNumDirections = 2
        self.h0 = torch.randn(numLayersTimesNumDirections, self.batch_size, self.emb_size, requires_grad=True)
    
    def forward(self, input_seq_emb):
        seq_emb, hn = self.gru(input_seq_emb, self.h0)
        return seq_emb


class GatedAttentionLayer(torch.nn.Module):
    def __init__(self):
        super(GatedAttentionLayer, self).__init__()
        self.softmax1 = nn.Softmax(dim=1)
    # compute gated-attention query-aware context sequence embeddings
    # context_emb, query_emb shape: (batch_size, seq_len, emb_dim)
    # output: query_aware_context (batch_size, context_seq_len, emb_dim)
    def forward(self, context_emb, query_emb):
        context_tr = context_emb.transpose(1,2) # (batch, emb_dim, seq)
        temp = torch.matmul(query_emb, context_tr)  # (batch, seq_query, seq_context)
        # softmax along query sequence dimension (for each context word, compute prob dist over all query words)
        alpha = self.softmax1(temp)  # (batch, seq_query, seq_context)
        # for each context word, compute weighted average of queries
        attention_weighted_query = torch.matmul(query_emb.transpose(1,2), alpha).transpose(1,2) # (batch, seq_context, emb_dim)
        # final element-multiplication to get new context embedding X
        query_aware_context = torch.mul(context_emb, attention_weighted_query) # (batch, seq_context, emb_dim)
        return query_aware_context


class AnswerPredictionLayer(torch.nn.Module):
    def __init__(self):
        super(AnswerPredictionLayer, self).__init__()
        self.softmax1 = nn.Softmax(dim=1)
    
    # doc_emb: B x N x 2Dh
    # query_emb: B x Q x 2Dh
    # Dh: hidden layer size of normal GRU for query embedding
    # cand: B x N x C (float)
    # cmask: B x N (float)
    def forward(self, doc_emb, query_emb, Dh, cand, cmask):
        q = torch.cat([query_emb[:,-1,:Dh], query_emb[:,0,Dh:]], dim=2).transpose(1,2) # B x 2Dh x 1
        p = torch.matmul(doc_emb, q).squeeze() # final query-aware document embedding: B x N
        prob = self.softmax1(p) # prob dist over document words, relatedness between word to entire query: B x N
        probmasked = prob * cmask + 1e-7  # B x N
        sum_probmasked = torch.sum(probmasked, 1) # B x 1
        probmasked = probmasked / sum_probmasked # B x N
        probmasked = probmasked.unsqueezed(1) # B x 1 x N
        probCandidate = torch.matmul(probmasked, cand).squeeze() # prob over candidates: B x C
        return probCandidate


class CorefQA(torch.nn.Module):
    def __init__(self, hidden_size, batch_size, K):
        super(CorefQA, self).__init__()
        self.embedding = EmbeddingLayer(emb_weights_path, emb_idx2word_path, emb_word2idx_path)
        embedding_size = 300 # GloVe 300d
        self.query_grus = [BiGRU(embedding_size, hidden_size, batch_size) for _ in range(K)]
        self.context_grus = [BiGRU(embedding_size, hidden_size, batch_size) for _ in range(K)] # TODO: swap to coref-gru
        self.ga = GatedAttentionLayer() # non-parametrized
        self.pred = AnswerPredictionLayer() # non-parametrized
        self.K = K
        self.hidden_size = hidden_size
    
    def forward(self, batch_data):
        # parse input
        context, context_mask, query, query_mask, context_char, context_char_mask, query_char, query_char_mask, \
            candidate, candidate_mask, a, dei, deo, dri, dro = batch_data

        # get embedding
        query_embedding = self.embedding(query, query_mask, query_char, query_char_mask)
        context_embedding = self.embedding(context, context_mask, context_char, context_char_mask)
        context_prev_layer = context_embedding
        
        # K - 1 layers of gated attention
        for i in range(self.K - 1):
            # run gru on query embedding
            query_gru = self.query_grus[i]
            query_hidden = query_gru(query_embedding)

            # run coref-gru on previous context layer
            context_gru = self.context_gru[i]
            context_hidden = context_gru(context_prev_layer)
            
            # update context hidden layer using ga
            context_prev_layer = ga(context_hidden, query_hidden)

        # get K-th context hidden representation
        context_gru = self.context_gru[-1]
        final_context_hidden = context_gru(context_prev_layer)
        
        # get K-th query hidden representation
        query_gru = self.query_grus[-1]
        final_query_hidden = query_gru(query_embedding)

        # the K-th layer is the answer prediction layer
        candidate_probs = self.pred(final_context_hidden, final_query_hidden, self.hidden_size, candidate.float(), candidate_mask.float())
            
        # output layer
        return candidate_probs # B x Cmax