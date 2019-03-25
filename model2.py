import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, W_init, config):
        super(EmbeddingLayer, self).__init__()

        self.num_token = W_init.shape[0]
        self.embed_dim = W_init.shape[1]
        self.char_dim = config["char_dim"]
        self.num_chars = config["num_characters"]
        self.filter_size = config["char_filter_size"]
        self.filter_width = config["char_filter_width"]

        self.token_emb_lookup = self.get_token_embedding(W_init)
        self.char_emb_lookup = self.get_char_embedding()
        self.fea_emb_lookup = self.get_feat_embedding()

        self.model_conv = nn.Conv2d(
            in_channels=self.char_dim, 
            out_channels=self.filter_size, 
            kernel_size=(1, self.filter_width), 
            stride=1)


    def prepare_input(self, d, q):
        f = np.zeros(d.shape).astype('int32')
        for i in range(d.shape[0]):
            f[i,:] = np.in1d(d[i,:],q[i,:])
        return f

    
    def get_feat_embedding(self):
        feat_embed_init = np.random.normal(0.0, 1.0, (2, 2))
        feat_embed = nn.Embedding(2, 2)
        feat_embed.weight.data.copy_(torch.from_numpy(feat_embed_init))
        feat_embed.weight.requires_grad = True  # update feat embedding
        return feat_embed


    def get_token_embedding(self, W_init):
        token_embedding = nn.Embedding(self.num_token, self.embed_dim)
        token_embedding.weight.data.copy_(torch.from_numpy(W_init))
        token_embedding.weight.requires_grad = True  # update token embedding
        return token_embedding


    def get_char_embedding(self):
        char_embed_init = np.random.uniform(0.0, 1.0, (self.num_chars, self.char_dim))
        char_emb = nn.Embedding(self.num_chars, self.char_dim)
        char_emb.weight.data.copy_(torch.from_numpy(char_embed_init))
        char_emb.weight.requires_grad = True  # update char embedding
        return char_emb

    
    def cal_char_embed(self, c_emb_init):
        doc_c_emb_new = c_emb_init.permute(0, 3, 1, 2)

        # get conv1d result: doc_c_emb
        doc_c_tmp = self.model_conv(doc_c_emb_new)
        
        # transfer back: B, W, N, H -> B, N, H, W
        doc_c_tmp = doc_c_tmp.permute(0, 2, 3, 1)
        doc_c_tmp = F.relu(doc_c_tmp)
        doc_c_emb = torch.max(doc_c_tmp, dim=2)[0]  # B x N x filter_size

        return doc_c_emb


    def forward(self, dw, dc, qw, qc, k_layer, K):
        # doc_w: B * N
        # doc_c: B * N * 10
        doc_w = torch.from_numpy(dw).type(torch.LongTensor)
        doc_c = torch.from_numpy(dc).type(torch.LongTensor)
        qry_w = torch.from_numpy(qw).type(torch.LongTensor)
        qry_c = torch.from_numpy(qc).type(torch.LongTensor)
        feat = torch.from_numpy(self.prepare_input(doc_w, qry_w)).type(torch.LongTensor)
    
        #----------------------------------------------------------
        doc_w_emb = self.token_emb_lookup(doc_w)  # B * N * emb_token_dim
        doc_c_emb_init = self.char_emb_lookup(doc_c)  # B * N * num_chars * emb_char_dim (B * N * 15 * 10)
        
        qry_w_emb = self.token_emb_lookup(qry_w)
        qry_c_emb_init = self.char_emb_lookup(qry_c)
        
        fea_emb = self.fea_emb_lookup(feat)  # B * N * 2

        #----------------------------------------------------------
        doc_c_emb = self.cal_char_embed(doc_c_emb_init)
        qry_c_emb = self.cal_char_embed(qry_c_emb_init)

        # concat token emb and char emb
        doc_emb = torch.cat((doc_w_emb, doc_c_emb), dim=2)
        qry_emb = torch.cat((qry_w_emb, qry_c_emb), dim=2)

        if k_layer == K-1:
            doc_emb = torch.cat((doc_emb, fea_emb), dim=2)
        
        return doc_emb, qry_emb


# Do not remove! This is for query hidden representation! Need to use normal GRU
class BiGRU(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, batch_size):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=1, bias=True, bidirectional=True, batch_first=True)
        self.batch_size = batch_size
        self.emb_size = emb_size
        
        numLayersTimesNumDirections = 2
        self.h0 = torch.randn(numLayersTimesNumDirections, self.batch_size, hidden_size, requires_grad=True)
    
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
        q = torch.cat((query_emb[:,-1,:Dh], query_emb[:,0,Dh:]), dim=1) # B x 2Dh
        # q = torch.cat([query_emb[:,-1,:Dh], query_emb[:,0,Dh:]], dim=2).transpose(1,2) # B x 2Dh x 1
        q = q[:,:,None] # B x 2Dh x 1
        p = torch.matmul(doc_emb, q).squeeze() # final query-aware document embedding: B x N
            
        prob = self.softmax1(p).type(torch.DoubleTensor) # prob dist over document words, relatedness between word to entire query: B x N
        probmasked = prob * cmask + 1e-7  # B x N
        
        sum_probmasked = torch.sum(probmasked, 1) # B x 1
        sum_probmasked = sum_probmasked[:,None]
        
        # probmasked = probmasked / sum_probmasked # B x N
        probmasked = torch.div(probmasked, sum_probmasked) # # B x N
        probmasked = probmasked.unsqueeze(1) # B x 1 x N

        probCandidate = torch.matmul(probmasked, cand).squeeze() # prob over candidates: B x C
        return probCandidate


class CorefQA(torch.nn.Module):
    def __init__(self, hidden_size, batch_size, K,  W_init, config):
        super(CorefQA, self).__init__()
        self.embedding = EmbeddingLayer(W_init, config)
        # embedding_size = W_init.shape[1]
        # embedding_size = 100 # GloVe 300d
        embedding_size = 150
        # self.query_grus = [BiGRU(embedding_size, hidden_size, batch_size) for _ in range(K)]
        # self.context_grus = [BiGRU(embedding_size, hidden_size, batch_size) for _ in range(K)] # TODO: swap to coref-gru
        self.ga = GatedAttentionLayer() # non-parametrized
        self.pred = AnswerPredictionLayer() # non-parametrized
        self.K = K
        self.hidden_size = hidden_size

        self.context_gru_1 = BiGRU(embedding_size, hidden_size, batch_size)
        self.query_gru_1 = BiGRU(embedding_size, hidden_size, batch_size)

        self.context_gru_2 = BiGRU(2*hidden_size, hidden_size, batch_size)
        self.query_gru_2 = BiGRU(embedding_size, hidden_size, batch_size)

        self.context_gru_3 = BiGRU(2*hidden_size, hidden_size, batch_size)
        self.query_gru_3 = BiGRU(embedding_size, hidden_size, batch_size)

    
    def forward(self, batch_data):
        # parse input
        context, context_mask, query, query_mask, context_char, context_char_mask, query_char, query_char_mask, \
            candidate, candidate_mask, a, dei, deo, dri, dro = batch_data

        context_embedding, query_embedding = self.embedding(
            context, context_char, 
            query, query_char, 
            0, self.K)

        #-----------------------------------------------------
        context_out_1 = self.context_gru_1(context_embedding)
        query_out_1 = self.query_gru_1(query_embedding)
        layer_out_1 = self.ga(context_out_1, query_out_1)
        # print(layer_out_1.shape)

        #-----------------------------------------------------
        context_out_2 = self.context_gru_2(layer_out_1)
        query_out_2 = self.query_gru_2(query_embedding)
        layer_out_2 = self.ga(context_out_2, query_out_2)
        # print(layer_out_2.shape)

        #-----------------------------------------------------
        context_out_3 = self.context_gru_3(layer_out_2)
        query_out_3 = self.query_gru_3(query_embedding)
        layer_out_3 = self.ga(context_out_3, query_out_2)
        # print(layer_out_3.shape)

        final_context_hidden = context_out_3
        final_query_hidden = query_out_3

        cand = torch.from_numpy(candidate).type(torch.DoubleTensor)
        cand_m = torch.from_numpy(candidate_mask).type(torch.DoubleTensor)

        candidate_probs = self.pred(
            final_context_hidden, 
            final_query_hidden, 
            self.hidden_size, 
            cand, 
            cand_m
            )
            
        # output layer
        return candidate_probs # B x Cmax