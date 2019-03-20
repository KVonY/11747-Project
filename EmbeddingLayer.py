import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# import json
# config_p = "config.json"

class InputEmbeddingLayer(torch.nn.Module):
    def __init__(self, W_init, config):
        super(InputEmbeddingLayer, self).__init__()

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


# def load_config(config_p):
#     with open(config_p, 'r') as config_file:
#         config = json.load(config_file)
#     if config['stopping_criterion'] == 'True':
#         config['stopping_criterion'] = True
#     else:
#         config['stopping_criterion'] = False
#     return config

# xx = np.ones((10, 200))
# yy = np.ones((10, 200, 15))

# config = load_config(config_p)
# W_init_test = np.random.uniform(0.0, 1.0, (1234, 300))

# nn_net = InputEmbeddingLayer(W_init_test, config)

# x, y = nn_net(xx, yy, xx, yy, 1, 2)

# print(x.shape)
# print(y.shape)

