import sys
import os
import json
import numpy as np
import model_ha as model
import torch
import torch.nn as nn

from pre_process import load_config
from pre_process import build_dict
from pre_process import load_word2vec_embedding
from pre_process import generate_examples

from utils import generate_batch_data
from utils import extract_data
from utils import cal_acc
from utils import evaluate_result


# check CPU or GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using " + str(device))

# config path
config_path = "config.json"

# use GloVe pre-trained embedding
word_embedding_path = "GloVe/word2vec_glove.txt"

# vocab file for tokens in a specific dataset
vocab_path = "data/wikihop/vocab.txt"

# vocab file for chars in a specific dataset
vocab_char_path = "data/wikihop/vocab.txt.chars"

# train and dev set
train_path = "data/wikihop/training.json"
valid_path = "data/wikihop/validation.json"

# model save path
torch_model_p = "model/coref.pkl"


def main():
    # load config file
    config = load_config(config_path)

    # build dict for token (vocab_dict) and char (vocab_c_dict)
    vocab_dict, vocab_c_dict = build_dict(vocab_path, vocab_char_path)

    # load pre-trained embedding
    # W_init: token index * token embeding
    # embed_dim: embedding dimension
    W_init, embed_dim = load_word2vec_embedding(word_embedding_path, vocab_dict)
    
    K = 3

    # generate train/valid examples
    train_data, sen_cut_train = generate_examples(train_path, vocab_dict, vocab_c_dict, config, "train")
    dev_data, sen_cut_dev = generate_examples(valid_path, vocab_dict, vocab_c_dict, config, "dev")

    #------------------------------------------------------------------------
    # training process begins
    hidden_size = config['nhidden']
    batch_size = config['batch_size']

    coref_model = model.CorefQA(hidden_size, batch_size, K, W_init, config).to(device)

    if len(sys.argv) > 4 and str(sys.argv[4]) == "load":
        try:
            coref_model.load_state_dict(torch.load(torch_model_p))
            print("saved model loaded")
        except:
            print("no saved model")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(coref_model.parameters(), lr=config['learning_rate']) # TODO: use hyper-params in paper

    iter_index = 0
    batch_acc_list = []
    batch_loss_list = []
    dev_acc_list = []

    max_iter = int(config['num_epochs'] * len(train_data) / batch_size)
    print("max iteration number: " + str(max_iter))

    while True:
        # building batch data
        # batch_xxx_data is a list of batch data (len 15)
        # [dw, m_dw, qw, m_qw, dc, m_dc, qc, m_qc, cd, m_cd, a, dei, deo, dri, dro]
        batch_train_data, sen_cut_batch = generate_batch_data(train_data, config, "train", -1, sen_cut_train)  # -1 means random sampling
        # dw, m_dw, qw, m_qw, dc, m_dc, qc, m_qc, cd, m_cd, a, dei, deo, dri, dro = batch_train_data

        print(len(sen_cut_batch))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        dw, dc, qw, qc, cd, cd_m = extract_data(batch_train_data)
        cand_probs = coref_model(dw, dc, qw, qc, cd, cd_m, sen_cut_batch) # B x Cmax

        answer = torch.tensor(batch_train_data[10]).type(torch.LongTensor) # B x 1
        loss = criterion(cand_probs, answer)

        # evaluation process
        acc_batch = cal_acc(cand_probs, answer, batch_size)
        batch_acc_list.append(acc_batch)
        batch_loss_list.append(loss)
        dev_acc_list = evaluate_result(iter_index, config, dev_data, batch_acc_list, batch_loss_list, dev_acc_list, coref_model, sen_cut_dev)

        # save model
        if iter_index % config['model_save_frequency'] == 0 and len(sys.argv) > 4:
            torch.save(coref_model.state_dict(), torch_model_p)

        # back-prop
        loss.backward()
        optimizer.step()

        # check stopping criteria
        iter_index += 1
        if iter_index > max_iter: break


if __name__ == "__main__":
    main()
