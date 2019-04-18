import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn


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


def load_config(config_p):
    with open(config_p, 'r') as config_file:
        config = json.load(config_file)

    if config['stopping_criterion'] == 'True':
        config['stopping_criterion'] = True
    else:
        config['stopping_criterion'] = False
    
    if len(sys.argv) > 3:
        if str(sys.argv[3]) == 'log':
            try:
                os.remove(iter_10_p)
            except:
                print('no log file')
            try:
                os.remove(iter_50_p)
            except:
                print('no log file')
            try:
                os.remove(dev_10_p)
            except:
                print('no log file')
            try:
                os.remove(dev_whole_p)
            except:
                print('no log file')

    return config


def build_dict(vocab_p, vocab_char_p):
    vocab_data = open(vocab_p, 'r', encoding="utf-8").readlines()
    vocab_c_data = open(vocab_char_p, 'r', encoding="utf-8").readlines()

    vocab_dict = {}  # key: token, val: cnt
    vocab_c_dict = {}  # key: char, val: cnt

    for one_line in vocab_data:
        tmp_list = one_line.rstrip('\n').split('\t')
        vocab_dict[tmp_list[0]] = int(tmp_list[1])

    for one_line in vocab_c_data:
        tmp_list = one_line.rstrip('\n').split('\t')
        vocab_c_dict[tmp_list[0]] = int(tmp_list[1])

    vocab_ordered_list = sorted(vocab_dict.items(), key=lambda item:item[1], reverse=True)
    vocal_c_ordered_list = sorted(vocab_c_dict.items(), key=lambda item:item[1], reverse=True)

    vocab_index_dict = {}  # key: token, val: index
    vocab_c_index_dict = {}  # key: char, val: index

    for index, one_tuple in enumerate(vocab_ordered_list):
        vocab_index_dict[one_tuple[0]] = index
    
    for index, one_tuple in enumerate(vocal_c_ordered_list):
        vocab_c_index_dict[one_tuple[0]] = index

    return vocab_index_dict, vocab_c_index_dict


def load_word2vec_embedding(w2v_p, vocab_dict):
    w2v_data = open(w2v_p, 'r', encoding="utf-8").readlines()

    info = w2v_data[0].split()
    embed_dim = int(info[1])

    vocab_embed = {}  # key: token, value: embedding

    for line_index in range(1, len(w2v_data)):
        line = w2v_data[line_index].split()
        embed_part = [float(ele) for ele in line[1:]]
        vocab_embed[line[0]] = np.array(embed_part, dtype='float32')

    vocab_size = len(vocab_dict)
    W = np.random.randn(vocab_size, embed_dim).astype('float32')
    exist_cnt = 0

    for token in vocab_dict:
        if token in vocab_embed:
            token_index = vocab_dict[token]
            W[token_index,:] = vocab_embed[token]
            exist_cnt += 1

    print("%d/%d vocabs are initialized with word2vec embeddings." % (exist_cnt, vocab_size))
    return W, embed_dim


def get_doc_index_list(doc, token_dict, unk_dict):
    ret = []
    for token in doc:
        if token in token_dict:
            ret.append(token_dict[token])
        else:
            ret.append(unk_dict[token])
    return ret


def get_doc_index_list_cut_sen(doc, token_dict, unk_dict, config):
    ret = []

    sen_start_end_list = []
    cur_start = 0
    max_sen_cut = config["max_sen_len"]

    for index, token in enumerate(doc):
        if token == '.':
            if cur_start <= index - 1:
                sen_start_end_list.append([cur_start, index])
            cur_start = index + 1

        if token in token_dict:
            ret.append(token_dict[token])
        else:
            ret.append(unk_dict[token])

    if cur_start <= len(doc)-1:
        sen_start_end_list.append([cur_start, len(doc)])

    sen_start_end_list = sen_start_end_list[0: min(max_sen_cut, len(sen_start_end_list))]

    return ret, sen_start_end_list


def get_char_index_list(doc, char_dict, max_word_len):
    ret = []
    for token in doc:
        one_res = []
        for index in range(len(token)):
            one_char = token[index]
            if one_char in char_dict:
                one_res.append(char_dict[one_char])
            else:
                one_res.append(char_dict["__unkchar__"])
        ret.append(one_res[:max_word_len])
    return ret


def generate_examples(input_p, vocab_dict, vocab_c_dict, config, data_type):
    max_chains = config['max_chains']
    max_doc_len = config['max_doc_len']
    num_unks = config["num_unknown_types"]
    max_word_len = config["max_word_len"]

    ret = []
    print("begin loading " + data_type + " data")

    sen_list = []
    n_sen_list = []

    with open(input_p, 'r', encoding="utf-8") as infile:
        for index, one_line in enumerate(infile):
            data = json.loads(one_line.rstrip('\n'))

            doc_raw = data["document"].split()[:max_doc_len]
            qry_raw = data["query"].split()

            doc_lower = [t.lower() for t in doc_raw]
            qry_lower = [t.lower() for t in qry_raw]
            ans_lower = [t.lower() for t in data["answer"].split()]
            can_lower = [[t.lower() for t in cand] for cand in data["candidates"]]

            #------------------------------------------------------------------------
            # build oov dict for each example
            all_token = doc_lower + qry_lower + ans_lower
            for one_cand in can_lower:
                all_token += one_cand

            oov_set = set()
            for token in all_token:
                if token not in vocab_dict:
                    oov_set.add(token)

            unk_dict = {}  # key: token, val: index
            for ii, token in enumerate(oov_set):
                unk_dict[token] = vocab_dict["__unkword%d__" % (ii % num_unks)]
            
            #------------------------------------------------------------------------
            # tokenize
            # doc_words = get_doc_index_list(doc_lower, vocab_dict, unk_dict)
            doc_words, sen_cut = get_doc_index_list_cut_sen(doc_lower, vocab_dict, unk_dict, config)
            sen_list.append(sen_cut)
            n_sen_list.append(len(sen_cut))

            qry_words = get_doc_index_list(qry_lower, vocab_dict, unk_dict)
            ans_words = get_doc_index_list(ans_lower, vocab_dict, unk_dict)
            can_words = []
            for can in can_lower:
                can_words.append(get_doc_index_list(can, vocab_dict, unk_dict))

            doc_chars = get_char_index_list(doc_raw, vocab_c_dict, max_word_len)
            qry_chars = get_char_index_list(qry_raw, vocab_c_dict, max_word_len)

            #------------------------------------------------------------------------
            # other information
            annotations = data["annotations"]
            sample_id = data["id"]
            mentions = data["mentions"]
            corefs = data["coref_onehot"][:max_chains-1]

            one_sample = [doc_words, qry_words, ans_words, can_words, doc_chars, qry_chars]
            one_sample += [corefs, mentions, annotations, sample_id]

            ret.append(one_sample)
            
            if data_type == "train" and len(sys.argv) > 2:
                n_train = int(sys.argv[1])
                if index > n_train: break  # for train
            
            if data_type == "dev" and len(sys.argv) > 2:
                n_dev = int(sys.argv[2])
                if index > n_dev: break  # for dev
            
            if index % 2000 == 0: print("loading progress: " + str(index))
    
    print("max number of sentences: " + str(max(n_sen_list)))
    return ret, sen_list

