import sys
import os
import json
import numpy as np
import model_ha as model
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

# model save path
torch_model_p = "model/coref.pkl"

# log files
log_path = "logs/"
iter_10_p = log_path + 'iter_10_acc.txt'
iter_50_p = log_path + 'iter_50_acc.txt'
dev_10_p = log_path + 'dev_10_acc.txt'
dev_whole_p = log_path + 'dev_whole_acc.txt'

# check CPU or GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using " + str(device))


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


def get_graph(edges):
    dei, deo = edges
    dri, dro = np.copy(dei).astype("int32"), np.copy(deo).astype("int32")
    dri[:, :, 0] = 0
    dro[:, :, 0] = 0
    return dei, deo, dri, dro


def generate_batch_data(data, config, data_type, batch_i, sen_cut):
    max_word_len = config['max_word_len']
    max_chains = config['max_chains']
    batch_size = config['batch_size']
    
    n_data = len(data)
    max_doc_len, max_qry_len, max_cands = 0, 0, 0

    sen_cut_batch = []
    
    if batch_i == -1:
        batch_index = np.random.choice(n_data, batch_size, replace=True)
    else:
        batch_index = []
        start_i = batch_i * batch_size
        end_i = (batch_i + 1) * batch_size
        for tmp_i in range(start_i, end_i):
            batch_index.append(tmp_i)

    for index in batch_index:
        doc_w, qry_w, ans, cand, doc_c, qry_c, corefs, mentions, annotations, fname = data[index]
        max_doc_len = max(max_doc_len, len(doc_w))
        max_qry_len = max(max_qry_len, len(qry_w))
        max_cands = max(max_cands, len(cand))
        sen_cut_batch.append(sen_cut[index])

    #------------------------------------------------------------------------
    dw = np.zeros((batch_size, max_doc_len), dtype='int32') # document words
    m_dw = np.zeros((batch_size, max_doc_len), dtype='float32')  # document word mask
    qw = np.zeros((batch_size, max_qry_len), dtype='int32') # query words
    m_qw = np.zeros((batch_size, max_qry_len), dtype='float32')  # query word mask

    dc = np.zeros((batch_size, max_doc_len, max_word_len), dtype="int32")
    m_dc = np.zeros((batch_size, max_doc_len, max_word_len), dtype="float32")
    qc = np.zeros((batch_size, max_qry_len, max_word_len), dtype="int32")
    m_qc = np.zeros((batch_size, max_qry_len, max_word_len), dtype="float32")

    cd = np.zeros((batch_size, max_doc_len, max_cands), dtype='int32')   # candidate answers
    m_cd = np.zeros((batch_size, max_doc_len), dtype='float32') # candidate mask

    edges_in = np.zeros((batch_size, max_doc_len, max_chains), dtype="float32")
    edges_out = np.zeros((batch_size, max_doc_len, max_chains), dtype="float32")
    edges_in[:, :, 0] = 1.
    edges_out[:, :, 0] = 1.

    a = np.zeros((batch_size, ), dtype='int32')    # correct answer
    # fnames = ['']*batch_size
    # annots = []

    #------------------------------------------------------------------------
    for n in range(batch_size):
        doc_w, qry_w, ans, cand, doc_c, qry_c, corefs, mentions, annotations, fname = data[batch_index[n]]

        # document and query
        dw[n, :len(doc_w)] = doc_w
        qw[n, :len(qry_w)] = qry_w
        m_dw[n, :len(doc_w)] = 1
        m_qw[n, :len(qry_w)] = 1
        for t in range(len(doc_c)):
            dc[n, t, :len(doc_c[t])] = doc_c[t]
            m_dc[n, t, :len(doc_c[t])] = 1
        for t in range(len(qry_c)):
            qc[n, t, :len(qry_c[t])] = qry_c[t]
            m_qc[n, t, :len(qry_c[t])] = 1

        # search candidates in doc
        for it, cc in enumerate(cand):
            index = [ii for ii in range(len(doc_w)) if doc_w[ii] in cc]
            m_cd[n, index] = 1
            cd[n, index, it] = 1
            if ans == cc: 
                found_answer = True
                a[n] = it # answer

        # graph edges
        for ic, chain in enumerate(corefs):
            for item in chain:
                if item[2] != -1:
                    if mentions[item[2]][0] < max_doc_len:
                        edges_in[n, mentions[item[2]][0], ic+1] = 1.
                if item[0] != -1:
                    if mentions[item[0]][1]-1 < max_doc_len:
                        edges_out[n, mentions[item[0]][1]-1, ic+1] = 1.

        # annots.append(annotations)
        # fnames[n] = fname

    dei, deo, dri, dro = get_graph((edges_in, edges_out))
    ret = [dw, m_dw, qw, m_qw, dc, m_dc, qc, m_qc, cd, m_cd, a, dei, deo, dri, dro]
    return ret, sen_cut_batch


def cal_acc(cand_probs, answer, batch_size):
    cand_a = torch.argmax(cand_probs, dim=1)
    acc_cnt = 0
    for acc_i in range(batch_size):
        if cand_a[acc_i] == answer[acc_i]: acc_cnt += 1
    return acc_cnt / batch_size


def extract_data(batch_data):
    context = torch.from_numpy(batch_data[0]).type(torch.LongTensor).to(device)
    context_char = torch.from_numpy(batch_data[4]).type(torch.LongTensor).to(device)
    query = torch.from_numpy(batch_data[2]).type(torch.LongTensor).to(device)
    query_char = torch.from_numpy(batch_data[6]).type(torch.LongTensor).to(device)
    candidate = torch.from_numpy(batch_data[8]).type(torch.DoubleTensor)
    candidate_mask = torch.from_numpy(batch_data[9]).type(torch.DoubleTensor)
    return context, context_char, query, query_char, candidate, candidate_mask


def evaluate_result(iter_index, config, dev_data, batch_acc_list, batch_loss_list, dev_acc_list, coref_model, sen_cut_dev):
    if iter_index % config['logging_frequency'] == 0:
        n = len(batch_acc_list)
        if n > 15:
            acc_aver = 0
            loss_aver = 0
            for i in range(n-10, n):
                acc_aver += batch_acc_list[i] / 10
                loss_aver += batch_loss_list[i] / 10

            print("iter (10) -- acc: " + str(round(acc_aver, 4)) + ", loss: " + str(round(loss_aver.data.item(), 4)))
            if len(sys.argv) > 3:
                if str(sys.argv[3]) == 'log':
                    with open(iter_10_p, 'a') as of1:
                        of1.writelines(str(acc_aver) + ',' + str(loss_aver) + '\n')

        if n > 55:
            acc_aver = 0
            loss_aver = 0
            for i in range(n-50, n):
                acc_aver += batch_acc_list[i] / 50
                loss_aver += batch_loss_list[i] / 50
            print("iter (50) -- acc: " + str(round(acc_aver, 4)) + ", loss: " + str(round(loss_aver.data.item(), 4)))
            if len(sys.argv) > 3:
                if str(sys.argv[3]) == 'log':
                    with open(iter_50_p, 'a') as of2:
                        of2.writelines(str(acc_aver) + ',' + str(loss_aver) + '\n')

    if iter_index % config['validation_frequency'] == 0:
        dev_data_batch, sen_cut_batch = generate_batch_data(dev_data, config, "dev", -1, sen_cut_dev)  # -1 means random sampling

        dw, dc, qw, qc, cd, cd_m = extract_data(dev_data_batch)
        cand_probs_dev = coref_model(dw, dc, qw, qc, cd, cd_m, sen_cut_batch)

        answer_dev = torch.tensor(dev_data_batch[10]).type(torch.LongTensor)
        acc_dev = cal_acc(cand_probs_dev, answer_dev, config['batch_size'])
        dev_acc_list.append(acc_dev)

        aver_dev_acc = 0
        if len(dev_acc_list) > 15:
            tmp_list = dev_acc_list[len(dev_acc_list)-10: len(dev_acc_list)]
            aver_dev_acc = sum(tmp_list) / 10

        print("-- dev acc: " + str(round(acc_dev, 4)) + ', aver dev acc: ' + str(round(aver_dev_acc, 4)))
        if len(sys.argv) > 3:
            if str(sys.argv[3]) == 'log':
                with open(dev_10_p, 'a') as of3:
                    of3.writelines(str(acc_dev) + ',' + str(aver_dev_acc) + '\n')
    
    if iter_index % config['validation_frequency_whole_dev'] == 0:
        n_batch_data = int(len(dev_data) / config['batch_size']) - 1
        acc_dev_list = []
        
        for batch_i in range(n_batch_data):
            dev_data_batch, sen_cut_batch = generate_batch_data(dev_data, config, "dev", batch_i, sen_cut_dev)

            dw, dc, qw, qc, cd, cd_m = extract_data(dev_data_batch)
            cand_probs_dev = coref_model(dw, dc, qw, qc, cd, cd_m, sen_cut_batch)

            answer_dev = torch.tensor(dev_data_batch[10]).type(torch.LongTensor)
            acc_dev = cal_acc(cand_probs_dev, answer_dev, config['batch_size'])
            acc_dev_list.append(acc_dev)
        
        acc_dev_whole = sum(acc_dev_list) / n_batch_data
        print("---- dev acc whole: " + str(round(acc_dev_whole, 4)))

        if len(sys.argv) > 3:
            if str(sys.argv[3]) == 'log':
                with open(dev_whole_p, 'a') as of4:
                    of4.writelines(str(acc_dev_whole) + '\n')

    return dev_acc_list


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
        del batch_train_data
        iter_index += 1
        if iter_index > max_iter: break


if __name__ == "__main__":
    main()
