import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn

# log files
log_path = "logs/"
iter_10_p = log_path + 'iter_10_acc.txt'
iter_50_p = log_path + 'iter_50_acc.txt'
dev_10_p = log_path + 'dev_10_acc.txt'
dev_whole_p = log_path + 'dev_whole_acc.txt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_batch_data(data, config, data_type, batch_i, sen_cut):
    max_word_len = config['max_word_len']
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
        doc_w, qry_w, ans, cand, doc_c, qry_c, _, _, _, _ = data[index]
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

    a = np.zeros((batch_size, ), dtype='int32')    # correct answer

    #------------------------------------------------------------------------
    for n in range(batch_size):
        doc_w, qry_w, ans, cand, doc_c, qry_c, _, _, _, _ = data[batch_index[n]]

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
                a[n] = it # answer

    ret = [dw, m_dw, qw, m_qw, dc, m_dc, qc, m_qc, cd, m_cd, a]
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