import wikihopDataset

hidden_size = 4
batch_size = 5
K = 2
model = CorefQA(hidden_size,batch_size,K)
dataset = wikihopDataset.wikihopDataset(train_data_path, emb_word2idx_path)
training_set = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4, collate_fn=wikihopDataset.wikihopBatchCollate)

temp = None
for sample in training_set:
    temp = sample
    break
temp