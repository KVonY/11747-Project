import json
import CorefGraph
import pickle

coref_grapher = CorefGraph.CorefGraph(200)
word2idx_path = "GloVe/6B.300_idx.pkl"
word2idx = pickle.load(open(word2idx_path,'rb'))
unknownWordId = 400000


# transform the sentence tokens to ids
def stringListToIdList(words):
    idList = [word2idx[word.lower()] if word in word2idx else unknownWordId for word in words]
    return idList

def preprocessJson(input_path, output_path, word2idx_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    error_idx = []
    for i, sample in enumerate(data):
        print(i)
        concat_document = ""
        for document in sample['supports']:
            concat_document += document
        try:
            Ei, Eo, Ri, Ro, R_start, R_end, document_tokens = coref_grapher(concat_document)
            sample['Ei'] = Ei
            sample['Eo'] = Eo
            sample['Ri'] = Ri
            sample['Ro'] = Ro
            sample['R_start'] = R_start
            sample['R_end'] = R_end
            sample['supports'] = stringListToIdList(document_tokens)
        except:
            print("error:", i)
            error_idx.append(i)

    with open(output_path, 'w+') as f:
        json.dump(data, f)



preprocessJson("data/wikihop/train.json", "data/wikihop/train_coref.json", word2idx_path)
preprocessJson("data/wikihop/dev.json", "data/wikihop/dev_coref.json", word2idx_path)