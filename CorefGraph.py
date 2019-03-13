from allennlp.predictors.predictor import Predictor

class CorefGraph(object):
    def __init__(self, max_num_per_window):
        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
        self.max_num_per_window = max_num_per_window

    # analyze pred, construct coref graph, and fill in data structures
    def pred_to_graph(self, pred, graph, idx_offset):
        for cluster in pred['clusters']:
            for i, span in enumerate(cluster[1:]): # there should be at least 2 spans in the cluster
                # find out the nodes in this relation
                # coref relation is from start of a span to end of previous span
                idx = i + 1 # we are starting from idx = 1
                word_idx = span[0] + idx_offset # add offset to get index in original long context
                prev_span = cluster[idx-1]
                coreferenced_word_idx = prev_span[1] + idx_offset
                graph[word_idx] = coreferenced_word_idx

    def __call__(self, long_document_str):
        graph = {}

        long_document_words = long_document_str.split()
        num_all_words = len(long_document_words)
        
        # sliding window (no overlap between windows)
        idx_start = 0
        all_tokens = []
        while idx_start < num_all_words:
            idx_end = min([num_all_words, idx_start + self.max_num_per_window])
            chunk_paragraph = " ".join(long_document_words[idx_start:idx_end])
            pred = self.predictor.predict(chunk_paragraph)
            token_idx_offset = len(all_tokens)
            self.pred_to_graph(pred, graph, token_idx_offset)
            all_tokens += pred["document"]
            idx_start += self.max_num_per_window

        # another sliding window pass (only the overlapping windows)
        idx_start = int(self.max_num_per_window / 2)
        num_all_tokens = len(all_tokens)
        while idx_start < num_all_tokens:
            idx_end = min([num_all_tokens, idx_start + self.max_num_per_window])
            chunk_paragraph = " ".join(all_tokens[idx_start:idx_end])
            pred = self.predictor.predict(chunk_paragraph)
            token_idx_offset = idx_start
            self.pred_to_graph(pred, graph, token_idx_offset)
            idx_start += self.max_num_per_window

        # now we have the complete graph, put it in edge/relation format
        # suppose C<-B<-A  (A refer to B, B refer to C)
        # each node can have at most 1 out-edge and at most 1 in-edge
        # for relation A->B (A coreference B), A has out-edge, B has in-edge
        # array indexed in A,B,C order
        #            Ei = [0,1,1]  True/False Mask, whether or not the node has in-edge
        #            Eo = [1,1,0]  True/False Mask, whether or not the node has out-edge
        #            R  = [(B->C), (A->B)] set of coreference pair (i.e. relations)
        #            Ri = [_,1,0], index of relation (of the in-edge)
        #            Ro = [1,0,_], index of relation (of the out-edge)
        Ei = [0 for i in range(num_all_tokens)]
        Eo = [0 for i in range(num_all_tokens)]
        R_start = []
        R_end = []
        Ri = [-1 for i in range(num_all_tokens)]
        Ro = [-1 for i in range(num_all_tokens)]
        for word_idx, coreferenced_word_idx in graph.items():
            # add relation to data structure
            R_start.append(word_idx)
            R_end.append(coreferenced_word_idx)                
            relation_id = len(R_start)
            Ri[coreferenced_word_idx] = relation_id
            Ro[word_idx] = relation_id
            Ei[coreferenced_word_idx] = 1
            Eo[word_idx] = 1

        return Ei, Eo, Ri, Ro, R_start, R_end, all_tokens

    def print_corefs(self, R_start, R_end, all_tokens):
        for r_start, r_end in zip(R_start, R_end):
            print(all_tokens[r_start-1: r_start + 2], all_tokens[r_end-1: r_end + 2])