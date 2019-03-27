import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn

def glorot(d1,d2):
    return np.sqrt(6./(d1+d2))

class CorefGRU(nn.Module):
    """Coref-GRU model which uses coreference to update hidden states of a GRU.

    This class is designed to work with any Directed Acyclic Graph (DAG) of
    annotations over the input sequence, and output a sequence of vectors as the
    output. Full details of this layer are described in the following paper:

    ```
    Linguistic Knowledge as Memory for Recurrent Neural Networks
    Bhuwan Dhingra, Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov
    https://arxiv.org/pdf/1703.02620.pdf
    ```

    As a special case the DAG might only consist of coreference annotations as
    described in the paper:

    ```
    Neural Models for Reasoning over Multiple Mentions using Coreference
    Bhuwan Dhingra, Qiao Jin, Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov
    NAACL, 2018
    http://aclweb.org/anthology/N18-2007
    ```

    In this repository we only use this class with coreferences.

    To use this layer, first initialize the model and then call compute with the
    input tensors:

    ```
    cgru = CorefGRU(num_relations, indim, relationdim, max_chains, reverse=False)
    out, mem, agg = cgru.compute(inp, mask, edgein, edgeout, relin, relout)
    ```

    See `compute()` for more details about the inputs to that function.

    Args:
        num_relations: Number of distinct relations in the DAG, including the
            sequential next-word relation. For coreference, this will be 2.
        input_dim: Dimensionality of input.
        relation_dim: Hidden state size per relation. The actual output size of
            this layer will be `num_relations * relation_dim`.
        max_chains: Number of linear chains the DAG is decomposed into. For
            coreference, we assume each chain corresponds to one entity cluster,
            hence this is equal to the maximum number of clusters in any input,
            plus one for the sequential relationship.
        reverse: (Optional) If true processes the sequence in backwards. This
            is used for bidirectional models.
        concat: (Deprecated) If true concatenates the incoming hidden states
            instead of an attention mechanism.
    """

    def __init__(self, num_relations, input_dim, relation_dim, max_chains, 
            reverse=False, concat=False):
        super(CorefGRU, self).__init__()
        self.num_relations = num_relations
        self.rdims = relation_dim
        self.input_dim = input_dim
        self.output_dim = self.num_relations*self.rdims
        self.max_chains = max_chains
        self.reverse = reverse
        self.concat = concat
        h_to_h = self.rdims*self.num_relations
        self.W = torch.randn(self.input_dim, self.output_dim)*glorot(self.input_dim,self.output_dim)
        self.U = torch.randn(h_to_h, self.output_dim)*glorot(h_to_h,self.output_dim)
        self.b = torch.zeros(self.output_dim, dtype=torch.float32)


        # initialize gates
        # def _gate_params(name):
        #     gate = {}
        #     #h_to_h = self.rdims*self.num_relations if self.concat else self.rdims
        #     h_to_h = self.rdims*self.num_relations
        #     gate["W"] = tf.Variable(tf.random_normal((self.input_dim,self.output_dim), 
        #         mean=0.0, stddev=glorot(self.input_dim,self.output_dim)),
        #         name="W"+name, dtype=tf.float32)
        #     gate["U"] = tf.Variable(tf.random_normal((h_to_h,self.output_dim),
        #         mean=0.0, stddev=glorot(h_to_h,self.output_dim)),
        #         name="U"+name, dtype=tf.float32)
        #     gate["b"] = tf.Variable(tf.zeros((self.output_dim,)), 
        #         name="b"+name, dtype=tf.float32)
        #     return gate
        # self.resetgate = _gate_params("r")
        # self.updategate = _gate_params("u")
        # self.hiddengate = _gate_params("h")
        self.resetgate = {"W": self.W, "U": self.U, "b": self.b}
        self.updategate = {"W": self.W, "U": self.U, "b": self.b}
        self.hiddengate = {"W": self.W, "U": self.U, "b": self.b}
        self.Wstacked = torch.cat([self.resetgate["W"], self.updategate["W"],
                self.hiddengate["W"]], 1) # Din x 3Dout
        self.Ustacked = torch.cat([self.resetgate["U"], self.updategate["U"],
                self.hiddengate["U"]], 1) # Dr x 3Dout

        # initialize attention params
        if not self.concat:
            # self.Watt = tf.Variable(tf.random_normal((self.num_relations,self.input_dim),
            #     mean=0.0, stddev=0.1),
            #     name="Watt", dtype=tf.float32) # Dr x Din
            self.Watt = torch.randn(self.num_relations, self.input_dim)*0.1

        # initialize initial memory state
        self.mem_init = torch.zeros((self.max_chains, self.rdims),
                                 dtype=torch.float32)

    def forward(self, X, M, Ei, Eo, Ri, Ro, init=None, mem_init=None):
        """Apply Coref-GRU layer to the given tensors.

        The input DAG is parameterized using four tensors described below.

        Assume that B is the batch size, N is the max sequence length, Din is
        the size of input embeddings, Dout is the size of the output embeddings,
        Drel is the size of embedding for each relation, and C is the maximum
        number of chains in the DAG.

        Args:
            X: Input batch of sequences of size B x N x Din.
            M: Mask over the input batch of sequences B x N.
            Ei: One-hot mask which indicates which chains have an incoming edge
                at each timestep. Size B x N x C. Each element is 0/1.
            Eo: One-hot mask which indicates which chains have an outgoing edge
                at each timestep. Size B x N x C. Each element is 0/1.
            Ri: Index of the relations for the incoming edges in Ei. Goes from
                0 to num_relations - 1. For positions where Ei=0, this can be
                any value.
            Ro: Index of the relations for the outgoing edges in Eo. Goes from
                0 to num_relations - 1. For positions where Eo=0, this can be
                any value.
            init: Hidden state to initialize from.
            mem_init: Memory state along each chain to initialize from.

        Returns:
            outs: B x N x Dout Tensor of output states at each timestep.
            mems: B x C x Drel Tensor of hidden state along each chain.
            aggs: B x N x num_relation Tensor of attention score over relations
                at each timestep.

        As an example with only coreference, suppose the input sequence is
        "Mary loves her cat", where "Mary" and "her" belong to one coreference
        chain and "cat" belongs to another coreference chain. In this case, we
        have two relations, one for the sequental relationship between each pair
        of adjacent words and one for coreference. We have 3 chains in the input
        one for the sequential relationship and two for coreference. In this
        case (assuming batch size = 1):

        ```
        Ei = [[[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1]]]  # 1 x 4 x 3
        Ri = [[[0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]]  # 1 x 4 x 3
        Eo = [[[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1]]]  # 1 x 4 x 3
        Ro = [[[0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]]  # 1 x 4 x 3
        ```
        """
        # reshape for scan
        Xre = X.permute(1, 0, 2)
        Mre = M.permute(1, 0)
        Eire = Ei.permute(1, 0, 2)
        Eore = Eo.permute(1, 0, 2)
        Rire = Ri.permute(1, 0, 2)
        Rore = Ro.permute(1, 0, 2)

        if self.reverse:
            Xre = torch.flip(Xre, [0])
            Mre = torch.flip(Mre, [0])
            Eire = torch.flip(Eire, [0])
            Eore = torch.flip(Eore, [0])
            Rire = torch.flip(Rire, [0])
            Rore = torch.flip(Rore, [0])

        # precompute input
        Xpre = torch.tensordot(Xre, self.Wstacked, [[2],[0]]) # N x B x 3Dout

        # update
        if init is None: init = torch.zeros((X.shape[0], self.output_dim), 
                dtype=torch.float32)
        if mem_init is None:
            mem_init = self.mem_init.unsqueeze(0).repeat(X.shape[0], 1, 1)
        agg_init = torch.zeros((X.shape[0], self.num_relations),
                dtype = torch.float32)
        hnew, mnew, agg = self._step((mem_init, agg_init), (Xre[0], Xpre[0], Mre[0], Eire[0], Eore[0], Rire[0], Rore[0]))
        outs = hnew.unsqueeze(0)
        mems = mnew.unsqueeze(0)
        aggs = agg.unsqueeze(0)
        for i in range(1, Xre.shape[0]):
            hnow, mnow, anow = self._step((mnew, agg), (Xre[i], Xpre[i], Mre[i], Eire[i], Eore[i], Rire[i], Rore[i]))
            outs = torch.cat((outs, hnow.unsqueeze(0)), 0)
            mems = torch.cat((mems, hnow.unsqueeze(0)), 0)
            aggs = torch.cat((aggs, hnow.unsqueeze(0)), 0)
            mnew = mnow
            agg = anow

        # outs, mems, aggs = tf.scan(self._step, (Xre, Xpre, Mre, Eire, Eore, Rire, Rore), 
        #         initializer=(init,mem_init,agg_init)) # N x B x Dout

        if self.reverse:
            outs = torch.flip(outs, [0])
            mems = torch.flip(mems, [0])
            aggs = torch.flip(aggs, [0])

        return (outs.permute(1, 0, 2), mems.permute(1, 0, 2, 3), 
                aggs.permute(1, 0, 2))

    def _attention(self, x, c_r, e, r):
        EPS = 1e-100
        v = torch.tensordot(r, self.Watt, [[2],[0]]) # B x C x Din
        actvs = torch.squeeze(torch.mm(v,x.unsqueeze(2)),axis=2) # B x C
        alphas_m = torch.exp(actvs)*e + EPS # B x C
        return alphas_m/torch.sum(alphas_m, 1, keepdim=True) # B x C

    def _hid_prev(self, x, c_r, e, r):
        if not self.concat:
            alphas = self._attention(x, c_r, e, r) # B x C
            agg = torch.unsqueeze(alphas, 2)*r
            agg = agg.permute(0, 2, 1)
        else:
            agg = torch.unsqueeze(alphas, 2)*r
            agg = agg.permute(0, 2, 1)/torch.unsqueeze(torch.sum(e, 1, keepdim=True), 1) # B x R x C
        mem = torch.mm(agg, c_r) # B x R x Dr
        return torch.reshape(mem, [-1, self.num_relations*self.rdims]), \
                torch.sum(agg, 2) # B x RDr

    def _step(self, prev, inps):
        hprev, mprev = prev[0], prev[1] # hprev : B x Dout, mprev : B x C x Dr
        x, xp, m, ei, eo, ri, ro = inps[0], inps[1], inps[2], inps[3], inps[4], \
                inps[5], inps[6] # x : B x Din, m : B, ei/o : B x C, ri/o : B x C

        hnew, agg = self._gru_cell(x, xp, mprev, ei, ri, self.resetgate, self.updategate,
                self.hiddengate) # B x Dout, B x R x C
        hnew_r = torch.reshape(hnew, 
                [x.shape[0], self.num_relations, self.rdims]) # B x R x Dr
        ro_re = ro.unsqueeze(2)
        B, N = ro.shape[0], ro.shape[1]
        ro1hot = torch.zeros(B, N, self.num_relations).scatter_(2, ro_re.data, 1)  # B x C x R
        # ro1hot = tf.one_hot(ro, self.num_relations, axis=2) # B x C x R
        mnew = torch.mm(ro1hot, hnew_r) # B x C x Dr
        hnew.set_shape([None,self.output_dim])

        m_r = torch.unsqueeze(m, 1) # B x 1
        hnew = (1.-m_r)*hprev + m_r*hnew

        eo_r = torch.unsqueeze(m_r*eo, 2) # B x C x 1
        mnew = (1.-eo_r)*mprev + eo_r*mnew

        return hnew, mnew, agg

    def _gru_cell(self, x, xp, c, e, ri, rgate, ugate, hgate):
        def _slice(a, n):
            s = a[:,n*self.output_dim:(n+1)*self.output_dim]
            return s
        ri_re = ri.unsqueeze(2)
        B, N = ri.shape[0], ri.shape[1]
        print(ri.shape)

        r1hot = torch.zeros(B, N, self.num_relations).scatter_(2, ri_re.data, 1)  # B x C x R
        # r1hot = tf.one_hot(ri, self.num_relations) # B x C x R
        prev, agg = self._hid_prev(x, c, e, r1hot) # B x RDr
        hid_to_hid = torch.mm(prev, self.Ustacked) # B x 3Dout
        r = torch.sigmoid(_slice(xp,0) + _slice(hid_to_hid,0) + rgate["b"])
        z = torch.sigmoid(_slice(xp,1) + _slice(hid_to_hid,1) + ugate["b"])
        ht = torch.tanh(_slice(xp,2) + r*_slice(hid_to_hid,2) + hgate["b"])
        h = (1.-z)*prev + z*ht
        return h, agg
