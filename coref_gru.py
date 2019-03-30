import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def root_six(d1,d2):
    return np.sqrt(6./(d1+d2))

class CorefGRU(nn.Module):
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
        self.W = (torch.randn(self.input_dim, self.output_dim)*root_six(self.input_dim,self.output_dim)).to(device)
        self.U = (torch.randn(h_to_h, self.output_dim)*root_six(h_to_h,self.output_dim)).to(device)
        self.b = (torch.zeros(self.output_dim, dtype=torch.float32)).to(device)
        self.resetgate = {"W": self.W, "U": self.U, "b": self.b}
        self.updategate = {"W": self.W, "U": self.U, "b": self.b}
        self.hiddengate = {"W": self.W, "U": self.U, "b": self.b}
        self.Wstacked = torch.cat([self.resetgate["W"], self.updategate["W"], self.hiddengate["W"]], 1).to(device)
        self.Ustacked = torch.cat([self.resetgate["U"], self.updategate["U"], self.hiddengate["U"]], 1).to(device)

        if not self.concat:
            self.Watt = (torch.randn(self.num_relations, self.input_dim)*0.1).to(device)
        self.mem_init = torch.zeros((self.max_chains, self.rdims), dtype=torch.float32).to(device)

    def forward(self, X, M, Ei, Eo, Ri, Ro, init=None, mem_init=None):
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

        Xpre = torch.tensordot(Xre, self.Wstacked, [[2],[0]])

        if init is None: 
            init = torch.zeros((X.shape[0], self.output_dim), dtype=torch.float32).to(device)
        if mem_init is None:
            mem_init = self.mem_init.unsqueeze(0).repeat(X.shape[0], 1, 1).to(device)
        agg_init = torch.zeros((X.shape[0], self.num_relations), dtype = torch.float32).to(device)
        
        hnew, mnew, agg = self._step((init, mem_init), (Xre[0], Xpre[0], Mre[0], Eire[0], Eore[0], Rire[0], Rore[0]))
        outs = hnew.unsqueeze(0)
        mems = mnew.unsqueeze(0)
        aggs = agg.unsqueeze(0)
        for i in range(1, Xre.shape[0]):
            hnow, mnow, anow = self._step((hnew, mnew), (Xre[i], Xpre[i], Mre[i], Eire[i], Eore[i], Rire[i], Rore[i]))
            outs = torch.cat((outs, hnow.unsqueeze(0)), 0)
            mems = torch.cat((mems, mnow.unsqueeze(0)), 0)
            aggs = torch.cat((aggs, anow.unsqueeze(0)), 0)
            mnew = mnow
            agg = anow
            hnew = hnow

        if self.reverse:
            outs = torch.flip(outs, [0])
            mems = torch.flip(mems, [0])
            aggs = torch.flip(aggs, [0])

        return (outs.permute(1, 0, 2), mems.permute(1, 0, 2, 3), aggs.permute(1, 0, 2))

    def _attention(self, x, c_r, e, r):
        EPS = 1e-100
        v = torch.tensordot(r, self.Watt, [[2],[0]])
        actvs = torch.squeeze(torch.matmul(v,x.unsqueeze(2)),2)
        e = e.type(torch.FloatTensor).to(device)
        alphas_m = torch.exp(actvs)*e + EPS
        return alphas_m/torch.sum(alphas_m, 1, keepdim=True)

    def _hid_prev(self, x, c_r, e, r):
        if not self.concat:
            alphas = self._attention(x, c_r, e, r)
            agg = torch.unsqueeze(alphas, 2)*r
            agg = agg.permute(0, 2, 1)
        else:
            agg = torch.unsqueeze(alphas, 2)*r
            agg = agg.permute(0, 2, 1)/torch.unsqueeze(torch.sum(e, 1, keepdim=True), 1) 
        mem = torch.matmul(agg, c_r) 
        return torch.reshape(mem, [-1, self.num_relations*self.rdims]), torch.sum(agg, 2)

    def _step(self, prev, inps):
        hprev, mprev = prev[0], prev[1]
        x, xp, m, ei, eo, ri, ro = inps[0], inps[1], inps[2], inps[3], inps[4], inps[5], inps[6]
        hnew, agg = self._gru_cell(x, xp, mprev, ei, ri, self.resetgate, self.updategate, self.hiddengate)
        hnew_r = torch.reshape(hnew, [x.shape[0], self.num_relations, self.rdims])
        ro_re = ro.unsqueeze(2)
        B, N = ro.shape[0], ro.shape[1]
        ro1hot_tmp = torch.zeros(B, N, self.num_relations).to(device)
        ro1hot = ro1hot_tmp.scatter_(2, ro_re.data, 1)
        mnew = torch.matmul(ro1hot, hnew_r)
        hnew = torch.reshape(hnew, [-1, self.output_dim])
        m_r = torch.unsqueeze(m, 1)
        m_r = m_r.type(torch.FloatTensor).to(device)
        hnew = (1.-m_r)*hprev + m_r*hnew
        eo = eo.type(torch.FloatTensor).to(device)
        eo_r = torch.unsqueeze(m_r*eo, 2)
        mnew = (1.-eo_r)*mprev + eo_r*mnew
        return hnew, mnew, agg

    def _gru_cell(self, x, xp, c, e, ri, rgate, ugate, hgate):
        def _slice(a, n):
            s = a[:,n*self.output_dim:(n+1)*self.output_dim]
            return s
        ri_re = ri.unsqueeze(2)
        B, N = ri.shape[0], ri.shape[1]
        r1hot_tmp = torch.zeros(B, N, self.num_relations).to(device)
        r1hot = r1hot_tmp.scatter_(2, ri_re.data, 1) 
        prev, agg = self._hid_prev(x, c, e, r1hot)
        hid_to_hid = torch.matmul(prev, self.Ustacked)
        r = torch.sigmoid(_slice(xp,0) + _slice(hid_to_hid,0) + rgate["b"])
        z = torch.sigmoid(_slice(xp,1) + _slice(hid_to_hid,1) + ugate["b"])
        ht = torch.tanh(_slice(xp,2) + r*_slice(hid_to_hid,2) + hgate["b"])
        h = (1.-z)*prev + z*ht
        return h, agg
