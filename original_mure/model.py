import numpy as np
import torch
from utils import *


class MuRP(torch.nn.Module):
    def __init__(self, d, dim, entity_mat=None):
        super(MuRP, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        if entity_mat is not None:
            entity_mat = entity_mat.double()
            self.Eh.weight.data = self.Eh.weight.data.double()
            self.Eh.weight.data = 1e-3 * entity_mat
            self.Eh.to(device)
        else:
            self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device=device))

        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device=device))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device=device))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        Ru = self.Wu[r_idx]
        rvh = self.rvh.weight[r_idx]


        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1, 
                        u/(torch.norm(u, 2, dim=-1, keepdim=True)-1e-5), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= 1, 
                        v/(torch.norm(v, 2, dim=-1, keepdim=True)-1e-5), v)
        rvh = torch.where(torch.norm(rvh, 2, dim=-1, keepdim=True) >= 1, 
                          rvh/(torch.norm(rvh, 2, dim=-1, keepdim=True)-1e-5), rvh)   
        u_e = p_log_map(u)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        v_m = p_sum(v, rvh)
        u_m = torch.where(torch.norm(u_m, 2, dim=-1, keepdim=True) >= 1, 
                          u_m/(torch.norm(u_m, 2, dim=-1, keepdim=True)-1e-5), u_m)
        v_m = torch.where(torch.norm(v_m, 2, dim=-1, keepdim=True) >= 1, 
                          v_m/(torch.norm(v_m, 2, dim=-1, keepdim=True)-1e-5), v_m)
        
        sqdist = (2.*artanh(torch.clamp(torch.norm(p_sum(-u_m, v_m), 2, dim=-1), 1e-10, 1-1e-5)))**2

        return -sqdist + self.bs[u_idx] + self.bo[v_idx] 



class MuRE(torch.nn.Module):
    def __init__(self, d, dim, entity_mat=None, rel_vec=None, rel_mat=None):
        super(MuRE, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.E = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        if entity_mat is not None:
            entity_mat = entity_mat.double()
            self.E.weight.data = self.E.weight.data.double()
            self.E.weight.data = 1e-3 * entity_mat
            self.E.to(device)
        else:
            self.E.weight.data = self.E.weight.data.double()
            self.E.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device=device))
        
        if rel_mat is not None:
            self.Wu = torch.nn.Parameter(rel_mat.repeat(2,1))
            self.Wu.requires_grad = True
            self.Wu.to(device)
        else:
            self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                            dim)), dtype=torch.double, requires_grad=True, device=device))

        self.rv = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        if rel_vec is not None:
            self.rv.weight.data = rel_vec.repeat(2,1)
            self.rv.to(device)
        else:
            self.rv.weight.data = self.rv.weight.data.double()
            self.rv.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device=device))

        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device))
        self.loss = torch.nn.BCEWithLogitsLoss()
       
    def forward(self, u_idx, r_idx, v_idx):
        u = self.E.weight[u_idx]
        v = self.E.weight[v_idx]
        Ru = self.Wu[r_idx]
        rv = self.rv.weight[r_idx]
        
        u_W = u * Ru

        sqdist = torch.sum(torch.pow(u_W - (v+rv), 2), dim=-1)
        return -sqdist + self.bs[u_idx] + self.bo[v_idx] 


class MuRE_TransE(torch.nn.Module):
    def __init__(self, d, dim, entity_mat=None, rel_vec=None):
        super(MuRE_TransE, self).__init__()
        print("Initializing Mure_TransE model...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.E = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        if entity_mat is not None:
            self.E.weight.data = self.E.weight.data.double()
            self.E.weight.data = entity_mat.double()
            self.E.to(device)
        else:
            self.E.weight.data = self.E.weight.data.double()
            self.E.weight.data = (torch.randn((len(d.entities), dim), dtype=torch.double, device=device))
        
        self.rv = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        if rel_vec is not None:
            self.rv.weight.data = rel_vec.repeat(2,1)
            self.rv.to(device)
        else:
            self.rv.weight.data = self.rv.weight.data.double()
            self.rv.weight.data = (torch.randn((len(d.relations), dim), dtype=torch.double, device=device))

        self.loss = torch.nn.BCEWithLogitsLoss()
       
    def forward(self, u_idx, r_idx, v_idx):
        u = self.E.weight[u_idx]
        v = self.E.weight[v_idx]
        rv = self.rv.weight[r_idx]
        
        sqdist = torch.sum(torch.pow(u - (v+rv), 2), dim=-1)
        return -sqdist