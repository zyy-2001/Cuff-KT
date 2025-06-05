import os

import numpy as np
import torch
from torch import nn

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import one_hot
from . import common

class DKVMN(Module):
    def __init__(self, convert, num_skills, method, rank, dim_s, size_m, dropout=0.2):
        super().__init__()
        self.convert = convert
        self.method = method
        self.rank = rank
        self.num_skills = num_skills
        self.dim_s = dim_s
        self.size_m = size_m

        self.k_emb_layer = Embedding(self.num_skills, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_skills * 2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        if self.convert:
            self.p_layer = Linear(self.dim_s, self.dim_s)
            self.trans = Linear(self.dim_s, self.num_skills)
        else:
            self.p_layer = Linear(self.dim_s, 1)

        if self.method == 'cuff' or self.method == 'cuff+':
            self.generator = common.Generator(self.num_skills, self.dim_s, self.dim_s, self.dim_s, self.rank)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)
        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(self, feed_dict):
        q = feed_dict['skills']
        r = feed_dict['responses']
        masked_r = r * (r > -1).long()

        if self.convert:
            cshft = q[:, 1:]
            q_input = q[:, :-1]
            r_input = masked_r[:, :-1]
        else:
            q_input = q
            r_input = masked_r

        if self.method == 'cuff' or self.method == 'cuff+':
            attention_reweight = feed_dict['attention_reweight'][:, :-1]
        
        batch_size = q.shape[0]
        x = q_input + self.num_skills * r_input
        k = self.k_emb_layer(q_input)
        v = self.v_emb_layer(x)
        
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )

        p = self.p_layer(self.dropout_layer(f))
        if self.method in ["cuff", "cuff+"]:
            p = self.generator(p, r_input, q_input, attention_reweight)
        
        if self.convert:
            p = self.trans(p)
                
        
        state = None
        p = torch.sigmoid(p)
        if self.convert:
            state = p
            p = (p * one_hot(cshft.long(), self.num_skills)).sum(-1)
            true = r[:, 1:].float()
        else:
            p = p.squeeze(-1)
            p = p[:, 1:]
            true = r[:, 1:].float()
        out_dict = {
            "pred": p,
            "true": true,
            "state": state
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss, len(pred[mask]), true[mask].sum().item()