"""
Common modules
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import LSTM, Embedding
import math
import torch.nn.init as init


from . import initializer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            initializer.default_weight_init(linear.weight)
            initializer.default_bias_init(linear.bias)
            modules.append(linear)

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

        self.net = torch.nn.Sequential(*modules)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class Linear(torch.nn.Module):
    def __init__(self, in_dimension, out_dimension, bias):
        super(Linear, self).__init__()
        self.net = torch.nn.Linear(in_dimension, out_dimension, bias)
        initializer.default_weight_init(self.net.weight)
        if bias:
            initializer.default_weight_init(self.net.bias)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, num_skills, out_feature, embed_size, in_dimension, rank):
        super(Generator, self).__init__()
        self.out_feature = out_feature
        self.num_skills = num_skills
        self.embed_size = embed_size
        self.in_dimension = in_dimension
        self.rank = rank
        self._id_encoder = Embedding(num_skills, self.embed_size)
        self._mlp_trans = StackedDense(
            embed_size,
            [in_dimension] * 2,
            ([torch.nn.Tanh] * (1)) + [None]
        )
        self.SAA_question = SAA(in_dimension, 2)
        self.SAA_response = SAA(in_dimension, 2)
        self.SFE = torch.nn.GRU(
            in_dimension,
            in_dimension,
            batch_first=True
        )
        
        initializer.default_weight_init(self.SFE.weight_hh_l0)
        initializer.default_weight_init(self.SFE.weight_ih_l0)
        initializer.default_bias_init(self.SFE.bias_ih_l0)
        initializer.default_bias_init(self.SFE.bias_hh_l0)
        if rank == 0:
            self.w1 = Parameter(torch.randn(in_dimension, in_dimension*out_feature))
            init.kaiming_normal_(self.w1, mode='fan_in', nonlinearity='leaky_relu')
        else:
            self.B = Parameter(torch.randn(in_dimension, self.rank))
            self.C = Parameter(torch.randn(self.rank, in_dimension*out_feature))
        self.b1 = Parameter(torch.randn(in_dimension*out_feature))
        self.w2 = Parameter(torch.randn(in_dimension, out_feature))
        self.b2 = Parameter(torch.randn(out_feature))
        init.normal_(self.b1, mean=0, std=0.01)
        init.kaiming_normal_(self.w2, mode='fan_in', nonlinearity='leaky_relu')
        init.normal_(self.b2, mean=0, std=0.01)
    def forward(self, x, r, s, attention_reweight):
        '''
        x: [batch_size, seq_len, hidden_size] torch.float32
        r: [batch_size, seq_len]
        s: [batch_size, seq_len]
        '''
        s = self._id_encoder(s)
        r = self._id_encoder(r)
        s = self._mlp_trans(s)
        r = self._mlp_trans(r)
        s, _ = self.SFE(s)
        r, _ = self.SFE(r)
        s = self.SAA_question(s, attention_reweight=attention_reweight)
        r = self.SAA_response(r, True, attention_reweight=attention_reweight)
        s = s + r

        batch_size = x.size()[0]
        seq_len = x.size()[1]
        in_dimension = x.size()[-1]
        x = x.reshape(batch_size * seq_len, -1)
        s = s.reshape(batch_size* seq_len, -1)
        if self.rank == 0:
            weight = torch.matmul(s, self.w1) + self.b1
        else:
            weight = s @ self.B @ self.C + self.b1
        bias = torch.matmul(s, self.w2) + self.b2

        x = torch.bmm(x.unsqueeze(1), weight.view(batch_size * seq_len, in_dimension, -1)).squeeze(1) + bias
        x = x.view(batch_size, seq_len, -1)

        return x


class SAA(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SAA, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)


    def forward(self, input, mask=False, attention_reweight=None):
        batch_size = input.size(0)
        seq_len = input.size(1)
        # Linear transformations
        query = self.query_linear(input)
        key = self.key_linear(input)
        value = self.value_linear(input)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.d_k)
        key = key.view(batch_size, seq_len, self.num_heads, self.d_k)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose dimensions for matrix multiplication
        query = query.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        key = key.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        value = value.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        if mask:
            sequence_mask = torch.triu(torch.ones((seq_len, seq_len), device=input.device, dtype=input.dtype), diagonal=1)
            scores -= sequence_mask * 1e9
        
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if attention_reweight != None:
            scores = scores.reshape(batch_size*self.num_heads*seq_len, -1)
            attention_reweight = attention_reweight.unsqueeze(1)
            attention_reweight = attention_reweight.expand(-1, self.num_heads*seq_len, -1)
            attention_reweight = attention_reweight.reshape(-1, seq_len)
            scores = attention_reweight * scores
            scores = scores.reshape(batch_size, self.num_heads, seq_len, -1)

        attention_weights = nn.Softmax(dim=-1)(scores)
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len, d_k]

        # Concatenate and reshape
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Linear transformation for final output
        attention_output = self.output_linear(attention_output)

        return attention_output