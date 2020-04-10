import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from onmt.Utils import aeq


class ComplexAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(ComplexAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.final_linear_2 = nn.Linear(model_dim, model_dim)

        self.phase = nn.Parameter(torch.Tensor(500, model_dim))
        torch.nn.init.xavier_uniform_(self.phase)

    def forward(self, key, value, query, mask=None, return_key=False, all_attn=False):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        batch, k_len, d = key.size()
        batch_, k_len_, d_ = value.size()
        aeq(batch, batch_)
        aeq(k_len, k_len_)
        aeq(d, d_)
        batch_, q_len, d_ = query.size()
        aeq(batch, batch_)
        aeq(d, d_)
        aeq(self.model_dim % 8, 0)
        if mask is not None:
            batch_, q_len_, k_len_ = mask.size()
            aeq(batch_, batch)
            aeq(k_len_, k_len)
            aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        co = torch.cos(self.phase)
        sin = torch.sin(self.phase)

        key_imag = key.mul(sin[:key.size(1)].unsqueeze(0).expand(batch, k_len, d))
        value_imag = value.mul(sin[:value.size(1)].unsqueeze(0).expand(batch, k_len, d))
        query_imag = query.mul(sin[:query.size(1)].unsqueeze(0).expand(batch_, q_len, d_))

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        key_up_real = shape(self.linear_keys(key))
        key_up_imag = shape(self.linear_keys(key_imag))
        value_up_real = shape(self.linear_values(value))
        value_up_imag = shape(self.linear_values(value_imag))
        query_up_real = shape(self.linear_query(query))
        query_up_imag = shape(self.linear_query(query_imag))

        # 2) Calculate and scale scores.
        # query_up = query_up / math.sqrt(dim_per_head)
        # scores = torch.matmul(query_up, key_up.transpose(2, 3))

        scores_real = torch.matmul(query_up_real, key_up_real.transpose(2, 3)) + \
                      torch.matmul(query_up_imag, key_up_imag.transpose(2, 3))
        scores_imag = torch.matmul(query_up_imag, key_up_real.transpose(2, 3)) - \
                      torch.matmul(query_up_real, key_up_imag.transpose(2, 3))
        scores_real = scores_real / math.sqrt(dim_per_head)
        scores_imag = scores_imag / math.sqrt(dim_per_head)

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores_real)
            scores_real = scores_real.masked_fill(Variable(mask), -1e18)
            scores_imag = scores_imag.masked_fill(Variable(mask), -1e18)
            # scores = scores.masked_fill(Variable(mask), -1e18)

        # 3) Apply attention dropout and compute context vectors.
        # attn = self.sm(scores)
        attn_real = self.sm(scores_real)
        attn_imag = self.sm(scores_imag)

        # drop_attn = self.dropout(attn)
        # context = unshape(torch.matmul(drop_attn, value_up))
        drop_attn_real = self.dropout(attn_real)
        drop_attn_imag = self.dropout(attn_imag)
        context_real = unshape(torch.matmul(drop_attn_real, value_up_real))
        context_imag = unshape(torch.matmul(drop_attn_imag, value_up_imag))

        # output = self.final_linear(context)
        output_real = self.final_linear(context_real)
        output_imag = self.final_linear(context_imag)

        batch_, q_len_, d_ = output_real.size()

        if return_key:
            # key_context = unshape(torch.matmul(drop_attn, key_up))
            # key_context = self.final_linear_2(key_context)
            # output = (output, key_context)
            key_context_real = unshape(torch.matmul(drop_attn_real, key_up_real))
            key_context_real = self.final_linear_2(key_context_real)
            key_context_imag = unshape(torch.matmul(drop_attn_imag, key_up_imag))
            key_context_imag = self.final_linear_2(key_context_imag)
            output_real = (output_real, key_context_real)
            output_imag = (output_imag, key_context_imag)
        # CHECK

        aeq(q_len, q_len_)
        aeq(batch, batch_)
        aeq(d, d_)

        output_real = torch.pow(output_real, 2)
        output_imag = torch.pow(output_imag, 2)
        output = output_real + output_imag
        output = torch.pow(output, 0.5)

        if all_attn:
            # top_attn = attn \
            #             #     .view(batch_size, head_count,
            #             #           query_len, key_len)
            top_attn_real = attn_real.view(batch_size, head_count, query_len, key_len)
            top_attn_imag = attn_imag.view(batch_size, head_count, query_len, key_len)
        else:
            # Return one attn
            # top_attn = attn \
            #                .view(batch_size, head_count,
            #                      query_len, key_len)[:, 0, :, :] \
            #     .contiguous()
            top_attn_real = attn_real.view(batch_size, head_count, query_len, key_len)[:, 0, :, :].contiguous()
            top_attn_imag = attn_imag.view(batch_size, head_count, query_len, key_len)[:, 0, :, :].contiguous()
        # END CHECK
        top_attn_real = torch.pow(top_attn_real, 2)
        top_attn_imag = torch.pow(top_attn_imag, 2)
        top_attn = top_attn_real + top_attn_imag
        top_attn = torch.pow(top_attn, 0.5)
        return output, top_attn
