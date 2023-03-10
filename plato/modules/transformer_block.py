"""
TransformerBlock class.
"""

import torch
import torch.nn as nn

from plato.modules.feedforward import FeedForward
from plato.modules.multihead_attention import MultiheadAttention


class TransformerBlock(nn.Module):
    """
    Transformer block module.
    """

    def __init__(self, hidden_dim, num_heads, dropout, attn_dropout, ff_dropout):
        super(TransformerBlock, self).__init__()

        self.attn = MultiheadAttention(hidden_dim=hidden_dim,
                                       num_heads=num_heads,
                                       dropout=attn_dropout)
        self.attn_norm = nn.LayerNorm(normalized_shape=hidden_dim,
                                      eps=1e-12,
                                      elementwise_affine=True)
        self.ff_list = nn.ModuleList([FeedForward(hidden_dim=hidden_dim,
                                                  inner_dim=4 * hidden_dim,
                                                  dropout=ff_dropout) for _ in range(3)])
        self.ff_norm = nn.ModuleList([nn.LayerNorm(normalized_shape=hidden_dim,
                                                   eps=1e-12,
                                                   elementwise_affine=True) for _ in range(3)])
        self.dropout_layer = nn.Dropout(p=dropout)
        return

    def forward(self, inp, task_id, mask=None, cache=None):
        """
        Forward process on one transformer layer.

        @param : x
        @type : Variable(shape: [batch_size, seq_len, hidden_size])

        @param : memory
        @type : Variable(shape: [batch_size, seq_len, hidden_size])

        @param : mask

        @param : cache
        """
        attn_out = self.attn(inp, mask, cache)
        attn_out = self.dropout_layer(attn_out)
        attn_out = self.attn_norm(attn_out + inp)

        ff_out = self.ff_list[task_id](attn_out)
        ff_out = self.dropout_layer(ff_out)
        ff_out = self.ff_norm[task_id](ff_out + attn_out)

        return ff_out
