"""
Embedder class.
"""

import torch
import torch.nn as nn


class TextEmbedder(nn.Module):
    """
    Composite embedding layer.
    """

    def __init__(self,
                 hidden_dim,
                 num_token_embeddings,
                 num_pos_embeddings,
                 num_turn_embeddings,
                 dropout=0.1,
                 pos_trainable=False):
        super(TextEmbedder, self).__init__()

        self.token_embedding = nn.Embedding(num_token_embeddings, hidden_dim)
        self.pos_embedding = nn.Embedding(num_pos_embeddings, hidden_dim)
        self.pos_embedding.weight.requires_grad = pos_trainable
        self.turn_embedding = nn.Embedding(num_turn_embeddings, hidden_dim)
        self.text_embedding = nn.Parameter(torch.Tensor(hidden_dim))
        self.dropout_layer = nn.Dropout(p=dropout)

        # follow the default xavier_uniform initializer in paddle version
        # otherwise, there are bugs for dec_probs computation in weight typing setting
        # default norm initializer in nn.Embedding in pytorch, which samples larger values
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.turn_embedding.weight)
        nn.init.normal_(self.text_embedding, std=0.02)
        return

    def forward(self, token_inp, pos_inp, turn_inp, use_turn=True):
        embed = self.token_embedding(token_inp) + \
                self.pos_embedding(pos_inp) + \
                self.text_embedding

        if use_turn:
            embed += self.turn_embedding(turn_inp)

        embed = self.dropout_layer(embed)
        return embed


class ImageEmbedder(nn.Module):
    """
    图像embedder
    """

    def __init__(self,
                 hidden_dim,
                 num_pos_embeddings,
                 img_patch_size=16,
                 img_in_channel=3,
                 dropout=0.1,
                 pos_trainable=False):
        super(ImageEmbedder, self).__init__()

        self.hidden_dim = hidden_dim

        self.patch_embedding = nn.Conv2d(in_channels=img_in_channel,
                                         out_channels=hidden_dim,
                                         kernel_size=img_patch_size,
                                         stride=img_patch_size)
        self.pos_embedding = nn.Embedding(num_pos_embeddings, hidden_dim)
        self.pos_embedding.weight.requires_grad = pos_trainable
        self.img_embedding = nn.Parameter(torch.Tensor(hidden_dim))
        self.dropout_layer = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.patch_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        nn.init.normal_(self.img_embedding, std=0.02)

    def forward(self, img, pos_inp):
        embed = self.patch_embedding(img).view(img.shape[0], self.hidden_dim, -1).permute(0, 2, 1) + \
                self.pos_embedding(pos_inp) + \
                self.img_embedding
        embed = self.dropout_layer(embed)
        return embed
