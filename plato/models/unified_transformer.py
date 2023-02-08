"""
UnifiedTransformer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plato.modules.functions as F_alias

from plato.args import str2bool
from plato.modules.embedder import TextEmbedder, ImageEmbedder
from plato.models.model_base import ModelBase
from plato.modules.transformer_block import TransformerBlock


class UnifiedTransformer(ModelBase):
    """
    Implement unified transformer.
    """

    @classmethod
    def add_cmdline_argument(cls, group):
        """ Add cmdline argument. """
        group.add_argument("--img_in_channel", type=int, default=3,
                           help="The image channel.")
        group.add_argument("--img_patch_size", type=int, default=16,
                           help="The patch size of the image embedding.")
        group.add_argument("--num_enc_layers", type=int, default=4,
                           help="The number of encode layers during multimodal mode.")

        group.add_argument("--num_token_embeddings", type=int, default=-1,
                           help="The number of tokens in vocabulary. "
                                "It will be automatically calculated after loading vocabulary.")
        group.add_argument("--num_pos_embeddings", type=int, default=512,
                           help="The maximum number of position.")
        # group.add_argument("--num_type_embeddings", type=int, default=2,
        #                    help="The number of different type of tokens.")
        group.add_argument("--num_turn_embeddings", type=int, default=16,
                           help="The maximum number of turn.")
        group.add_argument("--num_latent", type=int, default=5,
                           help="The number of latent.")

        group.add_argument("--tau", type=float, default=0.67,
                           help="The parameter of gumbel softmax.")
        group.add_argument("--with_bow", type=str2bool, default=True,
                           help="Whether to use BoW loss.")
        group.add_argument("--hidden_dim", type=int, default=768,
                           help="The size of hidden vector in transformer.")
        group.add_argument("--num_heads", type=int, default=12,
                           help="The number of heads in multi head attention.")
        group.add_argument("--num_layers", type=int, default=12,
                           help="The number of layers in transformer.")
        group.add_argument("--padding_idx", type=int, default=0,
                           help="The padding index.")
        group.add_argument("--dropout", type=float, default=0.1,
                           help="The dropout ratio after multi head attention and feed forward network.")
        group.add_argument("--embed_dropout", type=float, default=0.0,
                           help="The dropout ratio of embedding layers.")
        group.add_argument("--attn_dropout", type=float, default=0.1,
                           help="The dropout ratio of multi head attention.")
        group.add_argument("--ff_dropout", type=float, default=0.1,
                           help="The dropout ratio of feed forward network.")
        group.add_argument("--use_discriminator", type=str2bool, default=False,
                           help="Whether to use discriminator loss.")

        group.add_argument("--bow_ratio", type=float, default=1.0)
        group.add_argument("--dis_ratio", type=float, default=1.0,
                           help="The ratio of discriminator loss.")

        group.add_argument("--weight_sharing", type=str2bool, default=True,
                           help="Whether to share weight between token embedding and "
                                "predictor FC layer.")
        group.add_argument("--pos_trainable", type=str2bool, default=True,
                           help="Whether to train position embeddings.")
        group.add_argument("--two_layer_predictor", type=str2bool, default=False,
                           help="Use two layer predictor. "
                                "Traditional BERT use two FC layers to predict masked token.")
        # group.add_argument("--bidirectional_context", type=str2bool, default=True,
        #                    help="Whether to use bidirectional self-attention in context tokens.")
        # group.add_argument("--label_smooth", type=float, default=0.0,
        #                    help="Use soft label to calculate NLL loss and BoW loss.")
        group.add_argument("--initializer_range", type=float, default=0.02,
                           help="Use to initialize parameters.")

        group.add_argument("--lr", type=float, default=5e-5,
                           help="The inital learning rate for Adam.")
        group.add_argument("--weight_decay", type=float, default=0.0,
                           help="The weight decay for Adam.")
        group.add_argument("--max_grad_norm", type=float, default=None,
                           help="The maximum norm of gradient.")
        return group

    def __init__(self, hparams, generator, dtype="float32"):
        super(UnifiedTransformer, self).__init__(hparams)
        self.generator = generator
        self.img_in_channel = hparams.img_in_channel
        self.img_patch_size = hparams.img_patch_size
        self.num_enc_layers = hparams.num_enc_layers
        self.num_token_embeddings = hparams.num_token_embeddings
        self.num_pos_embeddings = hparams.num_pos_embeddings
        # self.num_type_embeddings = hparams.num_type_embeddings
        self.num_turn_embeddings = hparams.num_turn_embeddings
        self.num_latent = hparams.num_latent
        self.tau = hparams.tau
        self.with_bow = hparams.with_bow
        self.hidden_dim = hparams.hidden_dim
        self.num_heads = hparams.num_heads
        self.num_layers = hparams.num_layers
        self.padding_idx = hparams.padding_idx
        self.dropout = hparams.dropout
        self.embed_dropout = hparams.embed_dropout
        self.attn_dropout = hparams.attn_dropout
        self.ff_dropout = hparams.ff_dropout
        self.use_discriminator = hparams.use_discriminator
        self.weight_sharing = hparams.weight_sharing
        self.pos_trainable = hparams.pos_trainable
        self.two_layer_predictor = hparams.two_layer_predictor
        # self.bidirectional_context = hparams.bidirectional_context
        self.initializer_range = hparams.initializer_range
        self._dtype = dtype

        self.bow_ratio = hparams.bow_ratio

        self.text_embedder = TextEmbedder(self.hidden_dim,
                                          self.num_token_embeddings,
                                          self.num_pos_embeddings,
                                          self.num_turn_embeddings,
                                          dropout=self.embed_dropout,
                                          pos_trainable=self.pos_trainable)
        self.img_embedder = ImageEmbedder(self.hidden_dim,
                                          self.num_pos_embeddings,
                                          img_patch_size=self.img_patch_size,
                                          img_in_channel=self.img_in_channel,
                                          dropout=self.embed_dropout,
                                          pos_trainable=self.pos_trainable)
        self.embed_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_dim,
                                             eps=1e-12,
                                             elementwise_affine=True)

        self.layers = nn.ModuleList([TransformerBlock(self.hidden_dim,
                                                      self.num_heads,
                                                      self.dropout,
                                                      self.attn_dropout,
                                                      self.ff_dropout) for _ in range(hparams.num_layers)])

        self.encoder = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.encoder_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_dim,
                                               eps=1e-12,
                                               elementwise_affine=True)

        if self.num_latent > 0:
            self.post_network = nn.Linear(self.hidden_dim, self.num_latent, bias=False)
            # Todo: discriminator??
            if self.use_discriminator:
                self.dis_ratio = hparams.dis_ratio
                self.discriminator = nn.Sequential(
                    nn.Linear(self.hidden_dim, 1),
                    nn.Sigmoid()
                )

        if self.two_layer_predictor:
            self.pre_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU()
            )
            if self.num_latent > 0 and self.with_bow:
                self.pre_bow_predictor = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU()
                )
        if not self.weight_sharing:
            self.predictor = nn.Linear(self.hidden_dim, self.num_token_embeddings, bias=False)
        if self.num_latent > 0 and self.with_bow:
            self.bow_predictor = nn.Linear(self.hidden_dim, self.num_token_embeddings, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self._create_parameters()

        self.nll_loss = nn.NLLLoss(ignore_index=self.padding_idx, reduction='none')
        self.match_nll_loss = nn.NLLLoss(reduction='none')

        self.max_grad_norm = hparams.max_grad_norm
        if self.max_grad_norm is not None:
            self.grad_clip = self.max_grad_norm
        else:
            self.grad_clip = None
        self.weight_decay = hparams.weight_decay
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=hparams.lr, weight_decay=self.weight_decay)

        if self.use_gpu:
            self.cuda()

        return

    def _create_parameters(self):
        """ Create model's paramters. """
        self.txt_cls_embed = nn.Parameter(torch.Tensor(1, 1, self.hidden_dim))
        self.img_cls_embed = nn.Parameter(torch.Tensor(1, 1, self.hidden_dim))
        nn.init.normal_(self.txt_cls_embed, std=self.initializer_range)
        nn.init.normal_(self.txt_cls_embed, std=self.initializer_range)

        if self.num_latent > 0:
            self.mask_embed = nn.Parameter(torch.Tensor(1, 1, self.hidden_dim))
            self.latent_embeddings = nn.Parameter(torch.Tensor(self.num_latent, self.hidden_dim))
            nn.init.normal_(self.mask_embed, std=self.initializer_range)
            nn.init.normal_(self.latent_embeddings, std=self.initializer_range)

        sequence_mask = np.tri(self.num_pos_embeddings, self.num_pos_embeddings, dtype=self._dtype)
        self.sequence_mask = torch.tensor(sequence_mask)
        if self.use_gpu:
            self.sequence_mask = self.sequence_mask.cuda()
        return

    def _create_mask(self, input_mask, append_head=False, auto_regressive=False):
        """
        Create attention mask.
        创建从序列形式到矩阵形式的mask：[batch_size, max_seq_len， 1] -> [batch_size, max_seq_len, max_seq_len]
        mask除了要考虑attention mask（自回归），还需要考虑pad的mask（自回归和双向）
        注：
        1. 一个句子中的非<pad>词看整个句子，该句中只有<pad>词才被mask
        2. 一个句子中的<pad>词看整个句子，该句的所有词都应该被mask

        @param : input_mask
        @type : Variable(shape: [batch_size, max_seq_len])

        @param : auto_regressive
        @type : bool
        """
        seq_len = input_mask.shape[1]

        input_mask = input_mask.float()
        mask1 = input_mask.unsqueeze(-1).repeat(1, 1, seq_len)
        mask2 = mask1.permute(0, 2, 1)
        mask = mask1 * mask2

        if append_head:
            # 拼接上句首位置([M]/z)的mask
            mask = torch.cat([mask[:, :1, :], mask], dim=1)
            mask = torch.cat([mask[:, :, :1], mask], dim=2)
            seq_len += 1

        if auto_regressive:
            # 将tgt端的<pad> mask和自回归attention mask融合
            seq_mask = self.sequence_mask[:seq_len, :seq_len]
            mask = mask * seq_mask

        mask = 1 - mask
        return mask

    def _join_mask(self, mask1, mask2):
        """
        Merge source attention mask and target attention mask.
        合并后的整个mask矩阵可以分为四个部分：左上lu/右上ru/左下lb/右下rb

        @param : mask1 : source attention mask
        @type : Variable(shape: [batch_size, max_src_len, max_src_len])

        @param : mask1 : target attention mask
        @type : Variable(shape: [batch_size, max_tgt_len, max_tgt_len])
        """
        batch_size = mask1.shape[0]
        seq_len1 = mask1.shape[1]
        seq_len2 = mask2.shape[1]
        seq_len = seq_len1 + seq_len2

        mask_lu = mask1
        mask_ru = torch.ones(batch_size, seq_len1, seq_len2)

        if self.use_gpu:
            mask_ru = mask_ru.cuda()
        mask3 = mask2[:, :, :1].repeat(1, 1, seq_len1)
        mask4 = mask1[:, :1, :].repeat(1, seq_len2, 1)

        mask_lb = mask3 + mask4 - mask3 * mask4
        mask_rb = mask2
        mask_u = torch.cat([mask_lu, mask_ru], dim=2)
        mask_b = torch.cat([mask_lb, mask_rb], dim=2)
        mask = torch.cat([mask_u, mask_b], dim=1)
        return mask

    def _encode_network(self, input_mask, embed, data_type, num_layers=None):
        # 根据不同输入数据类型使用不同模态专家
        num_layers = self.num_enc_layers if num_layers is None else num_layers
        fnn_id = 0 if data_type == "img" else 1

        encod_embed = embed
        encod_mask = input_mask

        for idx in range(num_layers):
            encod_embed = self.layers[idx](encod_embed, fnn_id, mask=encod_mask)

        return encod_embed

    def _match_network(self, input_mask, embed, batch_size, data_type):
        """"""
        # 根据不同输入类型拼接不同mask embedding
        # 图片或文字
        # shape [1, 1, hidden_size]
        cls_embed = self.img_cls_embed if data_type == "img" else self.txt_cls_embed

        # 扩展并拼接输入embedding
        # shape [batch_size, seq_len + 1, hidden_size]
        cls_embed = cls_embed.repeat(batch_size, 1, 1)
        cls_embed = self.embed_layer_norm(cls_embed)
        encod_embed = torch.cat([cls_embed, embed], dim=1)  # 各自过embedding后再拼接

        # 创建注意力掩码矩阵
        # shape [batch_size, seq_len + 1, seq_len + 1]
        mask = self._create_mask(input_mask, auto_regressive=False, append_head=True)

        encod_embed = self._encode_network(mask, encod_embed, data_type, self.num_layers)
        encod_embed = encod_embed[:, 0]
        encod_embed = self.encoder(encod_embed)
        encod_embed = self.encoder_layer_norm(encod_embed)
        return encod_embed

    def _unim_generation_network(self, input_mask, embed):
        """"""
        dec_embed = embed

        # Create generation network mask
        dec_mask = self._create_mask(input_mask, auto_regressive=True, append_head=False)

        for layer in self.layers:
            dec_embed = layer(embed, 1, mask=dec_mask)

        if self.two_layer_predictor:
            dec_embed = self.pre_predictor(dec_embed)
        if self.weight_sharing:
            token_embedding = self.text_embedder.token_embedding.weight
            dec_logits = torch.matmul(dec_embed, token_embedding.T)
        else:
            dec_logits = self.predictor(dec_embed)

        dec_probs = self.softmax(dec_logits)
        return dec_probs

    def _posteriori_network(self, input_mask, embed, batch_size, src_len, tgt_len):
        """ Basic posteriori network implement. """
        src_embed = embed[:, :src_len]
        tgt_embed = embed[:, src_len:]
        src_mask = input_mask[:, :src_len]
        tgt_mask = input_mask[:, src_len:]

        src_mask = self._create_mask(src_mask, auto_regressive=False, append_head=False)
        tgt_mask = self._create_mask(tgt_mask, auto_regressive=False, append_head=False)
        post_mask = self._create_mask(input_mask, auto_regressive=False, append_head=self.num_latent > 0)

        src_embed = self._encode_network(src_mask, src_embed, "img")
        tgt_embed = self._encode_network(tgt_mask, tgt_embed, "txt")

        if self.num_latent > 0:
            mask_embed = self.mask_embed
            mask_embed = mask_embed.repeat(batch_size, 1, 1)
            mask_embed = self.embed_layer_norm(mask_embed)
            post_embed = torch.cat([mask_embed, src_embed, tgt_embed], dim=1)
        else:
            post_embed = torch.cat([src_embed, tgt_embed], dim=1)

        for idx in range(self.num_enc_layers, self.num_layers):
            post_embed = self.layers[idx](post_embed, 2, mask=post_mask)

        post_embed = post_embed[:, 0]
        post_logits = self.post_network(post_embed)
        post_probs = self.softmax(post_logits)
        post_logits = torch.log(post_probs)
        return post_embed, post_probs, post_logits

    def _discriminator_network(self, input_mask, embed, batch_size, src_len, tgt_len, pos_embed):
        """ Basic discriminator network implement. """
        src_embed = embed[:, :src_len]
        tgt_embed = embed[:, src_len:]
        neg_tgt_embed = torch.cat([tgt_embed[1:], tgt_embed[:1]], dim=0) if batch_size > 1 \
            else tgt_embed

        src_mask = input_mask[:, :src_len]
        tgt_mask = input_mask[:, src_len:]
        neg_tgt_mask = torch.cat([tgt_mask[1:], tgt_mask[:1]], dim=0) if batch_size > 1 \
            else tgt_mask
        neg_mask = torch.cat([src_mask, neg_tgt_mask], dim=1)

        src_mask = self._create_mask(src_mask, auto_regressive=False, append_head=False)
        neg_tgt_mask = self._create_mask(neg_tgt_mask, auto_regressive=False, append_head=False)
        neg_mask = self._create_mask(neg_mask, auto_regressive=False, append_head=self.num_latent > 0)

        src_embed = self._encode_network(src_mask, src_embed, "img")
        neg_tgt_embed = self._encode_network(neg_tgt_mask, neg_tgt_embed, "txt")

        if self.num_latent > 0:
            mask_embed = self.mask_embed
            mask_embed = mask_embed.repeat(batch_size, 1, 1)
            mask_embed = self.embed_layer_norm(mask_embed)
            neg_embed = torch.cat([mask_embed, src_embed, neg_tgt_embed], dim=1)
        else:
            neg_embed = torch.cat([src_embed, neg_tgt_embed], dim=1)

        for idx in range(self.num_enc_layers, self.num_layers):
            neg_embed = self.layers[idx](neg_embed, 2, mask=neg_mask)

        neg_embed = neg_embed[:, 0]
        pos_probs = self.discriminator(pos_embed)
        neg_probs = self.discriminator(neg_embed)

        return pos_probs, neg_probs

    def _generation_network(self, input_mask, embed, batch_size, src_len, tgt_len, latent_embed):
        """ Basic generation network implement. """
        src_embed = embed[:, :src_len]
        tgt_embed = embed[:, src_len:]

        src_mask = input_mask[:, :src_len]
        tgt_mask = input_mask[:, src_len:]
        enc_mask = self._create_mask(src_mask, auto_regressive=False, append_head=self.num_latent > 0)
        src_mask = self._create_mask(src_mask, auto_regressive=False, append_head=False)
        tgt_mask = self._create_mask(tgt_mask, auto_regressive=True, append_head=False)
        dec_mask = self._join_mask(enc_mask, tgt_mask)

        src_embed = self._encode_network(src_mask, src_embed, "img")
        tgt_embed = self._encode_network(tgt_mask, tgt_embed, "txt")

        if self.num_latent > 0:
            latent_embed = latent_embed.unsqueeze(1)
            latent_embed = self.embed_layer_norm(latent_embed)
            dec_embed = torch.cat([latent_embed, src_embed, tgt_embed], dim=1)
        else:
            dec_embed = torch.cat([src_embed, tgt_embed], dim=1)

        for idx in range(self.num_enc_layers, self.num_layers):
            dec_embed = self.layers[idx](dec_embed, 2, mask=dec_mask)

        if self.num_latent > 0:
            latent_embed = dec_embed[:, 0]
        else:
            latent_embed = None

        dec_embed = dec_embed[:, -tgt_len:]
        if self.two_layer_predictor:
            dec_embed = self.pre_predictor(dec_embed)
        if self.weight_sharing:
            token_embedding = self.text_embedder.token_embedding.weight
            dec_logits = torch.matmul(dec_embed, token_embedding.T)
        else:
            dec_logits = self.predictor(dec_embed)

        dec_probs = self.softmax(dec_logits)

        return latent_embed, dec_probs

    def _forward_image_match(self, inputs, is_training):
        outputs = {}

        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]

        tgt_token = inputs["tgt_token"]
        tgt_mask = inputs["tgt_mask"]
        tgt_pos = inputs["tgt_pos"]

        src_embed = self.img_embedder(src_token, src_pos)
        tgt_embed = self.text_embedder(tgt_token, tgt_pos)
        src_embed = self.embed_layer_norm(src_embed)
        tgt_embed = self.embed_layer_norm(tgt_embed)

        batch_size = src_token.shape[0]

        # shape [batch_size, hidden_size]
        src_encod = self._match_network(src_mask, src_embed, batch_size, data_type="img")
        tgt_encod = self._match_network(tgt_mask, tgt_embed, batch_size, data_type="txt")

        outputs["src_encod"] = src_encod
        outputs["tgt_encod"] = tgt_encod

        return outputs

    def _forward_image_caption(self, inputs, is_training):
        outputs = {}

        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]

        tgt_token = inputs["tgt_token"][:, :-1]
        tgt_mask = inputs["tgt_mask"][:, :-1]
        tgt_pos = inputs["tgt_pos"][:, :-1]

        input_mask = torch.cat([src_mask, tgt_mask], dim=1)
        src_embed = self.img_embedder(src_token, src_pos)
        tgt_embed = self.text_embedder(tgt_token, tgt_pos)
        embed = torch.cat([src_embed, tgt_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        batch_size = src_token.shape[0]
        src_len = src_embed.shape[1]
        tgt_len = tgt_token.shape[1]

        if self.num_latent > 0:
            """计算latent后验概率"""
            post_embed, post_probs, post_logits = self._posteriori_network(
                input_mask, embed, batch_size, src_len, tgt_len)
            outputs["post_logits"] = post_logits  # 使用取log的prob是因为gumbel-softmax公式中需要log计算

            if self.use_discriminator:
                pos_probs, neg_probs = self._discriminator_network(
                    input_mask, embed, batch_size, src_len, tgt_len, post_embed)
                outputs["pos_probs"] = pos_probs
                outputs["neg_probs"] = neg_probs

            if is_training:
                # 训练时，z是soft one-hot的形式(通过softmax加温度来近似)，其embedding是加权
                z = F.gumbel_softmax(logits=post_logits, tau=self.tau)
            else:
                # 非训练时(evaluate)，z是hard one-hot形式(直接argmax求出)，其embedding就是对应index的embedding(无加权)
                indices = torch.argmax(post_logits, dim=1)
                z = F.one_hot(indices, num_classes=self.num_latent).float()
            latent_embeddings = self.latent_embeddings
            latent_embed = torch.matmul(z, latent_embeddings)
            outputs["latent_embed"] = latent_embed
        else:
            latent_embed = None

        latent_embed, dec_probs = self._generation_network(
            input_mask, embed, batch_size, src_len, tgt_len, latent_embed)
        outputs["dec_probs"] = dec_probs

        if self.num_latent > 0 and self.with_bow:
            if self.two_layer_predictor:
                latent_embed = self.pre_bow_predictor(latent_embed)
            bow_logits = self.bow_predictor(latent_embed)
            bow_probs = self.softmax(bow_logits)
            outputs["bow_probs"] = bow_probs

        return outputs

    def _forward_poem_language(self, inputs, is_training):
        outputs = {}

        tgt_token = inputs["tgt_token"][:, :-1]
        tgt_mask = inputs["tgt_mask"][:, :-1]
        tgt_pos = inputs["tgt_pos"][:, :-1]
        tgt_turn = inputs["tgt_turn"][:, :-1]

        tgt_embed = self.text_embedder(tgt_token, tgt_pos, tgt_turn)
        tgt_embed = self.embed_layer_norm(tgt_embed)

        batch_size = tgt_token.shape[0]

        dec_probs = self._unim_generation_network(tgt_mask, tgt_embed)
        outputs["dec_probs"] = dec_probs

        return outputs

    def _forward_image_poem(self, inputs, is_training):
        outputs = {}

        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]

        tgt_token = inputs["tgt_token"][:, :-1]
        tgt_mask = inputs["tgt_mask"][:, :-1]
        tgt_pos = inputs["tgt_pos"][:, :-1]
        tgt_turn = inputs["tgt_turn"][:, :-1]

        input_mask = torch.cat([src_mask, tgt_mask], dim=1)
        src_embed = self.img_embedder(src_token, src_pos)
        tgt_embed = self.text_embedder(tgt_token, tgt_pos, tgt_turn)
        embed = torch.cat([src_embed, tgt_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        batch_size = src_token.shape[0]
        src_len = src_embed.shape[1]
        tgt_len = tgt_token.shape[1]

        if self.num_latent > 0:
            """计算latent后验概率"""
            post_embed, post_probs, post_logits = self._posteriori_network(
                input_mask, embed, batch_size, src_len, tgt_len)
            outputs["post_logits"] = post_logits  # 使用取log的prob是因为gumbel-softmax公式中需要log计算

            if self.use_discriminator:
                pos_probs, neg_probs = self._discriminator_network(
                    input_mask, embed, batch_size, src_len, tgt_len, post_embed)
                outputs["pos_probs"] = pos_probs
                outputs["neg_probs"] = neg_probs

            if is_training:
                # 训练时，z是soft one-hot的形式(通过softmax加温度来近似)，其embedding是加权
                z = F.gumbel_softmax(logits=post_logits, tau=self.tau)
            else:
                # 非训练时(evaluate)，z是hard one-hot形式(直接argmax求出)，其embedding就是对应index的embedding(无加权)
                indices = torch.argmax(post_logits, dim=1)
                z = F.one_hot(indices, num_classes=self.num_latent).float()
            latent_embeddings = self.latent_embeddings
            latent_embed = torch.matmul(z, latent_embeddings)
            outputs["latent_embed"] = latent_embed
        else:
            latent_embed = None

        latent_embed, dec_probs = self._generation_network(
            input_mask, embed, batch_size, src_len, tgt_len, latent_embed)
        outputs["dec_probs"] = dec_probs

        if self.num_latent > 0 and self.with_bow:
            if self.two_layer_predictor:
                latent_embed = self.pre_bow_predictor(latent_embed)
            bow_logits = self.bow_predictor(latent_embed)
            bow_probs = self.softmax(bow_logits)
            outputs["bow_probs"] = bow_probs

        return outputs

    def _forward_image_poem_match(self, inputs, is_training):
        outputs = {}

        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]

        tgt_token = inputs["tgt_token"]
        tgt_mask = inputs["tgt_mask"]
        tgt_pos = inputs["tgt_pos"]
        tgt_turn = inputs["tgt_turn"]

        src_embed = self.img_embedder(src_token, src_pos)
        tgt_embed = self.text_embedder(tgt_token, tgt_pos, tgt_turn)
        src_embed = self.embed_layer_norm(src_embed)
        tgt_embed = self.embed_layer_norm(tgt_embed)

        batch_size = src_token.shape[0]

        # shape [batch_size, hidden_size]
        src_encod = self._match_network(src_mask, src_embed, batch_size, data_type="img")
        tgt_encod = self._match_network(tgt_mask, tgt_embed, batch_size, data_type="txt")

        outputs["src_encod"] = src_encod
        outputs["tgt_encod"] = tgt_encod

        return outputs

    def _collect_metrics_image_match(self, inputs, outputs):
        """图像文本匹配任务"""
        metrics = {}

        # shape [batch_size, hidden_size]
        src_encod = outputs["src_encod"]
        tgt_encod = outputs["tgt_encod"]

        batch_size = src_encod.shape[0]

        # 计算相似度
        # shape [batch_size, 1]
        tgt_encod_neg = tgt_encod
        src_encod_neg = src_encod
        img2txt_sim_pred = torch.stack([torch.dot(src_encod[i], tgt_encod_neg[i]).unsqueeze(0)
                                        for i in range(batch_size)], dim=0)
        txt2img_sim_pred = torch.stack([torch.dot(tgt_encod[i], src_encod_neg[i]).unsqueeze(0)
                                        for i in range(batch_size)], dim=0)

        for _ in range(1, batch_size):
            tgt_encod_neg = torch.cat([tgt_encod_neg[1:], tgt_encod_neg[:1]], dim=0)
            src_encod_neg = torch.cat([src_encod_neg[1:], src_encod_neg[:1]], dim=0)
            img2txt_sim_neg = torch.stack([torch.dot(src_encod[i], tgt_encod_neg[i]).unsqueeze(0)
                                           for i in range(batch_size)], dim=0)
            txt2img_sim_neg = torch.stack([torch.dot(tgt_encod[i], src_encod_neg[i]).unsqueeze(0)
                                           for i in range(batch_size)], dim=0)

            # shape [batch_size, +1]
            img2txt_sim_pred = torch.cat([img2txt_sim_pred, img2txt_sim_neg], dim=1)
            txt2img_sim_pred = torch.cat([txt2img_sim_pred, txt2img_sim_neg], dim=1)

        img2txt_sim_pred = self.softmax(img2txt_sim_pred)
        txt2img_sim_pred = self.softmax(txt2img_sim_pred)
        sim_label = torch.zeros(batch_size, dtype=torch.int64).to(img2txt_sim_pred.device)

        dis_img2txt = self.match_nll_loss(torch.log(img2txt_sim_pred + 1e-12), sim_label)
        dis_txt2img = self.match_nll_loss(torch.log(txt2img_sim_pred + 1e-12), sim_label)
        dis_img2txt = torch.sum(dis_img2txt)
        dis_txt2img = torch.sum(dis_txt2img)
        metrics["dis_img2txt"] = dis_img2txt
        metrics["dis_txt2img"] = dis_txt2img
        metrics["loss"] = dis_img2txt + dis_txt2img

        return metrics

    def _collect_metrics_image_caption(self, inputs, outputs):
        """ Calculate loss function by using inputs and outputs. """
        metrics = {}

        tgt_len = torch.sum(torch.sum(inputs["tgt_mask"], dim=1) - 1)

        label = inputs["tgt_token"][:, 1:]
        nll = self.nll_loss(torch.log(outputs["dec_probs"] + 1e-12).permute(0, 2, 1), label)
        nll = torch.sum(nll, dim=1)
        token_nll = torch.sum(nll) / tgt_len
        nll = torch.mean(nll)
        metrics["nll"] = nll  # 一个batch中句子级别的平均nll
        metrics["token_nll"] = token_nll  # 一个batch中token级别的平均nll
        loss = nll

        if self.num_latent > 0 and self.with_bow:
            bow_probs = outputs["bow_probs"].unsqueeze(1)
            bow_probs = bow_probs.repeat(1, label.shape[1], 1)
            bow = self.nll_loss(torch.log(bow_probs + 1e-12).permute(0, 2, 1), label)
            bow = torch.sum(bow, dim=1)
            token_bow = torch.sum(bow) / tgt_len
            bow = torch.mean(bow)
            metrics["bow"] = bow
            metrics["token_bow"] = token_bow
            loss = loss + bow * self.bow_ratio

        if self.num_latent > 0 and self.use_discriminator:
            dis = 0.0 - (torch.log(outputs["pos_probs"]) + torch.log(1.0 - outputs["neg_probs"]))
            dis = torch.mean(dis)
            metrics["dis"] = dis
            loss = loss + dis * self.dis_ratio

        metrics["loss"] = loss
        metrics["token_num"] = tgt_len
        return metrics

    def _collect_metrics_poem_language(self, inputs, outputs):
        """诗词语言模型任务"""
        metrics = {}

        label = inputs["tgt_token"][:, 1:]
        nll = self.nll_loss(torch.log(outputs["dec_probs"] + 1e-12).permute(0, 2, 1), label)
        nll = torch.sum(nll, dim=1)
        nll = torch.mean(nll)

        metrics["nll"] = nll  # 一个batch中句子级别的平均nll
        metrics["loss"] = nll

        return metrics

    def _collect_metrics_image_poem(self, inputs, outputs):
        return self._collect_metrics_image_caption(inputs, outputs)

    def _collect_metrics_image_poem_match(self, inputs, outputs):
        return self._collect_metrics_image_match(inputs, outputs)

    def _optimize(self, loss):
        """ Optimize loss function and update model. """
        self.optimizer.zero_grad()

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.grad_clip is not None and self.grad_clip > 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        return

    def _init_state_image_caption(self, inputs):
        """ Initialize decode state. """
        state = {}

        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]

        batch_size = src_token.shape[0]

        src_embed = self.img_embedder(src_token, src_pos)
        src_embed = self.embed_layer_norm(src_embed)
        seq_len = src_embed.shape[1]

        mask = self._create_mask(src_mask, auto_regressive=False, append_head=self.num_latent > 0)
        src_mask = self._create_mask(src_mask, auto_regressive=False, append_head=False)

        cache = {}
        for l in range(self.num_enc_layers):
            cache[f"layer_{l}"] = {}
            src_embed = self.layers[l](src_embed, 0, mask=src_mask)

        if self.num_latent > 0:
            src_embed = src_embed.unsqueeze(1)
            src_embed = src_embed.repeat(1, self.num_latent, 1, 1)
            src_embed = src_embed.reshape(-1, seq_len, self.hidden_dim)

            latent_embed = self.latent_embeddings
            latent_embed = latent_embed.unsqueeze(1)
            latent_embed = latent_embed.repeat(batch_size, 1, 1)
            latent_embed = self.embed_layer_norm(latent_embed)

            enc_out = torch.cat([latent_embed, src_embed], dim=1)

            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.num_latent, 1, 1)
            mask = mask.reshape(-1, seq_len + 1, seq_len + 1)
        else:
            enc_out = src_embed

        for l in range(self.num_enc_layers, self.num_layers):
            cache[f"layer_{l}"] = {}
            enc_out = self.layers[l](enc_out, 2, mask=mask, cache=cache[f"layer_{l}"])

        state["cache"] = cache
        state["mask"] = mask[:, :1]  # 对src端seq的mask；tgt端每次decode一个token，所以mask矩阵只有一行
        if self.num_latent > 0:
            state["batch_size"] = batch_size * self.num_latent
            shape = [batch_size * self.num_latent, 1, 1]
        else:
            state["batch_size"] = batch_size
            shape = [batch_size, 1, 1]
        state["pred_mask"] = torch.ones(shape, dtype=torch.float32)
        state["pred_pos"] = torch.zeros(shape, dtype=torch.int64)
        if self.use_gpu:
            state["pred_mask"] = state["pred_mask"].cuda()
            state["pred_pos"] = state["pred_pos"].cuda()

        return state

    def _init_state_image_poem(self, inputs):
        """ Initialize decode state. """
        state = {}

        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]

        batch_size = src_token.shape[0]

        src_embed = self.img_embedder(src_token, src_pos)
        src_embed = self.embed_layer_norm(src_embed)
        seq_len = src_embed.shape[1]

        mask = self._create_mask(src_mask, auto_regressive=False, append_head=self.num_latent > 0)
        src_mask = self._create_mask(src_mask, auto_regressive=False, append_head=False)

        cache = {}
        for l in range(self.num_enc_layers):
            cache[f"layer_{l}"] = {}
            src_embed = self.layers[l](src_embed, 0, mask=src_mask)

        if self.num_latent > 0:
            src_embed = src_embed.unsqueeze(1)
            src_embed = src_embed.repeat(1, self.num_latent, 1, 1)
            src_embed = src_embed.reshape(-1, seq_len, self.hidden_dim)

            latent_embed = self.latent_embeddings
            latent_embed = latent_embed.unsqueeze(1)
            latent_embed = latent_embed.repeat(batch_size, 1, 1)
            latent_embed = self.embed_layer_norm(latent_embed)

            enc_out = torch.cat([latent_embed, src_embed], dim=1)

            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.num_latent, 1, 1)
            mask = mask.reshape(-1, seq_len + 1, seq_len + 1)
        else:
            enc_out = src_embed

        for l in range(self.num_enc_layers, self.num_layers):
            cache[f"layer_{l}"] = {}
            enc_out = self.layers[l](enc_out, 2, mask=mask, cache=cache[f"layer_{l}"])

        state["cache"] = cache
        state["mask"] = mask[:, :1]  # 对src端seq的mask；tgt端每次decode一个token，所以mask矩阵只有一行
        if self.num_latent > 0:
            state["batch_size"] = batch_size * self.num_latent
            shape = [batch_size * self.num_latent, 1, 1]
        else:
            state["batch_size"] = batch_size
            shape = [batch_size, 1, 1]
        state["pred_mask"] = torch.ones(shape, dtype=torch.float32)
        # state["pred_pos"] = torch.zeros(shape, dtype=torch.int64)
        if self.use_gpu:
            state["pred_mask"] = state["pred_mask"].cuda()
            # state["pred_pos"] = state["pred_pos"].cuda()

        return state

    def _decode_image_caption(self, state):
        # shape: [batch_size, 1, 1]
        # 第2维为1是tgt端每次只输入一个token(之前的有cache)
        # 第3维为1是为了进行embedding
        pred_token = state["pred_token"]
        pred_mask = state["pred_mask"]
        pred_pos = state["pred_pos"]
        pred_turn = state["pred_turn"]

        # shape: [batch_size, 1, seq_len]
        # 解码每个token时，对src端seq的mask
        # 因为每次只解码一个token，所以第2维为1，seq_len是src端seq的长度
        mask = state["mask"]
        tgt_mask = state["tgt_mask"] if "tgt_mask" in state.keys() \
            else None

        # list of shape(len: num_layers): [batch_size, seq_len, hidden_dim]
        cache = state["cache"]

        # [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        pred_embed = self.text_embedder(pred_token, pred_pos, pred_turn, use_turn=False).squeeze(-2)
        pred_embed = self.embed_layer_norm(pred_embed)

        # shape: [batch_size, 1, seq_len + 1]
        if tgt_mask is not None:
            tgt_mask = torch.cat([tgt_mask, 1 - pred_mask], dim=2)
        else:
            tgt_mask = torch.zeros([pred_token.shape[0], 1, 1], dtype=torch.int64)
            if self.use_gpu:
                tgt_mask = tgt_mask.cuda()

        mask = torch.cat([mask, 1 - pred_mask], dim=2)

        for l in range(self.num_enc_layers):
            pred_embed = self.layers[l](pred_embed, 1, mask=tgt_mask, cache=cache[f"layer_{l}"])

        for l in range(self.num_enc_layers, self.num_layers):
            pred_embed = self.layers[l](pred_embed, 2, mask=mask, cache=cache[f"layer_{l}"])

        # shape: [batch_size, 1, vocab_size]
        if self.two_layer_predictor:
            pred_embed = self.pre_predictor(pred_embed)
        if self.weight_sharing:
            token_embedding = self.text_embedder.token_embedding.weight
            pred_logits = torch.matmul(pred_embed, token_embedding.T)
        else:
            pred_logits = self.predictor(pred_embed)

        # shape: [batch_size, vocab_size]
        pred_logits = pred_logits[:, 0]
        pred_probs = self.softmax(pred_logits)
        pred_logits = torch.log(pred_probs)

        state["mask"] = mask
        state["tgt_mask"] = tgt_mask
        return pred_logits, state

    def _decode_image_poem(self, state):
        # shape: [batch_size, 1, 1]
        # 第2维为1是tgt端每次只输入一个token(之前的有cache)
        # 第3维为1是为了进行embedding
        pred_token = state["pred_token"]
        pred_mask = state["pred_mask"]
        pred_pos = state["pred_pos"]
        pred_turn = state["pred_turn"]

        # shape: [batch_size, 1, seq_len]
        # 解码每个token时，对src端seq的mask
        # 因为每次只解码一个token，所以第2维为1，seq_len是src端seq的长度
        mask = state["mask"]
        tgt_mask = state["tgt_mask"] if "tgt_mask" in state.keys() \
            else None

        # list of shape(len: num_layers): [batch_size, seq_len, hidden_dim]
        cache = state["cache"]

        # [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        pred_embed = self.text_embedder(pred_token, pred_pos, pred_turn).squeeze(-2)
        pred_embed = self.embed_layer_norm(pred_embed)

        # shape: [batch_size, 1, seq_len + 1]
        if tgt_mask is not None:
            tgt_mask = torch.cat([tgt_mask, 1 - pred_mask], dim=2)
        else:
            tgt_mask = torch.zeros([pred_token.shape[0], 1, 1], dtype=torch.int64)
            if self.use_gpu:
                tgt_mask = tgt_mask.cuda()

        mask = torch.cat([mask, 1 - pred_mask], dim=2)

        for l in range(self.num_enc_layers):
            pred_embed = self.layers[l](pred_embed, 1, mask=tgt_mask, cache=cache[f"layer_{l}"])

        for l in range(self.num_enc_layers, self.num_layers):
            pred_embed = self.layers[l](pred_embed, 2, mask=mask, cache=cache[f"layer_{l}"])

        # shape: [batch_size, 1, vocab_size]
        if self.two_layer_predictor:
            pred_embed = self.pre_predictor(pred_embed)
        if self.weight_sharing:
            token_embedding = self.text_embedder.token_embedding.weight
            pred_logits = torch.matmul(pred_embed, token_embedding.T)
        else:
            pred_logits = self.predictor(pred_embed)

        # shape: [batch_size, vocab_size]
        pred_logits = pred_logits[:, 0]
        pred_probs = self.softmax(pred_logits)
        pred_logits = torch.log(pred_probs)

        state["mask"] = mask
        state["tgt_mask"] = tgt_mask
        return pred_logits, state, pred_probs

    def _ranking_image_caption(self, inputs, predictions):
        """ Reranking generated responses. """
        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]
        src_embed = self.img_embedder(src_token, src_pos)

        batch_size, num_latent, tgt_seq_len = predictions.shape

        # shape: [batch_size, num_latent, seq_len]
        preds_token = predictions
        preds_mask = torch.not_equal(preds_token, self.padding_idx).type_as(src_mask)
        preds_pos = torch.arange(0, tgt_seq_len, 1, dtype=torch.float32)
        preds_pos = F_alias.unsqueeze(preds_pos, dims=[0, 0])
        preds_pos = preds_pos.repeat(batch_size, num_latent, 1)
        preds_pos = preds_pos.type_as(preds_token)

        scores = []
        for i in range(num_latent):
            pred_token = preds_token[:, i]
            pred_mask = preds_mask[:, i]
            pred_pos = preds_pos[:, i]
            pred_turn = preds_pos[:, i]

            pred_embed = self.text_embedder(pred_token, pred_pos, pred_turn, use_turn=False)

            input_mask = torch.cat([src_mask, pred_mask], dim=1)
            mask = self._create_mask(input_mask, auto_regressive=False, append_head=True)
            enc_mask = self._create_mask(src_mask, auto_regressive=False, append_head=False)
            dec_mask = self._create_mask(pred_mask, auto_regressive=False, append_head=False)

            enc_embed = self._encode_network(enc_mask, src_embed, "img")
            dec_embed = self._encode_network(dec_mask, pred_embed, "txt")

            embed = torch.cat([enc_embed, dec_embed], dim=1)

            mask_embed = self.mask_embed
            mask_embed = mask_embed.repeat(batch_size, 1, 1)
            mask_embed = self.embed_layer_norm(mask_embed)
            out = torch.cat([mask_embed, embed], dim=1)

            for l in range(self.num_enc_layers, self.num_layers):
                out = self.layers[l](out, 2, mask=mask)

            mask_embed = out[:, 0]
            score = self.discriminator(mask_embed)
            scores.append(score[:, 0])
        scores = torch.stack(scores, dim=1)
        return scores

    def _ranking_image_poem(self, inputs, predictions, pos_index, turn_index):
        """ Reranking generated responses. """
        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]
        src_embed = self.img_embedder(src_token, src_pos)

        batch_size, num_latent, tgt_seq_len = predictions.shape

        # shape: [batch_size, num_latent, seq_len]
        preds_token = predictions
        preds_mask = torch.not_equal(preds_token, self.padding_idx).type_as(src_mask)
        preds_pos = pos_index
        preds_turn = turn_index

        scores = []
        for i in range(num_latent):
            pred_token = preds_token[:, i]
            pred_mask = preds_mask[:, i]
            pred_pos = preds_pos[:, i]
            pred_turn = preds_turn[:, i]

            pred_embed = self.text_embedder(pred_token, pred_pos, pred_turn)

            input_mask = torch.cat([src_mask, pred_mask], dim=1)
            mask = self._create_mask(input_mask, auto_regressive=False, append_head=True)
            enc_mask = self._create_mask(src_mask, auto_regressive=False, append_head=False)
            dec_mask = self._create_mask(pred_mask, auto_regressive=False, append_head=False)

            enc_embed = self._encode_network(enc_mask, src_embed, "img")
            dec_embed = self._encode_network(dec_mask, pred_embed, "txt")

            embed = torch.cat([enc_embed, dec_embed], dim=1)

            mask_embed = self.mask_embed
            mask_embed = mask_embed.repeat(batch_size, 1, 1)
            mask_embed = self.embed_layer_norm(mask_embed)
            out = torch.cat([mask_embed, embed], dim=1)

            for l in range(self.num_enc_layers, self.num_layers):
                out = self.layers[l](out, 2, mask=mask)

            mask_embed = out[:, 0]
            score = self.discriminator(mask_embed)
            scores.append(score[:, 0])
        scores = torch.stack(scores, dim=1)
        return scores

    def _infer_image_caption(self, inputs):
        """ Real inference process of model. """
        results = {}

        # Initial decode state.
        state = self._init_state_image_caption(inputs)
        if "post_probs" in state:
            results["post_probs"] = state.pop("post_probs")

        # Generation process.
        gen_results = self.generator(self._decode_image_caption, state)
        results.update(gen_results)

        if self.num_latent > 0:
            batch_size = state["batch_size"] // self.num_latent
            results["scores"] = results["scores"].reshape(batch_size, self.num_latent)
            results["log_p"] = results["scores"]  # 解码产生的分数(generator)
            # results["src"] = inputs["src_token"].reshape(batch_size, -1)
            if "tgt_token" in inputs:
                results["tgt"] = inputs["tgt_token"].reshape(batch_size, -1)
            results["preds"] = results["preds"].reshape(batch_size, self.num_latent, -1)
            if self.use_discriminator:
                results["scores"] = self._ranking_image_caption(inputs, results["preds"])  # discriminator产生的分数
        else:
            batch_size = state["batch_size"]
            if "tgt_token" in inputs:
                results["tgt"] = inputs["tgt_token"].reshape(batch_size, -1)
        return results

    def _infer_image_poem(self, inputs):
        """ Real inference process of model. """
        results = {}

        # Initial decode state.
        state = self._init_state_image_poem(inputs)
        if "post_probs" in state:
            results["post_probs"] = state.pop("post_probs")

        # Generation process.
        gen_results = self.generator(self._decode_image_poem, state)
        results.update(gen_results)

        if self.num_latent > 0:
            batch_size = state["batch_size"] // self.num_latent
            # results["scores"] = results["scores"].reshape(batch_size, self.num_latent)
            # results["log_p"] = results["scores"]  # 解码产生的分数(generator)
            # results["src"] = inputs["src_token"].reshape(batch_size, -1)
            if "tgt_token" in inputs:
                results["tgt"] = inputs["tgt_token"].reshape(batch_size, -1)
            results["preds"] = results["preds"].reshape(batch_size, self.num_latent, -1)
            results["pos"] = results["pos"].reshape(batch_size, self.num_latent, -1)
            results["turns"] = results["turns"].reshape(batch_size, self.num_latent, -1)
            if self.use_discriminator:
                results["scores"] = self._ranking_image_poem(inputs, results["preds"], results["pos"], results["turns"])  # discriminator产生的分数
        else:
            batch_size = state["batch_size"]
            if "tgt_token" in inputs:
                results["tgt"] = inputs["tgt_token"].reshape(batch_size, -1)
        return results

    # def _init_state(self, inputs):
    #     """ Initialize decode state. """
    #     state = {}
    #
    #     src_token = inputs["src_token"]
    #     src_mask = inputs["src_mask"]
    #     src_pos = inputs["src_pos"]
    #     src_type = inputs["src_type"]
    #     src_turn = inputs["src_turn"]
    #
    #     batch_size = src_token.shape[0]
    #     seq_len = src_token.shape[1]
    #
    #     src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
    #     src_embed = self.embed_layer_norm(src_embed)
    #
    #     mask = self._create_mask(src_mask, append_head=self.num_latent > 0)
    #
    #     if self.num_latent > 0:
    #         src_embed = src_embed.unsqueeze(1)
    #         src_embed = src_embed.repeat(1, self.num_latent, 1, 1)
    #         src_embed = src_embed.reshape(-1, seq_len, self.hidden_dim)
    #
    #         latent_embed = self.latent_embeddings
    #         latent_embed = latent_embed.unsqueeze(1)
    #         latent_embed = latent_embed.repeat(batch_size, 1, 1)
    #         latent_embed = self.embed_layer_norm(latent_embed)
    #
    #         enc_out = torch.cat([latent_embed, src_embed], dim=1)
    #
    #         mask = mask.unsqueeze(1)
    #         mask = mask.repeat(1, self.num_latent, 1, 1)
    #         mask = mask.reshape(-1, seq_len + 1, seq_len + 1)
    #     else:
    #         enc_out = src_embed
    #
    #     cache = {}
    #     for l, layer in enumerate(self.layers):
    #         cache[f"layer_{l}"] = {}
    #         enc_out = layer(enc_out, mask, cache[f"layer_{l}"])
    #
    #     state["cache"] = cache
    #     state["mask"] = mask[:, :1]  # 对src端seq的mask；tgt端每次decode一个token，所以mask矩阵只有一行
    #     if self.num_latent > 0:
    #         state["batch_size"] = batch_size * self.num_latent
    #         shape = [batch_size * self.num_latent, 1, 1]
    #     else:
    #         state["batch_size"] = batch_size
    #         shape = [batch_size, 1, 1]
    #     state["pred_mask"] = torch.ones(shape, dtype=torch.float32)
    #     state["pred_pos"] = torch.zeros(shape, dtype=torch.int64)
    #     state["pred_type"] = torch.zeros(shape, dtype=torch.int64)
    #     state["pred_turn"] = torch.zeros(shape, dtype=torch.int64)
    #     if self.use_gpu:
    #         state["pred_mask"] = state["pred_mask"].cuda()
    #         state["pred_pos"] = state["pred_pos"].cuda()
    #         state["pred_type"] = state["pred_type"].cuda()
    #         state["pred_turn"] = state["pred_turn"].cuda()
    #
    #     if "tgt_token" in inputs and self.num_latent > 0:
    #         tgt_token = inputs["tgt_token"][:, :-1]
    #         tgt_mask = inputs["tgt_mask"][:, :-1]
    #         tgt_pos = inputs["tgt_pos"][:, :-1]
    #         tgt_type = inputs["tgt_type"][:, :-1]
    #         tgt_turn = inputs["tgt_turn"][:, :-1]
    #
    #         input_mask = torch.cat([src_mask, tgt_mask], dim=1)
    #         src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
    #         tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
    #         embed = torch.cat([src_embed, tgt_embed], dim=1)
    #         embed = self.embed_layer_norm(embed)
    #
    #         batch_size = src_token.shape[0]
    #         src_len = src_token.shape[1]
    #         tgt_len = tgt_token.shape[1]
    #
    #         post_embed, post_probs, post_logits = self._posteriori_network(
    #             input_mask, embed, batch_size, src_len, tgt_len)
    #         state["post_probs"] = post_probs  # 应该是用来判断选出的response对应的z是不是post_prob中最高的z
    #
    #     return state
    #

    # def _decode(self, state):
    #     """ Decoding one time stamp. """
    #     # shape: [batch_size, 1, seq_len]
    #     # 解码每个token时，对src端seq的mask
    #     # 因为每次只解码一个token，所以第2维为1，seq_len是src端seq的长度
    #     mask = state["mask"]
    #
    #     # shape: [batch_size, 1, 1]
    #     # 第2维为1是tgt端每次只输入一个token(之前的有cache)
    #     # 第3维为1是为了进行embedding
    #     pred_token = state["pred_token"]
    #     pred_mask = state["pred_mask"]
    #     pred_pos = state["pred_pos"]
    #     pred_type = state["pred_type"]
    #     pred_turn = state["pred_turn"]
    #
    #     # list of shape(len: num_layers): [batch_size, seq_len, hidden_dim]
    #     cache = state["cache"]
    #
    #     pred_embed = self.embedder(pred_token, pred_pos, pred_type, pred_turn).squeeze(-2)
    #     pred_embed = self.embed_layer_norm(pred_embed)
    #
    #     # shape: [batch_size, 1, seq_len + 1]
    #     mask = torch.cat([mask, 1 - pred_mask], dim=2)
    #
    #     # shape: [batch_size, 1, hidden_dim]
    #     for l, layer in enumerate(self.layers):
    #         pred_embed = layer(pred_embed, mask, cache[f"layer_{l}"])
    #
    #     # shape: [batch_size, 1, vocab_size]
    #     if self.two_layer_predictor:
    #         pred_embed = self.pre_predictor(pred_embed)
    #     if self.weight_sharing:
    #         token_embedding = self.embedder.token_embedding.weight
    #         pred_logits = torch.matmul(pred_embed, token_embedding.T)
    #     else:
    #         pred_logits = self.predictor(pred_embed)
    #     pred_logits = pred_logits[:, 0]
    #     pred_probs = self.softmax(pred_logits)
    #     pred_logits = torch.log(pred_probs)  # 取log方便整个句子的分数计算log(p)=log(p1*p2*p3)=log(p1)+log(p2)+log(p3)
    #
    #     state["mask"] = mask
    #     return pred_logits, state

    # def _ranking(self, inputs, predictions):
    #     """ Reranking generated responses. """
    #     src_token = inputs["src_token"]
    #     src_mask = inputs["src_mask"]
    #     src_pos = inputs["src_pos"]
    #     src_type = inputs["src_type"]
    #     src_turn = inputs["src_turn"]
    #     src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
    #
    #     batch_size, num_latent, tgt_seq_len = predictions.shape
    #
    #     # shape: [batch_size, num_latent, seq_len]
    #     preds_token = predictions
    #     preds_mask = torch.not_equal(preds_token, self.padding_idx).type_as(src_mask)
    #     preds_pos = torch.arange(0, tgt_seq_len, 1, dtype=torch.float32)
    #     preds_pos = F_alias.unsqueeze(preds_pos, dims=[0, 0])
    #     preds_pos = preds_pos.repeat(batch_size, num_latent, 1)
    #     preds_pos = preds_pos.type_as(preds_token)
    #     preds_type = torch.zeros_like(preds_token)
    #     preds_turn = torch.zeros_like(preds_token)
    #
    #     scores = []
    #     for i in range(num_latent):
    #         pred_token = preds_token[:, i]
    #         pred_mask = preds_mask[:, i]
    #         pred_pos = preds_pos[:, i]
    #         pred_type = preds_type[:, i]
    #         pred_turn = preds_turn[:, i]
    #
    #         input_mask = torch.cat([src_mask, pred_mask], dim=1)
    #         pred_embed = self.embedder(pred_token, pred_pos, pred_type, pred_turn)
    #         embed = torch.cat([src_embed, pred_embed], dim=1)
    #         embed = self.embed_layer_norm(embed)
    #
    #         mask_embed = self.mask_embed
    #         mask_embed = mask_embed.repeat(batch_size, 1, 1)
    #         mask_embed = self.embed_layer_norm(mask_embed)
    #
    #         out = torch.cat([mask_embed, embed], dim=1)
    #         mask = self._create_mask(input_mask, append_head=True)
    #
    #         for layer in self.layers:
    #             out = layer(out, mask, None)
    #
    #         mask_embed = out[:, 0]
    #         score = self.discriminator(mask_embed)
    #         scores.append(score[:, 0])
    #     scores = torch.stack(scores, dim=1)
    #     return scores

    # def _infer(self, inputs):
    #     """ Real inference process of model. """
    #     results = {}
    #
    #     # Initial decode state.
    #     state = self._init_state(inputs)
    #     if "post_probs" in state:
    #         results["post_probs"] = state.pop("post_probs")
    #
    #     # Generation process.
    #     gen_results = self.generator(self._decode, state)
    #     results.update(gen_results)
    #
    #     if self.num_latent > 0:
    #         batch_size = state["batch_size"] // self.num_latent
    #         results["scores"] = results["scores"].reshape(batch_size, self.num_latent)
    #         results["log_p"] = results["scores"]  # 解码产生的分数(generator)
    #         results["src"] = inputs["src_token"].reshape(batch_size, -1)
    #         if "tgt_token" in inputs:
    #             results["tgt"] = inputs["tgt_token"].reshape(batch_size, -1)
    #         results["preds"] = results["preds"].reshape(batch_size, self.num_latent, -1)
    #         if self.use_discriminator:
    #             results["scores"] = self._ranking(inputs, results["preds"])  # discriminator产生的分数
    #     else:
    #         batch_size = state["batch_size"]
    #         if "tgt_token" in inputs:
    #             results["tgt"] = inputs["tgt_token"].reshape(batch_size, -1)
    #     return results


UnifiedTransformer.register("UnifiedTransformer")
