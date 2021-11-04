import torch
import torch.nn as nn
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, AttentionLayer
from .Transformer import TransformerModel
from .PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)


class VisionTransformer_v3(nn.Module):
    def __init__(self, args,
                 conv_patch_representation=False,
                 use_representation=True,
                 positional_encoding_type="learned",
                 num_channels=2048
                 ):
        super(VisionTransformer_v3, self).__init__()
        img_dim = args.enc_layers
        out_dim = args.numclass
        num_channels = args.dim_feature
        assert args.embedding_dim % args.num_heads == 0
        assert img_dim % args.patch_dim == 0
        self.embedding_dim = args.embedding_dim
        self.num_heads = args.num_heads
        self.patch_dim = args.patch_dim
        self.num_channels = num_channels
        self.dropout_rate = args.dropout_rate
        self.attn_dropout_rate = args.attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int(img_dim // self.patch_dim)
        self.seq_length = self.num_patches + 1
        self.flatten_dim = self.patch_dim * self.patch_dim * num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        print('position encoding :', positional_encoding_type)

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.encoder = TransformerModel(
            self.embedding_dim,
            args.num_layers,
            self.num_heads,
            args.hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)

        d_model = args.decoder_embedding_dim
        use_representation = False
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(self.embedding_dim + d_model, args.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(args.hidden_dim // 2, out_dim),
            )
        else:
            self.mlp_head = nn.Linear(self.embedding_dim + d_model, out_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv1d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=self.patch_dim,
                stride=self.patch_dim,
                padding=self._get_padding(
                    'VALID', self.patch_dim,
                ),
            )
        else:
            self.conv_x = None

        self.to_cls_token = nn.Identity()

        feat_dim = 128
        self.sup_con_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, feat_dim)
        )

        # Decoder
        factor = 1  # 5
        dropout = args.decoder_attn_dropout_rate
        n_heads = args.decoder_num_heads
        d_layers = args.decoder_layers
        d_ff = args.decoder_embedding_dim_out
        activation = 'gelu'
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),  # True
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),  # False
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, args.query_num, d_model))
        if positional_encoding_type == "learned":
            self.decoder_position_encoding = LearnedPositionalEncoding(
                args.query_num, self.embedding_dim, args.query_num
            )
        elif positional_encoding_type == "fixed":
            self.decoder_position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        print('position decoding :', positional_encoding_type)
        self.classifier = nn.Linear(d_model, out_dim)
        self.after_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, sequence_input_rgb):
        x = sequence_input_rgb

        x = self.linear_encoding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)  # not delete

        # apply transformer
        x = self.encoder(x)
        x = self.pre_head_ln(x)  # [128, 65, 1024]
        # decoder
        decoder_cls_token = self.decoder_cls_token.expand(x.shape[0], -1, -1)
        dec = self.decoder(decoder_cls_token, x)  # [128, 8, 1024]
        dec = self.after_dropout(dec)  # add
        dec_for_token = dec.mean(dim=1)
        dec_cls_out = self.classifier(dec)
        x = torch.cat((self.to_cls_token(x[:, -1]), dec_for_token), dim=1)
        x = self.mlp_head(x)

        return x, dec_cls_out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)
