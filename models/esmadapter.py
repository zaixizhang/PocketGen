# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import math

import torch
import esm
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from esm.modules import (
    TransformerLayer,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
    ContactPredictionHead,
    ESM1LayerNorm,
    FeedForwardNetwork,
    NormalizedResidualBlock,
    gelu,
)
from esm.multihead_attention import MultiheadAttention


class ProteinBertModelWithStructuralAdatper(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--num_layers", default=36, type=int, metavar="N", help="number of layers"
        )
        parser.add_argument(
            "--embed_dim", default=1280, type=int, metavar="N", help="embedding dimension"
        )
        parser.add_argument(
            "--logit_bias", action="store_true", help="whether to apply bias to logits"
        )
        parser.add_argument(
            "--ffn_embed_dim",
            default=5120,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention_heads",
            default=20,
            type=int,
            metavar="N",
            help="number of attention heads",
        )

    @classmethod
    def from_pretrained(cls, args, override_args=None, name='esm1b_t33_650M_UR50S'):
        pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)

        args = OmegaConf.merge(vars(deepcopy(pretrained_model.args)), args)
        args.adapter_layer_indices = [6, 20 ,32]

        model = cls(args, deepcopy(alphabet))
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

        del pretrained_model

        # freeze pretrained parameters
        for pname, param in model.named_parameters():
            if 'adapter' not in pname:
                param.requires_grad = False
        return model

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = getattr(self.args, "emb_layer_norm_before", False)
        if self.args.arch == "roberta_large":
            self.model_version = "ESM-1b"
            self._init_submodules_esm1b()
        else:
            self.model_version = "ESM-1"
            self._init_submodules_esm1()

    def _init_submodules_common(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                self._init_layer(layer_idx)
                for layer_idx in range(self.args.layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.args.layers * self.args.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

    def _init_layer(self, layer_idx):
        if layer_idx in self.args.adapter_layer_indices:
            layer = TransforerLayerWithStructralAdapter(
                self.args.embed_dim,
                self.args.ffn_embed_dim,
                self.args.attention_heads,
                encoder_embed_dim=self.args.encoder.d_model,
                add_bias_kv=(self.model_version != "ESM-1b"),
                use_esm1b_layer_norm=(self.model_version == "ESM-1b"),
            )
        else:
            layer = TransformerLayer(
                self.args.embed_dim,
                self.args.ffn_embed_dim,
                self.args.attention_heads,
                add_bias_kv=(self.model_version != "ESM-1b"),
                use_esm1b_layer_norm=(self.model_version == "ESM-1b"),
            )
        return layer

    def _init_submodules_esm1b(self):
        self._init_submodules_common()
        self.embed_scale = 1
        self.embed_positions = LearnedPositionalEmbedding(
            self.args.max_positions, self.args.embed_dim, self.padding_idx
        )
        self.emb_layer_norm_before = (
            ESM1bLayerNorm(self.args.embed_dim) if self.emb_layer_norm_before else None
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def _init_submodules_esm1(self):
        self._init_submodules_common()
        self.embed_scale = math.sqrt(self.args.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(self.args.embed_dim, self.padding_idx)
        self.embed_out = nn.Parameter(torch.zeros((self.alphabet_size, self.args.embed_dim)))
        self.embed_out_bias = None
        if self.args.final_bias:
            self.embed_out_bias = nn.Parameter(torch.zeros(self.alphabet_size))

    def forward_layers(self, x, encoder_out, padding_mask, repr_layers=[], hidden_representations=[],
                       need_head_weights=False, attn_weights=[]):
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.args.adapter_layer_indices:
                x, attn = layer(
                    x, encoder_out, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
                )
            else:
                x, attn = layer(
                    x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
                )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        return x, hidden_representations, attn_weights

    def forward(self, tokens, encoder_out, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if getattr(self.args, "token_dropout", False):
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        x = x + self.embed_positions(tokens)

        if self.model_version == "ESM-1b":
            if self.emb_layer_norm_before:
                x = self.emb_layer_norm_before(x)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        # for layer_idx, layer in enumerate(self.layers):
        #     x, attn = layer(
        #         x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
        #     )
        #     if (layer_idx + 1) in repr_layers:
        #         hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        #     if need_head_weights:
        #         # (H, B, T, T) => (B, H, T, T)
        #         attn_weights.append(attn.transpose(1, 0))

        x, hidden_representations, attn_weights = self.forward_layers(
            x, encoder_out, padding_mask,
            repr_layers=repr_layers,
            hidden_representations=hidden_representations,
            need_head_weights=need_head_weights,
            attn_weights=attn_weights if need_head_weights else None
        )

        if self.model_version == "ESM-1b":
            x = self.emb_layer_norm_after(x)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

            # last hidden representation should have layer norm applied
            if len(self.layers) in repr_layers:
                hidden_representations[len(self.layers)] = x
            x = self.lm_head(x)
        else:
            x = F.linear(x, self.embed_out, bias=self.embed_out_bias)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if self.model_version == "ESM-1":
                # ESM-1 models have an additional null-token for attention, which we remove
                attentions = attentions[..., :-1]
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    @property
    def num_layers(self):
        return self.args.layers


class TransforerLayerWithStructralAdapter(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            encoder_embed_dim,
            add_bias_kv=True,
            use_esm1b_layer_norm=False,
            use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings

        self.encoder_embed_dim = encoder_embed_dim
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

        # structural adapter
        self.structural_adapter_attn = NormalizedResidualBlock(
            layer=MultiheadAttention(
                self.embed_dim,
                self.attention_heads,
                kdim=self.encoder_embed_dim,
                vdim=self.encoder_embed_dim,
                add_bias_kv=add_bias_kv,
                add_zero_attn=False,
                use_rotary_embeddings=True,
            ),
            embedding_dim=self.embed_dim,
            dropout=0.1
        )
        self.structural_adapter_ffn = NormalizedResidualBlock(
            layer=FeedForwardNetwork(
                self.embed_dim,
                self.embed_dim // 2,  # NOTE: bottleneck FFN is important
                # self.ffn_embed_dim,
                activation_dropout=0.1
            ),
            embedding_dim=self.embed_dim,
            dropout=0.1
        )

    def forward(
            self, x, encoder_out, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        # x = self.forward_adapter(x, encoder_out, attn_mask=self_attn_mask, attn_padding_mask=self_attn_padding_mask)

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        x = x + self.forward_adapter(x, encoder_out, attn_mask=self_attn_mask, attn_padding_mask=self_attn_padding_mask)
        return x, attn

    def forward_adapter(self, x, encoder_out, attn_mask, attn_padding_mask):
        encoder_feats = encoder_out['feats']
        encoder_feats = encoder_feats.transpose(0, 1)

        x = self.structural_adapter_attn(
            x,
            key=encoder_feats,
            value=encoder_feats,
            key_padding_mask=attn_padding_mask,
            attn_mask=attn_mask,
            need_weights=False
        )[0]

        x = self.structural_adapter_ffn(x)
        # x = x.transpose(0, 1)
        return x