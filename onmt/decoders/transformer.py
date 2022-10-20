"""
Implementation of "Attention is All You Need" and of
subsequent transformer based architectures
"""

import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask

class TransformerDecoder(DecoderBase):
    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        copy_attn,
        self_attn_type,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        aan_useffn,
        full_context_alignment,
        alignment_layer,
        alignment_heads,
        pos_ffn_activation_fn=ActivationFunction.relu,
    ):
        super(TransformerDecoder, self).__init__(
            d_model, copy_attn, embeddings, alignment_layer
        )

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank=None, step=None, **kwargs):
        print("new transformer decode")
        return [], []
        # if memory_bank is None:
        #     memory_bank = self.embeddings(tgt)
        # if step == 0:
        #     self._init_cache(memory_bank)

        # tgt_words = tgt[:, :, 0].transpose(0, 1)

        # emb = self.embeddings(tgt, step=step)
        # assert emb.dim() == 3  # len x batch x embedding_dim

        # output = emb.transpose(0, 1).contiguous()
        # src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        # pad_idx = self.embeddings.word_padding_idx
        # src_lens = kwargs["memory_lengths"]
        # src_max_len = self.state["src"].shape[0]
        # src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        # tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        # with_align = kwargs.pop("with_align", False)
        # attn_aligns = []

        # for i, layer in enumerate(self.transformer_layers):
        #     layer_cache = (
        #         self.state["cache"]["layer_{}".format(i)]
        #         if step is not None
        #         else None
        #     )
        #     output, attn, attn_align = layer(
        #         output,
        #         src_memory_bank,
        #         src_pad_mask,
        #         tgt_pad_mask,
        #         layer_cache=layer_cache,
        #         step=step,
        #         with_align=with_align,
        #     )
        #     if attn_align is not None:
        #         attn_aligns.append(attn_align)

        # output = self.layer_norm(output)
        # dec_outs = output.transpose(0, 1).contiguous()
        # attn = attn.transpose(0, 1).contiguous()

        # attns = {"std": attn}
        # if self._copy:
        #     attns["copy"] = attn
        # if with_align:
        #     attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
        #     # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros(
                    (batch_size, 1, depth), device=memory_bank.device
                )
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

