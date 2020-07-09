import torch
import torch.nn as nn
from SubLayers import MultiHeadAttention, PositionWiseFeedForward

class EncoderLayer(nn.Module):
    """ Two layer encoder layer """

    def __init__(self, model_d, inner_d, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, model_d, d_k, d_v, dropout=dropout) # self-attention
        self.pos_ffn = PositionWiseFeedForward(model_d, inner_d, dropout=0.1) # position-wise feed-forward network

    def forward(self, enc_input, slf_attn_mask=None):
        # get encoder output and self attention, q, k, v == enc_input
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        # feed output from self-attention layer to position-wise ffn
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    """ Three layer decoder layer """

    def __init__(self, model_d, inner_d, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, model_d, d_k, d_v, dropout=dropout) # self-attention for decoder
        self.enc_attn = MultiHeadAttention(n_head, model_d, d_k, d_v, dropout=dropout) # self-attention for encoder
        self.pos_ffn = PositionWiseFeedForward(model_d, inner_d, dropout=0.1)

    def forward(self, enc_output, dec_input, slf_attn_mask=None, dec_enc_attn_mask=None):
        # First sub-layer is a masked multi-head attention layer receiving decoder inputs
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        # Second sub-layer is a multi-head attention layer receiving decoder outputs (q) and encoder outputs (k, v)
        # Allows every position in the decoder to attend over all positions in the input sequence
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        # Final pos-wise ffn
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn

