import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from Layers import EncoderLayer, DecoderLayer
from utils import PositionalEncoding, padding_mask, sequence_mask

class Encoder(nn.Module):
    """ Encoder model with multiple encoder layers"""

    def __init__(self, n_vocab, vocab_size, n_position, d_k, d_v, num_layers=6, model_d=512,
                 n_head=8, ffn_d=2048, dropout=0.1):
        """
        Args:
            n_vocab: number of vocabs/embeddings
            vocab_size: max vocab dimension
            n_position: maximum number of positions
            num_layers: number of encoder layers
            d_k: dimension of q and k
            d_v: dimension of v
            model_d: model dimensions/embedding size
            n_head: number of attention heads
            ffn_d: inner layer dimension of the feed-forward network
            dropout: dropout rate
        """
        super().__init__()

        # Input word embedding with paddings masked
        self.word_embedding = nn.Embedding(n_vocab, vocab_size, padding_idx=0)
        # Position encoding
        self.position_encoding = PositionalEncoding(vocab_size, n_position)
        # Encoder stack consisting a list of encoder layers
        self.layer_stacks = nn.ModuleList([EncoderLayer(model_d, ffn_d, n_head, d_k, d_v, dropout=dropout) for _ in
                                           range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_d, eps=1e-6)

    def forward(self, x, mask, return_attn=False):
        """
        Args:
            x: input sequence
            mask: input masking
            return_attn: whether to return attention values
        """

        enc_slf_attn_list = [] # Create a list containing encoder self-attention values

        x = self.word_embedding(x) # get word embedding for input sequence
        x += self.position_encoding # add positional encoding
        x = self.dropout(x) # apply dropout
        x = self.layer_norm(x) # apply layer normalization

        # Compute encoder outputs and self-attention through the encoder layers
        for enc_layer in self.layer_stacks:
            enc_output, enc_slf_attn = enc_layer(x, self_attn_mask=mask)
            if return_attn:
                enc_slf_attn_list.append(enc_slf_attn)

        if return_attn:
            return enc_output, enc_slf_attn
        return enc_output

class Decoder(nn.Module):
    """ Decoder model with multiple decoder layers """

    def __init__(self, n_vocab, vocab_size, n_position, d_k, d_v, num_layers=6, model_d=512,
                 n_head=8, ffn_d=2048, dropout=0.1):

        super().__init__()

        self.dec_emb = nn.Embedding(n_vocab, vocab_size, padding_idx=0)
        self.positional_encoding = PositionalEncoding(vocab_size, n_position)
        self.layer_stacks = nn.ModuleList([DecoderLayer(model_d, ffn_d, n_head, d_k, d_v, dropout=dropout) for _ in
                                           range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_d, eps=1e-6)

    def forward(self, x, dec_mask, enc_output, enc_mask, return_attn=False):
        """enc_output: encoder output, enc_mask: encoder mask"""

        dec_slf_attn_list, dec_enc_attn_list = [], []

        x = self.word_embedding(x)  # get word embedding for input sequence
        x += self.position_encoding  # add positional encoding
        x = self.dropout(x)  # apply dropout
        x = self.layer_norm(x)  # apply layer normalization

        for dec_layer in self.layer_stacks:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(enc_output, x, self_attn_mask=dec_mask,
                                                               dec_enc_attn_mask=enc_mask)
            if return_attn:
                dec_slf_attn_list.append(dec_slf_attn)
                dec_enc_attn_list.append(dec_enc_attn)

        if return_attn:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output

class Transformer(nn.Module):
    """ seq2seq model with self-attention """

    def __init__(self, n_enc_vocab, enc_vocab_size, n_dec_vocab, dec_vocab_size, n_position=200, d_k=64, d_v=64,
                 model_d=512, n_head=8, ffn_d=2048, n_layer=6, dropout=0.1):
        """
        Args:
            n_enc_vocab: number of vocabs in encoder sequence
            enc_vocab_size: max vocab dimension in encoder sequence
            n_dec_vocab: number of vocabs in decoder sequence
            dec_vocab_size: max vocab dimension in decoder sequence
            n_position: number of positions in position encoding
            d_k: dimension of q and k
            d_v: dimension of v
            model_d: output dimension
            n_head: number of attention heads
            ffn_d: number of inner dimensions in ffn
            n_layer: number of encoder/decoder layers
            dropout: dropout rate
        """

        super().__init__()

        # Encoder
        self.encoder = Encoder(n_enc_vocab, enc_vocab_size, n_position, d_k, d_v, num_layers=n_layer, model_d=model_d,
                               n_head=n_head, ffn_d=ffn_d, dropout=dropout)
        # Decoder
        self.decoder = Decoder(n_dec_vocab, dec_vocab_size, n_position, d_k, d_v, num_layers=n_layer, model_d=model_d,
                               n_head=n_head, ffn_d=ffn_d, dropout=dropout)
        # Linear projection
        self.linear = nn.Linear(model_d, dec_vocab_size, bias=False)
        # Softmax
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_seq, dec_seq):

        enc_mask = padding_mask(enc_seq, enc_seq) # encoder sequence padding mask
        dec_pad_mask = padding_mask(dec_seq, dec_seq) # decoder sequence padding mask
        dec_mask_sub = sequence_mask(dec_seq) # decoder subsequent positions mask
        # decoder mask for padding and subsequent positions, values that are non-zero are masked
        dec_mask = torch.gt((dec_pad_mask + dec_mask_sub), 0)
        dec_enc_mask = padding_mask(dec_seq, enc_seq) # context padding mask

        # Feed inputs into the network
        enc_output, enc_slf_attn = self.encoder(enc_seq, enc_mask, return_attn=True)
        dec_output, dec_slf_attn, dec_enc_attn = self.decoder(dec_seq, dec_mask, enc_output, dec_enc_mask,
                                                              return_attn=True)
        output = self.linear(dec_output)
        output = self.softmax(output)

        return output, enc_slf_attn, dec_slf_attn, dec_enc_attn
