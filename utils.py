import torch
import torch.nn as nn
import numpy as np

def padding_mask(seq_q, seq_k): # mask all padding in self-attention
    """
    Args:
        seq_q = q sequence, shape = [batch_size, len_q]
        seq_k = k sequence, shape = [batch_size, len_k]
    """
    len_q = seq_q.size(1) # get the length of seq_q
    # PAD is 0
    pad_mask = seq_k.eq(0) # Obtain tensor where PAD = 0 in seq_k are labelled as True
    # reshape padding to [batch_size, len_q, len_k] as the dot product of q and k is of the same shape
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
    return pad_mask # True are represented as a value of 1, while False is 0

def sequence_mask(seq): # mask all decoder inputs after the current position to prevent leftward information flow
    batch_size, seq_len = seq.size()
    # mask all values above the main diagonal (inclusive)
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1) # reshape to [batch_size, seq_len, seq_len]
    return mask

def PositionalEncoding(model_d, n_position):
    """
    Args:
        model_d: model dimension
        n_position = number of positions
    """
    positional_encoding = np.array([pos / np.power(10000, 2 * (i // 2) / model_d) for i in range(model_d)
                                        for pos in range(n_position)])
    # Use sine and cosine function for different frequencies (sine for odd dimensions, cosine for even)
    positional_encoding[:, 0::2] = np.sin(positional_encoding[:, 0::2])  # dim 2i, odd
    positional_encoding[:, 1::2] = np.cos(positional_encoding[:, 1::2])  # dim 2i+1, even
    return torch.FloatTensor(positional_encoding).unsqueeze(0)


class PositionalEncoding(nn.Module):

    def __init__(self, model_d, n_position): # n_position = maximum length of the sentence
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, model_d))

    def _get_sinusoid_encoding_table(self, n_position, model_d):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (dim // 2) / model_d) for dim in range(model_d)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

