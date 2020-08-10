import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
	"""Scaled dot-product attention"""

	def __init__(self, scale, attn_dropout=0.1):
		super().__init__()
		self.scale = scale
		self.dropout = nn.Dropout(attn_dropout)

	def forward(self, q, k, v, attn_mask=None):
		"""
		Args:
			q: Queries, shape = [batch_l, query_l, query_d]
			k: Key, shape = [batch_l, key_l, key_d]
			v: Value, shape = [batch_l, value_l, value_d]
			attn_mask: masking out all values in the input of the softmax
		"""

		attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale # dot product between q and transposed k then scale

		if attn_mask is not None:
			attn = attn.masked_fill_(attn_mask, -1e9) # mask with negative infinity

		attn = self.dropout(F.softmax(attn, dim=-1)) # dropout
		output = torch.matmul(attn, v) # dot product with v

		return output, attn


class MultiHeadAttention(nn.Module):
	""" Multi-Head Attention """

	def __init__(self, n_head, model_d, d_k, d_v, dropout=0.1):
		"""
		Args:
			n_head = number of parallel attention heads
			model_d = model dimension
			d_k = dimension of k
			d_v = dimension of v
			dropout = dropout rate
		"""

		super().__init__()

		self.n_head = n_head
		self.d_k = d_k # d_k = model_d / n_head
		self.d_v = d_v

		# Linear transformation of q, k, and v, input and output dimension is the same
		self.w_qs = nn.Linear(model_d, n_head * d_k, bias=False) # d_q == d_k
		self.w_ks = nn.Linear(model_d, n_head * d_k, bias=False)
		self.w_vs = nn.Linear(model_d, n_head * d_v, bias=False)

		self.attention = ScaledDotProductAttention(d_k ** 0.5)
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(model_d, eps=1e-6)
		self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

	def forward(self, q, k, v, attn_mask=None):
		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
		batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

		residual = q

		# Linear projection for q, k, v
		# Separate different heads: [batch_l, len_q, n_head, d_v]
		q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
		k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
		v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

		# Transpose for attention dot product: [batch_l, n_head, len_q, d_v]
		q, k, v, = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

		if attn_mask is not None:
			attn_mask = attn_mask.unsqueeze(1) # For head axis broadcasting, add an extra dimension at dimension 1

		q, attn = self.attention(q, k, v, attn_mask=attn_mask) # compute attention
		q = self.dropout(q) # apply dropout

		# Transpose to move the head dimension back to [batch_l, len_q, n_head, d_v]
		# Combine the last two dimensions to concatenate the heads together
		q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
		q = self.fc(q) # Linear projection
		q += residual # add residual connection and output together

		q = self.layer_norm(q) # Layer normalization

		return q, attn


class PositionWiseFeedForward(nn.Module):
	""" Two layer feed-forward network"""

	def __init__(self, d_in, d_hid, dropout=0.1):
		"""
		Args:
			d_in: input dimension
			d_hid: output dimension
			dropout: dropout rate
		"""
		super().__init__()
		self.w_1 = nn.Linear(d_in, d_hid)
		self.w_2 = nn.Linear(d_hid, d_in) # Input and output of the network are the same
		self.layer_norm = nn.LayerNorm(d_in, eps=1e-6) # Layer Normalization
		self.dropout = nn.Dropout(dropout) # Dropout

	def forward(self, x):

		residual = x

		x = self.w_2(F.relu(self.w_1(x))) # Two linear transformations with ReLU activation in between
		x = self.dropout(x)
		x += residual

		x = self.layer_norm(x)

		return x
