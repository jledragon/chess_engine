"""
Copyright Kaedim Limited 2021.
"""

from torch import nn, einsum
from .DepthWiseConv2d import DepthWiseConv2d
from einops import rearrange


class Attention(nn.Module):
	"""
	CvT's attention class.

	"""
	def __init__(
		self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64,
		dropout=0.):
		"""
		Parameters
		----------
		dim : int
			q/kv input dimension.
		proj_kernel : int
			q/kv kernel size.
		kv_proj_stride : int
			kv stride.
		heads : int
			Number of attention heads.
		dim_head : int
			dim_head : inner_dim/scale parameter.
		dropout : float
			Dropout. Default = 0.0.

		"""
		super().__init__()
		inner_dim = dim_head * heads
		padding = proj_kernel // 2
		self.heads = heads
		self.scale = dim_head ** -0.5

		self.attend = nn.Softmax(dim=-1)

		self.to_q = DepthWiseConv2d(
			dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
		self.to_kv = DepthWiseConv2d(
			dim, inner_dim * 2, proj_kernel, padding=padding,
			stride=kv_proj_stride, bias=False)

		self.to_out = nn.Sequential(
			nn.Conv2d(inner_dim, dim, 1),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		"""
		Performs a forward pass through the attention mechanism.

		Parameters
		----------
		x : tensor
			Intermediate input.

		Returns
		-------
		The attention outputs.

		"""
		shape = x.shape
		_, _, _, y, h = *shape, self.heads
		q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
		q, k, v = map(lambda t: rearrange(
			t, 'b (h d) x y -> (b h) (x y) d', h=h), (q, k, v))

		dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

		attn = self.attend(dots)

		out = einsum('b i j, b j d -> b i d', attn, v)
		out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, y=y)
		return self.to_out(out)
