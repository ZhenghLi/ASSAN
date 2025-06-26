# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable, Any
from collections import OrderedDict
from functools import partial
from einops import rearrange, repeat
import numbers

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def A_scan(x: torch.Tensor):
        B, C, H, W = x.shape
        xs = x.new_empty((B * W, 2, C, H))
        x = rearrange(x, 'b c h w -> (b w) c h').contiguous()
        xs[:, 0] = x
        xs[:, 1] = torch.flip(x, dims=[-1])
        return xs

def A_merge(a_scan: torch.Tensor):
    a = a_scan[:, 0] + a_scan[:, 1].flip(dims=[-2]) # filp H
    B, D, H, W = a.shape
    return a.view(B, D, -1)

def selective_a_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    softmax_version=False,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    assert K == 2
    xs = A_scan(x)
    
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B*W, D*2, H).to(torch.float)
    dts = dts.contiguous().view(B*W, D*2, H).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    
    ys: torch.Tensor = selective_scan_fn(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, W, 2, D, H).permute(0, 2, 3, 4, 1).contiguous()
    
    y: torch.Tensor = A_merge(ys)

    if softmax_version:
        y = y.softmax(dim=-1)
        if to_dtype:
            y = y.to(x.dtype)
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
    else:
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm(y)
        if to_dtype:
            y = y.to(x.dtype)
    return y

# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class AlineSS(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        softmax_version=False,
        forward_type="v2",
        # ======================
        **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.softmax_version = softmax_version
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv

        # forward_type =======================================
        self.forward_core = dict(
            # v0=self.forward_corev0,
            # v0_seq=self.forward_corev0_seq,
            # v1=self.forward_corev2,
            v2=self.forward_corev2,
            # share_ssm=self.forward_corev0_share_ssm,
            # share_a=self.forward_corev0_share_a,
        ).get(forward_type, self.forward_corev2)
        self.K = 2 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(d_inner)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner))) 

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False):
            nrows = 1
            if not channel_first:
                x = x.permute(0, 3, 1, 2).contiguous()
            if self.ssm_low_rank:
                x = self.in_rank(x)
            x = selective_a_scan(
                x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
                self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version, 
                nrows=nrows, delta_softplus=True,
            )
            if self.ssm_low_rank:
                x = self.out_rank(x)
            return x
        
    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            if not self.softmax_version:
                z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x)) # (b, d, h, w)
            y = self.forward_core(x, channel_first=True)
            y = y * z
        else:
            if self.softmax_version:
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            y = self.forward_core(x, channel_first=False)
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

class AlineSSM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=False,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        softmax_version=False,
        forward_type="v2",
        # =============================
    ):
        super().__init__()
        # self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = AlineSS(
            d_model=hidden_dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            simple_init=ssm_simple_init,
            # ==========================
            softmax_version=softmax_version,
            forward_type=forward_type,
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input)))
        return x

class BlineAttention(nn.Module):
    def __init__(self, dim, num_heads, width, bias=False):
        super(BlineAttention, self).__init__()
        self.num_heads = num_heads
        self.w = width
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # create the relative bias for B-line attention by setting Wh in Swin attention to 1

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * width - 1), num_heads))  # 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(1)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += 1 - 1  # shift to start from 0
        relative_coords[:, :, 1] += width - 1
        relative_coords[:, :, 0] *= 2 * width - 1
        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_depth = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_pre = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.SiLU()
        )

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        b,c,h,w = x.shape

        x_gate = self.gate(x)
        
        x = self.qkv(x)
        qkv = self.qkv_depth(x)
        q,k,v = qkv.chunk(3, dim=1)

        q = q * self.scale
        
        q = rearrange(q, 'b (head c) h w -> b h head w c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b h head w c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b h head w c', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.w, self.w, -1)  # Ww,Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Ww, Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = (attn @ v) 
        
        out = rearrange(out, 'b h head w c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_pre(out)

        out = out * x_gate

        out = self.project_out(out)

        return out
    
##########################################################################
class LEFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, kernel_size):
        super(LEFN, self).__init__()
        hidden_dim = dim*ffn_expansion_factor
        self.convin = nn.Conv2d(dim, hidden_dim, 1)
        self.br1d = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (kernel_size, 1), 1, (kernel_size//2, 0), groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, (1, kernel_size), 1, (0, kernel_size//2), groups=hidden_dim)
        )
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.convout = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x:torch.Tensor):
        x = self.convin(x)
        x = self.act1(x)
        x = self.br1d(x)
        x = self.act2(x)
        x = self.convout(x)
        return x

# ============================
# including B-line Gated Attention and the LEFN
class BlineTransformer(nn.Module):
    def __init__(self, dim, num_heads, width, ffn_expansion_factor, kernel_size, bias=False):
        super(BlineTransformer, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn_b = BlineAttention(dim, num_heads, width, bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = LEFN(dim, ffn_expansion_factor, kernel_size)

    def forward(self, x):
        x = x + self.attn_b(self.norm1(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x + self.ffn(self.norm2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class ASSAL(nn.Module):
    """ Alternative SSM Attention Layer.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        width (int): Width of the input and the window size.
        ffn_expansion_factor (float): Ratio of lefn hidden dim to embedding dim.
        kernel_size (int): Size of the conv1ds in lefn.
    """

    def __init__(self, dim, depth, num_heads=6, width=64, ffn_expansion_factor=2, ffn_kernel_size=11, bias=False):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(AlineSSM(hidden_dim=dim))
            self.blocks.append(BlineTransformer(dim=dim, num_heads=num_heads, width=width, ffn_expansion_factor=ffn_expansion_factor, kernel_size=ffn_kernel_size, bias=bias))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ASSAG(nn.Module):
    """Alternative SSM Attention Group.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        width (int): Width of the input and the window size.
        ffn_expansion_factor (float): Ratio of lefn hidden dim to embedding dim.
        ffn_kernel_size (int): Size of the conv1ds in lefn.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, depth, num_heads=6, width=64, ffn_expansion_factor=2, ffn_kernel_size=11, bias=False, resi_connection='1conv'):
        super(ASSAG, self).__init__()

        self.dim = dim

        self.residual_group = ASSAL(dim, depth, num_heads=num_heads, width=width, ffn_expansion_factor=ffn_expansion_factor, ffn_kernel_size=ffn_kernel_size, bias=bias)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x):
        return self.conv(self.residual_group(x)) + x


class ASSAN(nn.Module):
    """Alternative SSM Attention Network.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        sparsity (int): Sparsity factor
        width (int): Width of the input and the window size.
        dim (int): Number of input channels.
        depths (list of int): Number of blocks.
        num_heads (list of int): Number of attention heads.
        ffn_expansion_factor (float): Ratio of lefn hidden dim to embedding dim.
        ffn_kernel_size (int): Size of the conv1ds in lefn.
        resi_connection: The convolutional block before residual connection.
    """
    def __init__(self, in_chans=4, out_chans=1, sparsity=2, width=64,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 ffn_expansion_factor=2, ffn_kernel_size=11,
                 bias=False, resi_connection='1conv'):
        super(ASSAN, self).__init__()
        num_in_ch = in_chans
        num_out_ch = out_chans
        num_feat = 64
        self.sparsity = sparsity

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1, bias=False)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim

        # build Alternative SSM Attention Group (ASSAG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ASSAG(embed_dim, depths[i_layer], num_heads=num_heads[i_layer], width=width, ffn_expansion_factor=ffn_expansion_factor, ffn_kernel_size=ffn_kernel_size, bias=bias, resi_connection=resi_connection)
            self.layers.append(layer)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, B-line shuffle reconstruction ################################
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                    nn.LeakyReLU(inplace=True))
        assert math.log(self.sparsity, 2) % 1 == 0
        self.upsample_convs = nn.ModuleList([nn.Conv2d(num_feat, 2 * num_feat, 3, 1, 1) for i in range(int(math.log(self.sparsity, 2)))])
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        # B-line shuffle
        for i in range(len(self.upsample_convs)):
            x = self.upsample_convs[i](x)
            n, c, h, w = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()
            x = x.reshape(n, w*2, c//2, h)
            x = x.permute(0, 2, 3, 1).contiguous()
        x = self.conv_last(x)
        return x


if __name__ == '__main__':
    sparsity = 4
    height = 512
    width = 64

    model = ASSAN(sparsity=sparsity, in_chans=4, depths=[6, 6, 6, 6],
                embed_dim=60, num_heads=[6, 6, 6, 6]).cuda()
    print(model)
    # print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 4, height, width)).cuda()
    x = model(x)
    print(x.shape)
