import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.nn.parameter import Parameter

import math
import numpy as np
from thop import profile
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath 

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)

        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
                )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            input_resolution
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim
        self.convtrans = ConvTransBlock(dim//2, dim//2, dim//2, 8, 0.0, 'SW', input_resolution)

    def forward(self, x_in):
        b, h, w, c = x_in.shape
        temp = x_in.permute(0, 3, 1, 2)
        x_in = self.convtrans(temp).permute(0, 2, 3, 1)
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            input_resolution
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads, input_resolution=input_resolution),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class MST(nn.Module):
    def __init__(self, in_dim=32, out_dim=32, dim=32, stage=2, num_blocks=[2,4,4]):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        input_resolution = 256
        for i in range(stage):
            input_resolution = input_resolution // 2
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim, input_resolution=input_resolution),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage * 2, dim_stage, 3, 1, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1], input_resolution=input_resolution)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim, input_resolution=input_resolution), 
            ]))
            dim_stage //= 2
            input_resolution = input_resolution * 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, de_list):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for i, (MSAB, FeaDownSample, Fusion) in enumerate(self.encoder_layers):
            if de_list == None:
                fea = MSAB(fea)
                fea_encoder.append(fea)
                fea = FeaDownSample(fea)
            else:
                fea = Fusion(torch.cat([fea, de_list[self.stage-1-i]], dim=1))
                fea = MSAB(fea)
                fea_encoder.append(fea)
                fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        de_list = []
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)
            de_list.append(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out, de_list

## Supervised Attention Module for Cross-Phase interaction in HFIM##
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2), bias=bias, stride=1)
        self.conv2 = nn.Conv2d(n_feat, 28, kernel_size, padding=(kernel_size//2), bias=bias, stride=1)
        self.conv3 = nn.Conv2d(28, n_feat, kernel_size, padding=(kernel_size//2), bias=bias, stride=1)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class MST_PMM(nn.Module):
    def __init__(self, in_channels=28, out_channels=28, n_feat=32, stage=3):
        super(MST_PMM, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.body = MST(dim=n_feat, stage=2, num_blocks=[1,1,1])
        
        ## Cross-phase Interaction in the HFIM ##
        # self.conv_out = SAM(n_feat=32, kernel_size=3, bias=False)
    
    def forward(self, x, de_list):
        shortcut = x
        x = self.conv_in(x)
        h, de_list = self.body(x, de_list)
        # ht, xt = self.conv_out(h, shortcut)
        # return ht, xt, de_list
        return h, shortcut, de_list

nC = 28
row = 256
step = 2
col = 256 + (nC - 1) * step

class MaskGuidedMechanism(nn.Module):
    def __init__(
            self, n_feat):
        super(MaskGuidedMechanism, self).__init__()

        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift):
        # x: b,c,h,w
        [bs, nC, row, col] = mask_shift.shape
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_emb = res + mask_shift
        mask_emb[mask_emb==0] = 1e-6
        return mask_emb

# forward operator 
def A(x, M):
    z = x * M
    y = torch.zeros(x.shape[0], 1, row, col).cuda()
    for i in range(nC):
        y[:, :, :, i*step:i*step+row] += z[:, i:i+1]
    return y

# pseudo-inverse of A, A(A_pinv(y)) = y
def A_pinv(y, M):
    # scaling factors for different columns
    alpha = torch.zeros(y.shape[0], 1, 1, col)
    for i in range(nC):
        alpha[:, :, :, i*step:i*step+row] += 1
    alpha = 1 / alpha
    alpha = alpha.cuda()
    z = y * alpha
    x = []
    for i in range(nC):
        x.append(z[:, :, :, i*step:i*step+row])
    x = torch.cat(x, dim=1)
    return x * M

class MST_ZRD(nn.Module):
    def __init__(self, ):
        super(MST_ZRD, self).__init__()
        self.delta = Parameter(torch.zeros(1), requires_grad=True)
        torch.nn.init.normal_(self.delta, mean=0.1, std=0.01)
        self.mm = MaskGuidedMechanism(28)
        self.mm_pinv = MaskGuidedMechanism(28)        
    
    def forward(self, y, Phi, x_pre):
        if x_pre == None:
            Phi_pinv_learned = self.mm_pinv(Phi)
            x_init = A_pinv(y, Phi_pinv_learned)
        else:
            Phi_learned = self.mm(Phi)
            temp = y - A(x_pre, Phi_learned)
            Phi_pinv_learned = self.mm_pinv(Phi)
            x_init = x_pre + self.delta * A_pinv(temp, Phi_pinv_learned)
                
        return x_init

class RNDHRNet_share(nn.Module):
    def __init__(self, phase=10):
        super(RNDHRNet_share, self).__init__()
        
        self.phase = phase

        self.ZRD_head = MST_ZRD()

        self.PMM_head = MST_PMM(in_channels=28, out_channels=28, n_feat=32, stage=1)

        self.ZRD_tail = MST_ZRD()

        self.PMM_tail = MST_PMM(in_channels=28, out_channels=28, n_feat=32, stage=1)

        self.ZRD_body = MST_ZRD()

        self.PMM_body = MST_PMM(in_channels=28, out_channels=28, n_feat=32, stage=1)       
        
        self.fuse = nn.ModuleList([nn.Conv2d(32 + 28, 28, kernel_size=1, stride=1, padding=0) for i in range(self.phase)])

        self.interact = nn.ModuleList([SAM(n_feat=32, kernel_size=3, bias=False) for i in range(self.phase)])
        

    def forward(self, y, Phi):
        rt = None
        xt = None
        de_list = None
        y = y.unsqueeze(1)

        for i in range(self.phase):  
            if i == 0:      
                rt = self.ZRD_head(y, Phi, x_pre=None)
                h, shortcut, de_list = self.PMM_head(rt, de_list=None)
                ht, xt = self.interact[i](h, shortcut)

            elif i == self.phase - 1:
                rt = self.ZRD_tail(y, Phi, x_pre=xt)
                rt_fuse = self.fuse[i](torch.cat([rt, ht], dim=1))
                h, shortcut, de_list = self.PMM_tail(rt_fuse, de_list)   
                ht, xt = self.interact[i](h, shortcut)
            
            else:
                rt = self.ZRD_body(y, Phi, x_pre=xt)
                rt_fuse = self.fuse[i](torch.cat([rt, ht], dim=1))
                h, shortcut, de_list = self.PMM_body(rt_fuse, de_list)   
                ht, xt = self.interact[i](h, shortcut) 
        
        return xt

if __name__ == '__main__':

    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    from thop import profile
    import time
    model = DUN_MST_ETA_MM_MSKIP_SHARE(phase=10).cuda()
    input1 = torch.randn(1, 256, 310)
    input1 = input1.cuda()
    mask1 = torch.randn(1, 28, 256, 256).cuda()
    torch.cuda.synchronize()
    start = time.time()
    out = model(input1, mask1)
    torch.cuda.synchronize()
    end = time.time()
    print(end-start)
    flops, params = profile(model, inputs=(input1, mask1))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

        
