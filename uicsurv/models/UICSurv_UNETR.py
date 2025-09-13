from math import pi, log
from functools import wraps
from typing import *

import torch
from torch import nn, einsum
import torch.nn.functional as F
from monai.networks.nets import ViT
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from loss import contrastive_loss


class UNETR_T2(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 img_size: Tuple[int, int, int], 
                 hidden_size: int = 64, 
                 mlp_dim: int = 64, 
                 num_heads: int = 2, 
                 pos_embed: str = "conv",
                 dropout_rate: float = 0.1):  
        super().__init__()

        self.num_layers = 6
        self.patch_size = (32,32,20)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.fc1 = nn.Linear(768*64, 768)
    def forward(self, x_in):
        x_in = x_in.permute(0,4,2,3,1)
        x, hidden_states_out = self.vit(x_in)
        # print(hidden_states_out[-1].size())
        # x =  x.view(hidden_states_out[-1].size(0), -1)
        # x = x.reshape(x.size(0), -1)
        # x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        x = x.mean(dim=1)
        # print(x.shape)
        # x = x.mean(dim=1)
        

                
        return x

class UNETR_DWI(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 img_size: Tuple[int, int, int], 
                 hidden_size: int = 64, 
                 mlp_dim: int = 64, 
                 num_heads: int = 2, 
                 pos_embed: str = "conv",
                 dropout_rate: float = 0.1): 
        super().__init__()

        self.num_layers = 6
        self.patch_size = (32,32,20)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        # self.fc1 = nn.Linear(36*768, 768)

    def forward(self, x_in):
        x_in = x_in.permute(0,4,2,3,1)
        x, hidden_states_out = self.vit(x_in)
        # print(hidden_states_out[-1].size())
        # x =  x.view(hidden_states_out[-1].size(0), -1)
        # x = x.reshape(x.size(0), -1)
        # x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # print(x.shape)
        x = x.mean(dim=1)


                
        return x




class UICSURV_U_Site_UNETR(nn.Module):
    def __init__(
        self,
        *,
        n_modalities: int,
        channel_dims: List,
        num_spatial_axes: List, 
        out_dims: int,
        depth: int = 3,
        num_freq_bands: int = 2,
        max_freq: float=10.,
        l_c: int = 128,
        l_d: int = 128,
        x_heads: int = 8,
        l_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        weight_tie_layers: bool = False,
        fourier_encode_data: bool = True,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True,
        snn: bool = True,
    ):

        
        
        super().__init__()
        assert len(channel_dims) == len(num_spatial_axes), 'input channels and input axis must be of the same length'
        assert len(num_spatial_axes) == n_modalities, 'input axis must be of the same length as the number of modalities'

        self.input_axes = num_spatial_axes
        self.input_channels=channel_dims
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.modalities = n_modalities
        self.self_per_cross_attn = self_per_cross_attn

        self.fourier_encode_data = fourier_encode_data
        self.T2_UNETR = UNETR_T2(in_channels=1, out_channels=1, img_size=(256,256,20),hidden_size=32,num_heads=2).cuda()
        self.DWI_UNETR = UNETR_DWI(in_channels=1, out_channels=1, img_size=(256,256,20),hidden_size=32,num_heads=2).cuda()


        # get fourier channels and input dims for each modality
        fourier_channels = []
        input_dims = []
        for axis in num_spatial_axes:
            fourier_channels.append((axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0)
        for f_channels, i_channels in zip(fourier_channels, channel_dims):
            input_dims.append(f_channels + i_channels)


        # initialise shared latent bottleneck
        self.latents = nn.Parameter(torch.randn(l_c, l_d)) # 128 * 128

        # modality-specific attention layers
        funcs = []
        for m in range(n_modalities):
            funcs.append(lambda m=m: PreNorm(l_d, Attention(l_d, input_dims[m], heads = x_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dims[m]))
        cross_attn_funcs = tuple(map(cache_fn, tuple(funcs)))

        get_latent_attn = lambda: PreNorm(l_d, Attention(l_d, heads = l_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_cross_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout = ff_dropout, snn = snn))
        get_latent_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout = ff_dropout, snn = snn))

        get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])


        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(get_latent_attn(**cache_args, key = block_ind))
                self_attns.append(get_latent_ff(**cache_args, key = block_ind))


            cross_attn_layers = []
            for j in range(n_modalities):
                cross_attn_layers.append(cross_attn_funcs[j](**cache_args))
                cross_attn_layers.append(get_cross_ff(**cache_args))


            self.layers.append(nn.ModuleList(
                [*cross_attn_layers, self_attns])
            )

        self.to_probability = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(l_d, out_dims)                                                                                                                                                                     
        )
        self.to_evidence_sum = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(l_d, l_d),
            nn.Linear(l_d, 1),
        )
        
        
        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(l_d, l_d),
            # nn.LayerNorm(l_d),
            nn.Linear(l_d, out_dims)
        ) if final_classifier_head else nn.Identity()




    def forward(self,
                tensors: List[Union[torch.Tensor, None]],
                mask: Optional[torch.Tensor] = None,
                return_embeddings: bool = False, 
                verbose: bool = False,
                site_labels: Optional[torch.Tensor] = None,
                target : Optional[torch.Tensor] = None,
                D_batch : Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None,
                ):
        
        
        missing_idx = [i for i, t in enumerate(tensors) if t is None]
        if verbose: 
            print(f"Missing modalities indices: {missing_idx}")
        # print(len(tensors))
        for i in range(len(tensors)):
            if i in missing_idx: 
                continue
            else: 
                # print(f"Modality {i+1} shape: {tensors[i].shape}")
                if i == 0:
                # if tensors[i].shape[2] == 256 :
                    # print("T2_UNETR")
                    # print("Sliding window 3D ConvNet")
                    # FeatureExtractor = SlidingWindow3DConvNet().cuda()
                    
                    feature= self.T2_UNETR(tensors[i]) # batch * 768
                    tensors[i] = feature
                    # 添加一个维度
                    tensors[i] = tensors[i].unsqueeze(1)
                elif i == 1:
                # elif tensors[i].shape[2] == 126:
                    # print("DWI_UNETR")

                    feature= self.DWI_UNETR(tensors[i])  # batch * 768
                    tensors[i] = feature
                    # 添加一个维度
                    tensors[i] = tensors[i].unsqueeze(1)
                     
                

                data = tensors[i]
                # sanity checks
                b, *axis, _, device, dtype = *data.shape, data.device, data.dtype # batch_size,1,13,gpu,float32/ batch,1,768,gpu,float32
                assert len(axis) == self.input_axes[i], (f'input data for modality {i+1} must hav'
                                                            f' the same number of axis as the input axis parameter')

            # fourier encode for each modality
            if self.fourier_encode_data:
                axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
                pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
                enc_pos = repeat(enc_pos, '... -> b ...', b = b)
                data = torch.cat((data, enc_pos), dim = -1) # batch,1,18/ batch,20,250,250,1+15 位置编码后的数据
                

            # concat and flatten axis for each modality
            data = rearrange(data, 'b ... d -> b (...) d') # batch,1,18/ batch,1,773
            tensors[i] = data


        x = repeat(self.latents, 'n d -> b n d', b = b) # note: batch dim should be identical across modalities batch * 128 * 128
        total_contrastive_loss = 0
        for layer_idx, layer in enumerate(self.layers):
            for i in range(self.modalities):
                if i in missing_idx: 
                    if verbose: 
                        print(f"Skipping update in fusion layer {layer_idx + 1} for missing modality {i+1}")
                    continue
                cross_attn=layer[i*2]
                cross_ff = layer[(i*2)+1]
                # try:
                x = cross_attn(x, context = tensors[i], mask = mask) + x
                x =  cross_ff(x) + x
                # except:
                #     pass
                if self.self_per_cross_attn > 0:
                    self_attn, self_ff = layer[-1]
                    x = self_attn(x) + x
                    x = self_ff(x) + x
            if site_labels is not None:
                site_labels = site_labels.cuda()
                target = target.cuda()
                D_batch = D_batch.cuda()
                uncertainty = uncertainty.cuda()
                c_loss = contrastive_loss(x, site_labels,target, D_batch, uncertainty)
                total_contrastive_loss += c_loss

        if return_embeddings:
            return x
        
        

        # logits = self.to_logits(x)
        evidence_sum_logit = self.to_evidence_sum(x)
        
        probability_logit = self.to_probability(x)
        evidence_sum = torch.exp(evidence_sum_logit) + 1.0
        probability = torch.softmax(probability_logit, dim=-1)
        evidence = evidence_sum * probability
        # evidence = F.softplus(logits)
        # evidence = torch.exp(logits)
        # print(x)
        if site_labels is not None:
            return evidence, x, evidence_sum, probability,total_contrastive_loss
        return evidence,evidence_sum, probability

    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Helper function which returns all attention weights for all attention layers in the model
        Returns:
            all_attn_weights: list of attention weights for each attention layer
        """
        all_attn_weights = []
        for module in self.modules():
            if isinstance(module, Attention):
                all_attn_weights.append(module.attn_weights)
        return all_attn_weights



# HELPERS/UTILS
"""
Helper class implementations based on: https://github.com/lucidrains/perceiver-pytorch
"""


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class SELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.selu(gates)

class RELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.relu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., snn: bool = False):
        super().__init__()
        activation = SELU() if snn else GELU()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            activation,
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def temperature_softmax(logits, temperature=1.0, dim=-1):
    """
    Temperature scaled softmax
    Args:
        logits:
        temperature:
        dim:

    Returns:
    """
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=dim)



class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        # add leaky relu
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.LeakyReLU(negative_slope=1e-2)
        )

        self.attn_weights = None
        # self._init_weights()

    def _init_weights(self):
    # Use He initialization for Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # Initialize bias to zero if there's any
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        # attn = sim.softmax(dim = -1)
        attn = temperature_softmax(sim, temperature=0.5, dim=-1)
        self.attn_weights = attn
        attn = self.dropout(attn)


        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)