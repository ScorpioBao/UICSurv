
from math import pi, log
from functools import wraps
from typing import *

import torch
from torch import nn, einsum
import torch.nn.functional as F
from monai.networks.nets import ViT
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import torchvision.models as models
from loss import contrastive_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

class DepthAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        # x形状: (batch_size, depth, feature_dim)
        attn_weights = self.attention(x)  # (batch_size, depth, 1)
        return torch.sum(x * attn_weights, dim=1)  # (batch_size, feature_dim)

class MultiModalModelWithResNet18(nn.Module):
    def __init__(self, num_output_features=64, frozen_layers=-1, depth_dim=20, nhead=2, num_layers=1):
        super(MultiModalModelWithResNet18, self).__init__()

        # 加载预训练的 ResNet18 模型
        self.resnet18 = models.resnet18(pretrained=True)
        self.freeze_resnet_layers(frozen_layers)

        # # 修改 ResNet18 的第一层卷积层以接受 3 通道输入
        # self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 移除 ResNet18 的最后一层全连接层（分类头）
        self.resnet18.fc = nn.Identity()  # 移除分类头，保留特征提取部分

        # 特征压缩层
        self.fc_conv = nn.Linear(512, num_output_features)  # ResNet18 的特征维度是 512
        self.bn = nn.BatchNorm1d(num_output_features)
        self.dropout = nn.Dropout(0.5)

        # Transformer 编码器用于深度方向的注意力
        self.depth_dim = depth_dim
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_output_features,  # 输入特征维度
            nhead=nhead,  # 注意力头数
            dim_feedforward=128,  # 前馈网络的隐藏层维度
            dropout=0.3,  # Dropout 概率
            activation="relu"  # 激活函数
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer,
            num_layers=num_layers  # Transformer 编码器层数
        )

       
        self.depth_attn = DepthAttention(num_output_features)

    def forward(self, t2):
        t2_features = self.extract_resnet_features(t2)
        
        return t2_features

    def extract_resnet_features(self, x):
        x = x.squeeze()  # 将输入张量的形状从 (batch_size, depth, height, width,1 ) 转换为 (batch_size, depth, height, width)
        # print(x.shape)
        if x.dim() == 4:
            batch_size, depth, height, width = x.size()
            # x = x.permute(0, 3, 1, 2)  # 将深度维度移到第二维，形状为 (batch_size, depth, height, width)
            x = x.reshape(batch_size * depth, 1, height, width)  # 将深度维度与批次维度合并，形状为 (batch_size * depth, 1, height, width)
        else:
            depth, height, width = x.size()
            batch_size = 1
            # x = x.permute(0, 3, 1, 2)  # 将深度维度移到第二维，形状为 (batch_size, depth, height, width)
            x = x.reshape(depth, 1, height, width)  # 将深度维度与批次维度合并，形状为 (batch_size * depth, 1, height, width)
        
        # 将单通道数据扩展为 3 通道（伪 RGB）
        x = self.convert_to_rgb(x)  # 输出形状为 (batch_size * depth, 3, height, width)

        # 提取特征
        features = self.resnet18(x)  # 输出形状为 (batch_size * depth, 512)

        # 特征压缩
        features = self.fc_conv(features)  # 将 512 维特征压缩到 num_output_features 维
        features = self.bn(features)
        features = self.dropout(features)
        features = torch.nn.functional.relu(features)

        # 将特征重塑为 (batch_size, depth, num_output_features)
        features = features.view(batch_size, depth, -1)

        # 使用 Transformer 编码器捕捉深度方向的依赖关系
        features = features.permute(1, 0, 2)  # Transformer 需要 (seq_len, batch_size, feature_dim)
        features = self.transformer_encoder(features)
        features = features.permute(1, 0, 2)  # 恢复为 (batch_size, seq_len, feature_dim)
        features = self.depth_attn(features)  # (batch_size, num_output_features)
        return features

    def convert_to_rgb(self, x):
        """
        将单通道数据扩展为 3 通道（伪 RGB）
        输入形状: (batch_size * depth, 1, height, width)
        输出形状: (batch_size * depth, 3, height, width)
        """
        return torch.cat([x, x, x], dim=1)  # 复制单通道数据 3 次

    def freeze_resnet_layers(self, frozen_layers=-1):
        if frozen_layers == 0:
            for param in self.resnet18.parameters():
                param.requires_grad = True
        elif frozen_layers == -1:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        else:
            layers_to_freeze = list(self.resnet18.children())[:frozen_layers]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
            for layer in list(self.resnet18.children())[frozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True   


class HealNet_U_Site_Resnet(nn.Module):
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
        """
        Network architecture for easy-to-use multimodal fusion for any number and type of modalities.
        
        The input for each modality should be of shape ``(b, (*spatial_dims) c)``, where ``c`` corresponds to the dimensions 
        where positional encoding does not matter (e.g., color channels, set-based features, or tabular features). 

        Args:
            n_modalities (int): Maximum number of modalities for forward pass. Note that fewer modalities can be passed
                if modalities for individual samples are missing (see ``.forward()``)
            channel_dims (List[int]): Number of channels or tokens for each modality. Length must match ``n_modalities``. 
                The channel_dims are non-spatial dimensions where positional encoding is not required. 
            num_spatial_axes (List[int]): Spatial axes for each modality.The each spatial axis will be assigned positional 
                encodings, so that ``num_spatial_axis`` is 2 for 2D images, 3 for Video/3D images. 
            out_dims (int): Output shape of task-specific head. Forward pass returns logits of this shape. 
            num_freq_bands (int, optional): Number of frequency bands for positional encodings. Defaults to 2.
            max_freq (float, optional): Maximum frequency for positional encoding. Defaults to 10.
            l_c (int, optional): Number of channels for latent bottleneck array (akin to a "learned query array"). Defaults to 128.
            l_d (int, optional): Dimensions for latent bottleneck. Defaults to 128.
            x_heads (int, optional): Number of heads for cross attention. Defaults to 8.
            l_heads (int, optional): Number of heads for latent attention. Defaults to 8.
            cross_dim_head (int, optional): Dimension of each cross attention head. Defaults to 64.
            latent_dim_head (int, optional): Dimension of each latent attention head. Defaults to 64.
            attn_dropout (float, optional): Dropout rate for attention layers. Defaults to 0.
            ff_dropout (float, optional): Dropout rate for feed-forward layers. Defaults to 0.
            weight_tie_layers (bool, optional): False for weight sharing between fusion layers, True for specific 
                weights for each layer. Note that the number of parameters will multiply by ``depth`` if True. 
                Defaults to False.
            fourier_encode_data (bool, optional): Whether to use positional encoding. Recommended if meaningful spatial 
                spatial structure should be preserved. Defaults to True.
            self_per_cross_attn (int, optional): Number of self-attention layers per cross-attention layer. Defaults to 1.
            final_classifier_head (bool, optional): Whether to include a final classifier head. Defaults to True.
            snn (bool, optional): Whether to use a self-normalizing network. Defaults to True.

        Example: 
        
            ```python
            from healnet import HealNet
            from healnet.etl import MMDataset
            import torch
            import einops

            # synthetic data example
            n = 100 # number of samples
            b = 4 # batch size
            img_c = 3 # image channels
            tab_c = 1 # tabular channels
            tab_d = 2000 # tabular features
            # 2D dims
            h = 224 # image height
            w = 224 # image width
            # 3d dim
            d = 12

            tab_tensor = torch.rand(size=(n, tab_c, tab_d)) 
            img_tensor_2d = torch.rand(size=(n, h, w, img_c)) # h w c
            img_tensor_3d = torch.rand(size=(n, d, h, w, img_c)) # d h w c
            dataset = MMDataset([tab_tensor, img_tensor_2d, img_tensor_3d])

            [tab_sample, img_sample_2d, img_sample_3d] = dataset[0]

            # batch dim for illustration purposes
            tab_sample = einops.repeat(tab_sample, 'c d -> b c d', b=1) # spatial axis: None (pass as 1)
            img_sample_2d = einops.repeat(img_sample_2d, 'h w c -> b h w c', b=1) # spatial axes: h w
            img_sample_3d = einops.repeat(img_sample_3d, 'd h w c -> b d h w c', b=1) # spatial axes: d h w

            tensors = [tab_sample, img_sample_2d, img_sample_3d]


            model = HealNet(
                        n_modalities=3, 
                        channel_dims=[2000, 3, 3], # (2000, 3, 3) number of channels/tokens per modality
                        num_spatial_axes=[1, 2, 3], # (1, 2, 3) number of spatial axes (will be positionally encoded to preserve spatial information)
                        out_dims = 4
                    )

            # example forward pass
            logits = model(tensors)
            ```
        """
        
        
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
        self.T2_UNETR = MultiModalModelWithResNet18(num_output_features=128, frozen_layers=-1, depth_dim=20, nhead=2, num_layers=1).cuda()
        self.DWI_UNETR = MultiModalModelWithResNet18(num_output_features=128, frozen_layers=-1, depth_dim=20, nhead=2, num_layers=1).cuda()

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
            nn.Linear(l_d, 1)
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
        for i in range(len(tensors)):
            if i in missing_idx: 
                continue
            else: 
                if i == 1:
                    # print("T2_UNETR")
                    # print("Sliding window 3D ConvNet")
                    # FeatureExtractor = SlidingWindow3DConvNet().cuda()
                    # print(tensors[i].shape)
                    # tensors[i] = tensors[i].permute(0,4,2,3,1) #batch 1 256 256 20
                    # tensors[i] = tensors[i].permute(0,1,4,2,3)
                    ## 去掉多余的维度
                    # tensors[i] = tensors[i].squeeze(1)
                    # print(tensors[i].shape)
                    feature= self.T2_UNETR(tensors[i])
                    tensors[i] = feature # 512
                    # 添加一个维度
                    tensors[i] = tensors[i].unsqueeze(1)
                elif i == 2:
                    # print("DWI_UNETR")
                    # tensors[i] = tensors[i].permute(0,4,2,3,1)
                    # tensors[i] = tensors[i].permute(0,1,4,2,3)
                    # 去掉多余的维度
                    # tensors[i] = tensors[i].squeeze(1)
                    feature= self.DWI_UNETR(tensors[i])
                    # print(feature.shape)
                    tensors[i] = feature
                    # 添加一个维度
                    tensors[i] = tensors[i].unsqueeze(1)
                    
                    
                

                data = tensors[i]
                # sanity checks
                b, *axis, _, device, dtype = *data.shape, data.device, data.dtype # 30,1,gpu,float32/30,[20,250,250],gpu,float32
                assert len(axis) == self.input_axes[i], (f'input data for modality {i+1} must hav'
                                                            f' the same number of axis as the input axis parameter')

            # fourier encode for each modality
            if self.fourier_encode_data:
                axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
                pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
                enc_pos = repeat(enc_pos, '... -> b ...', b = b)
                data = torch.cat((data, enc_pos), dim = -1) # 30,1,15+5/30,20,250,250,1+15 位置编码后的数据
                

            # concat and flatten axis for each modality
            data = rearrange(data, 'b ... d -> b (...) d') # 30,1,20/ 30,250*250,16
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
        evidence_sum = torch.exp(evidence_sum_logit)
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