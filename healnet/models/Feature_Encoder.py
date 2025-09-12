import torch
import torch.nn as nn
from torchvision import models
from einops.layers.torch import Reduce


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

        # 输出层
        # self.log_hazards_model = nn.Linear(num_output_features, 11)
        self.to_probability = nn.Sequential(
            # Reduce('b n d -> b d', 'mean'),
            nn.Linear(num_output_features, 10)
        )
        self.to_evidence_sum = nn.Sequential(
            # Reduce('b n d -> b d', 'mean'),
            nn.Linear(num_output_features, num_output_features),
            nn.Linear(num_output_features, 1),
        )
        self.depth_attn = DepthAttention(num_output_features)

    def forward(self, t2):
        t2_features = self.extract_resnet_features(t2)
        evidence_sum_logit = self.to_evidence_sum(t2_features)
        probability_logit = self.to_probability(t2_features)
        evidence_sum = torch.exp(evidence_sum_logit)
        probability = torch.softmax(probability_logit, dim=-1)
        evidence = evidence_sum * probability
        return evidence,evidence_sum, probability

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
                    
