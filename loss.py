import torch
import torch.nn.functional as F
from surv_tool import pad_col




def deephit_loss(neural_net_output, Y_batch, D_batch):
    """
    DeepHit 负对数似然损失函数的实现，加入删失和未删失数据的不均衡加权和正则项。
    
    参数:
    - neural_net_output: Tensor, 神经网络的输出 (batch_size, num_time_intervals)，表示每个时间段的概率
    - Y_batch: Tensor, 存活时间（或时间到截尾点），形状为 (batch_size,)
    - D_batch: Tensor, 事件发生标记 (1 表示事件发生, 0 表示截尾)，形状为 (batch_size,)
    - epochs: int, 训练的 epoch 数
    - censored_weight: float, 删失样本的权重，默认为 0.5
    - uncensored_weight: float, 未删失样本的权重，默认为 0.5
    - reg_weight: float, 正则化项的权重，默认为 0.1

    返回:
    - loss: Tensor, 损失值
    """

    batch_size, num_time_intervals = neural_net_output.shape
    preds = neural_net_output 
    probabilities = torch.softmax(pad_col(preds), 1)  # PMF

    # probabilities = F.softmax(neural_net_output, dim=1)

    # 索引时间间隔，假设 Y_batch 的值为 [0, num_time_intervals-1] 之间的整数
    Y_batch = Y_batch.long()

    # 损失的初始化
    loss = torch.tensor(0.0, dtype=torch.float, device=neural_net_output.device)
    

    for i in range(batch_size):
        y_i = Y_batch[i]
        d_i = D_batch[i]
        # 对应发生事件的部分 (Delta_i == 1)
        if d_i == 1:
            loss -= torch.log(probabilities[i, y_i] + 1e-10)  # 加权
        else:
            # 累加所有的生存概率，从 y_i + 1 开始到 L
            # sum_survival = torch.sum(probabilities[i, y_i + 1:])
            sum_survival = 1 - torch.sum(probabilities[i,0:y_i+1])
            loss -= torch.log(sum_survival + 1e-10)  # 加权


    # 总损失包括：负对数似然损失 + 正则化项
    loss = (loss / batch_size) 

    return loss


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device =device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum()
        + torch.lgamma(ones).sum()
        - torch.lgamma(ones.sum())
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum()
    )
    kl = first_term + second_term
    return kl

# def ra_reg(probability):
#     ra = probability * torch.log(1/probability)
#     return ra

def deephit_loss_uncertainty(neural_net_output, Y_batch, D_batch, e_sum, pro,epochs):
    """
    DeepHit 负对数似然损失函数的实现，加入删失和未删失数据的不均衡加权和正则项。
    
    参数:
    - neural_net_output: Tensor, 神经网络的输出 (batch_size, num_time_intervals)，表示每个时间段的概率
    - Y_batch: Tensor, 存活时间（或时间到截尾点），形状为 (batch_size,)
    - D_batch: Tensor, 事件发生标记 (1 表示事件发生, 0 表示截尾)，形状为 (batch_size,)
    - epochs: int, 训练的 epoch 数
    - censored_weight: float, 删失样本的权重，默认为 0.5
    - uncensored_weight: float, 未删失样本的权重，默认为 0.5
    - reg_weight: float, 正则化项的权重，默认为 0.1

    返回:
    - loss: Tensor, 损失值
    """

    batch_size, num_time_intervals = neural_net_output.shape
    neural_net_output = pad_col(neural_net_output, val=1, where='end')  # 增加一个时间段，用于计算 KL 散度

    # 获得每个时间段的概率，使得输出是一个有效的概率分布
    S = torch.sum(neural_net_output, dim=1, keepdim=True)
    probabilities = neural_net_output / S

    Y_batch = Y_batch.long()

    
    # 损失的初始化
    loss = torch.tensor(0.0, dtype=torch.float, device=neural_net_output.device)
    
    # 退火系数
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epochs / 10, dtype=torch.float32),
    )
    
    # initial_sigma = 10
    # sigma = torch.max(
    #     torch.tensor(1.0, dtype=torch.float32),
    #     torch.tensor(initial_sigma - epochs, dtype=torch.float32),
    # )
    
    

    for i in range(batch_size):
        y_i = Y_batch[i]
        d_i = D_batch[i]
        
        # 对应发生事件的部分 (Delta_i == 1)
        # 对应发生事件的部分 (Delta_i == 1)
        if d_i == 1:
            loss -= torch.log(probabilities[i, y_i] + 1e-10)  # 加权
            
            # for t in range(y_i + 1, num_time_intervals):
            #     loss -= uncensored_weight * torch.log(1 - probabilities[i, t] + 1e-10)  
            pro = pro.detach()
            #p_gt = pro[i, y_i]
            p_gt = 1 - pro[i, 0:y_i].sum()
            kl_div = annealing_coef*(e_sum[i]*(1-p_gt)).squeeze()
            loss += 1.0 * kl_div # 加入KL散度正则项
        
        # 对应被截尾的部分 (Delta_i == 0)
        else:
            pro = pro.detach()
            # p_gt = pro[i,y_i+1:].sum()
            p_gt = 1 - pro[i, 0:y_i+1].sum()
            kl_div = annealing_coef*(e_sum[i]*(1-p_gt)).squeeze()
            
            loss +=  1.0 * kl_div  # 加入KL散度正则项
            
            sum_survival = 1 - torch.sum(probabilities[i,0:y_i+1])
            loss -= torch.log(sum_survival + 1e-10)  # 加权


    # 总损失包括：负对数似然损失 + 正则化项
    loss = (loss / batch_size) 

    return loss




def contrastive_loss(site_logits, site, target, D_batch, uncertainty, temperature=0.1, sigma=3):
    """
    完善版不确定性引导的多站点对比损失 (InfoNCE 版本)

    参数：
    site_logits: (batch_size, feat_dim) - 样本特征 (已经过平均池化或其他处理)
    site: (batch_size,) - 站点标签
    target: (batch_size,) - 生存时间
    D_batch: (batch_size,) - 事件标签（1=发生，0=删失）
    uncertainty: (batch_size,) - 不确定性权重 (如果模型不提供，可以传入全1向量)
    temperature: 温度系数
    sigma: 时间差权重参数 (用于负样本权重)

    返回：
    loss: 对比损失值
    """
    device = site_logits.device
    batch_size = site_logits.size(0)

    # 特征归一化
    site_logits = site_logits.mean(dim=1)
    features = F.normalize(site_logits, p=2, dim=1)
    sim_matrix = torch.matmul(features, features.T) / temperature  # (batch, batch)

    ####################
    # 正样本对构建逻辑 #
    ####################
    # 条件：不同站点、事件都发生、生存时间相近 (引入 tolerance)
    time_diff = torch.abs(target.view(-1, 1) - target.view(1, -1))
    tolerance = 5  # 允许的生存时间差异 (根据数据集调整)
    same_time = time_diff < tolerance
    # same_time = target.view(-1, 1) == target.view(1, -1)
    diff_site = site.view(-1, 1) != site.view(1, -1)
    both_event = (D_batch.view(-1, 1) == 1) & (D_batch.view(1, -1) == 1)
    pos_mask = diff_site & same_time & both_event
    # print("pos_mask:", pos_mask)

    ####################
    # 负样本对构建逻辑 #
    ####################
    # 事件发生的负样本条件
    # Case1: 两样本都发生事件但时间不同 (大于 tolerance)
    both_event_mask = (D_batch.view(-1, 1) == 1) & (D_batch.view(1, -1) == 1)
    case1_mask = both_event_mask & (time_diff >= tolerance)  # 时间差异大于 10 天
    # case1_mask = both_event_mask & (target.view(-1, 1) != target.view(1, -1))

    # Case2: 一个事件发生一个删失，且删失时间更大
    case2a_mask = (D_batch.view(-1, 1) == 1) & (D_batch.view(1, -1) == 0) & (target.view(1, -1) > target.view(-1, 1))
    case2b_mask = (D_batch.view(-1, 1) == 0) & (D_batch.view(1, -1) == 1) & (target.view(-1, 1) > target.view(1, -1))
    case2_mask = case2a_mask | case2b_mask

    # 综合负样本条件
    neg_mask = (~pos_mask) & (case1_mask | case2_mask)
    # print("neg_mask:", neg_mask)

    ####################
    # 权重计算逻辑 (可选) #
    ####################
    # 负样本权重计算 (基于时间差)
    neg_weights = 1 - torch.exp(-time_diff / 2 * sigma**2)

    # 正样本不确定性权重 (更合理的方式: 平均不确定性)
    pos_weights =((1- uncertainty.view(-1, 1))+ (1-uncertainty.view(1, -1))) / 2
    #pos_weights = (uncertainty.view(-1, 1) + uncertainty.view(1, -1)) / 2


    ####################
    # 损失计算逻辑 (InfoNCE) #
    ####################
    exp_sim = torch.exp(sim_matrix)
    # print("exp_sim:", exp_sim)

    loss_total = 0.0
    num_pos_pairs = 0

    for i in range(batch_size):
        pos_indices = torch.where(pos_mask[i])[0]
        neg_indices = torch.where(neg_mask[i])[0]

        if len(pos_indices) > 0:
            for j in pos_indices:
                # InfoNCE loss for the pair (i, j)
                numerator = exp_sim[i, j] * pos_weights[i,j] #正样本也加权重
                denominator = (exp_sim[i, :] * neg_mask[i].float() * neg_weights[i,:]).sum() + numerator  # 负样本加权求和 + 正样本
                loss = -torch.log((numerator) / (denominator))
                loss_total += loss
                num_pos_pairs += 1

    if num_pos_pairs > 0:
        # print("yes")
        loss = loss_total / num_pos_pairs
    else:
        loss = torch.tensor(0.0, device=device)

    return loss


def _reduction(loss, reduction):
    """
    对损失进行降维操作。

    参数:
    - loss: torch.Tensor, 输入的损失值
    - reduction: str, 降维方式，可选 'none'、'mean' 或 'sum'

    返回:
    - torch.Tensor, 降维后的损失值
    """
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f"`reduction` = {reduction} 是无效的。请使用 'none'、'mean' 或 'sum'。")

def _rank_loss_deephit(pmf, y, rank_mat, sigma: float,
                       reduction: str = 'mean') :
    """
    Ranking loss from DeepHit.

    参数:
    - pmf: torch.Tensor, 概率质量函数矩阵，形状为 (batch_size, num_durations)
    - y: torch.Tensor, 事件时间的指示矩阵，形状为 (batch_size, num_durations)
    - rank_mat: torch.Tensor, 排名矩阵，形状为 (batch_size, batch_size)
    - sigma: float, DeepHit 中的超参数
    - reduction: str, 降维方式，可选 'none'、'mean' 或 'sum'

    返回:
    - torch.Tensor, 计算得到的排名损失
    """
    r = _diff_cdf_at_time_i(pmf, y)
    loss = rank_mat * torch.exp(-r / sigma)
    loss = loss.mean(1, keepdim=True)
    return _reduction(loss, reduction)

def _diff_cdf_at_time_i(pmf, y) :
    """
    计算 CDF 的差异矩阵 R_ij = F_i(T_i) - F_j(T_i)。

    参数:
    - pmf: torch.Tensor, 概率质量函数矩阵，形状为 (batch_size, num_durations)
    - y: torch.Tensor, 事件时间的指示矩阵，形状为 (batch_size, num_durations)

    返回:
    - torch.Tensor, CDF 差异矩阵，形状为 (batch_size, batch_size)
    """
    n = pmf.shape[0]
    ones = torch.ones((n, 1), device=pmf.device)
    r = pmf.cumsum(1).matmul(y.transpose(0, 1))
    diag_r = r.diag().view(1, -1)
    r = ones.matmul(diag_r) - r
    return r.transpose(0, 1)

def rank_loss_deephit_single(phi, idx_durations, events, rank_mat,
                             sigma: float, reduction: str = 'mean'):
    """
    Rank loss proposed by DeepHit authors for a single risk.

    参数:
    - phi: torch.Tensor, 模型输出，形状为 (batch_size, num_durations)
    - idx_durations: torch.Tensor, 事件时间索引，形状为 (batch_size,)
    - events: torch.Tensor, 事件指示器，形状为 (batch_size,)
    - rank_mat: torch.Tensor, 排名矩阵，形状为 (batch_size, batch_size)
    - sigma: float, DeepHit 中的超参数
    - reduction: str, 降维方式，可选 'none'、'mean' 或 'sum'

    返回:
    - torch.Tensor, 计算得到的排名损失
    """
    idx_durations = idx_durations.view(-1, 1)
    neural_net_output = pad_col(phi, val=1, where='end')  # 增加一个时间段，用于计算 KL 散度
    S = torch.sum(neural_net_output, dim=1, keepdim=True)
    probabilities = neural_net_output / S
    pmf = probabilities[:, :-1]  # 去掉最后一个时间段，因为它是截尾时间段
    # pmf = torch.softmax(phi, dim=1)  # 使用 softmax 计算概率质量函数
    # print('pmf', pmf)
    # print('idx_durations', idx_durations)
    y = torch.zeros_like(pmf).scatter(1, idx_durations, 1.0)  # 创建 one-hot 矩阵
    rank_loss = _rank_loss_deephit(pmf, y, rank_mat, sigma, reduction)
    return rank_loss