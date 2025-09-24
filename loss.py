import torch
import torch.nn.functional as F
from surv_tool import pad_col

# def deephit_loss(neural_net_output, Y_batch, D_batch):
#     batch_size, num_time_intervals = neural_net_output.shape
#     preds = neural_net_output 
#     probabilities = torch.softmax(pad_col(preds), 1)
#     Y_batch = Y_batch.long()
#     loss = torch.tensor(0.0, dtype=torch.float, device=neural_net_output.device)
#     for i in range(batch_size):
#         y_i = Y_batch[i]
#         d_i = D_batch[i]
#         if d_i == 1:
#             loss -= torch.log(probabilities[i, y_i] + 1e-10)
#         else:
#             sum_survival = 1 - torch.sum(probabilities[i,0:y_i+1])
#             loss -= torch.log(sum_survival + 1e-10)
#     loss = (loss / batch_size) 
#     return loss

# def kl_divergence(alpha, num_classes, device=None):
#     if not device:
#         device =device()
#     ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
#     sum_alpha = torch.sum(alpha)
#     first_term = (
#         torch.lgamma(sum_alpha)
#         - torch.lgamma(alpha).sum()
#         + torch.lgamma(ones).sum()
#         - torch.lgamma(ones.sum())
#     )
#     second_term = (
#         (alpha - ones)
#         .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
#         .sum()
#     )
#     kl = first_term + second_term
#     return kl

def deephit_loss_uncertainty(neural_net_output, Y_batch, D_batch, e_sum, pro,epochs):
    batch_size, num_time_intervals = neural_net_output.shape
    neural_net_output = pad_col(neural_net_output, val=1, where='end')
    S = torch.sum(neural_net_output, dim=1, keepdim=True)
    probabilities = neural_net_output / S
    Y_batch = Y_batch.long()
    loss = torch.tensor(0.0, dtype=torch.float, device=neural_net_output.device)
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epochs / 10, dtype=torch.float32),
    )
    for i in range(batch_size):
        y_i = Y_batch[i]
        d_i = D_batch[i]
        if d_i == 1:
            loss -= torch.log(probabilities[i, y_i] + 1e-10)
            pro = pro.detach()
            p_gt = 1 - pro[i, 0:y_i].sum()
            kl_div = annealing_coef*(e_sum[i]*(1-p_gt)).squeeze()
            loss += 1.0 * kl_div
        else:
            pro = pro.detach()
            p_gt = 1 - pro[i, 0:y_i+1].sum()
            kl_div = annealing_coef*(e_sum[i]*(1-p_gt)).squeeze()
            loss +=  1.0 * kl_div
            sum_survival = 1 - torch.sum(probabilities[i,0:y_i+1])
            loss -= torch.log(sum_survival + 1e-10)
    loss = (loss / batch_size) 
    return loss

import torch
import torch.nn.functional as F

def contrastive_loss(site_logits, site, target, D_batch, uncertainty,
                                 temperature=0.1, sigma=3):
    device = site_logits.device
    batch_size = site_logits.size(0)
    features = F.normalize(site_logits.mean(dim=1), p=2, dim=1)
    sim_matrix = torch.matmul(features, features.T) / temperature
    time_diff = torch.abs(target.view(-1, 1) - target.view(1, -1))
    tolerance = 5
    same_time = time_diff < tolerance
    diff_site = site.view(-1, 1) != site.view(1, -1)
    both_event = (D_batch.view(-1, 1) == 1) & (D_batch.view(1, -1) == 1)
    pos_mask = diff_site & same_time & both_event
    both_event_mask = (D_batch.view(-1, 1) == 1) & (D_batch.view(1, -1) == 1)
    case1_mask = both_event_mask & (time_diff >= tolerance)
    case2a_mask = (D_batch.view(-1, 1) == 1) & (D_batch.view(1, -1) == 0) & (target.view(1, -1) > target.view(-1, 1))
    case2b_mask = (D_batch.view(-1, 1) == 0) & (D_batch.view(1, -1) == 1) & (target.view(-1, 1) > target.view(1, -1))
    case2_mask = case2a_mask | case2b_mask
    neg_mask = (~pos_mask) & (case1_mask | case2_mask)
    #neg_weights = 1 - torch.exp(-time_diff / 2 * sigma**2)
    neg_weights = torch.exp(-time_diff / 2 * sigma**2)
    pos_weights = torch.zeros_like(sim_matrix, device=device)
    pos_weights[pos_mask] = torch.exp(-uncertainty.view(1, -1).expand(batch_size, batch_size)[pos_mask])
    pos_weights = pos_weights / pos_weights.sum()
    exp_sim = torch.exp(sim_matrix)
    loss_total = 0.0
    num_pos_pairs = 0
    for i in range(batch_size):
        pos_indices = torch.where(pos_mask[i])[0]
        if len(pos_indices) > 0:
            for j in pos_indices:
                numerator = exp_sim[i, j] * pos_weights[i, j]
                denominator = numerator + (exp_sim[i, :] * neg_mask[i].float() * neg_weights[i, :]).sum()
                loss = -torch.log(numerator / (denominator + 1e-10))
                loss_total += loss
                num_pos_pairs += 1
    if num_pos_pairs > 0:
        loss = loss_total / num_pos_pairs
    else:
        loss = torch.tensor(0.0, device=device)
    return loss
