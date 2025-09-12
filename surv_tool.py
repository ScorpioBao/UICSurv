import torch
import pandas as pd
import numpy as np
from typing import Tuple, Union
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from scipy.interpolate import interp1d
from pycox.evaluation import EvalSurv
from volumentations import *
from SurvivalEVAL.Evaluator import SurvivalEvaluator


def time_interview(num_time_steps, Y_train, D_train, Y_val=None, D_val=1):
    # 使用时间网格进行区间划分
    if num_time_steps == 0:
        mask = (D_train == 1)  # 布尔掩码，选择那些训练数据中发生了事件的病人
        label_transform = LabTransDiscreteTime(np.unique(Y_train[mask]))
    else:
        label_transform = LabTransDiscreteTime(num_time_steps, scheme='quantiles') # quantiles,equidistant

    # 使用标签变换
    Y_train_discrete_np, D_train_discrete_np = label_transform.fit_transform(Y_train, D_train)
    if Y_val is not None and D_val is not None:
        Y_val_discrete_np, D_val_discrete_np = label_transform.transform(Y_val, D_val)

    # 获取时间网格
    time_grid_train_np = label_transform.cuts

    # 打印时间步长信息
    output_num_time_steps = len(time_grid_train_np)
    print(f'Number of time steps to be used with DeepHit: {output_num_time_steps}')
    print('Time grid:', time_grid_train_np)

    # 基于时间网格生成时间区间标签
    time_labels = []
    for i in range(len(time_grid_train_np) - 1):
        if np.isinf(time_grid_train_np[i]):
            time_labels.append(f"(-inf - {time_grid_train_np[i+1]}]")  # 处理第一个时间段
        else:
            time_labels.append(f"({time_grid_train_np[i]} - {time_grid_train_np[i+1]}]")
    #time_labels.append(f"({time_grid_train_np[-2]} - {time_grid_train_np[-1]}]")  # 最后一个区间标记为最后两个点的区间
    time_labels.append(f"({time_grid_train_np[-1]} - inf)")  # 最后一个区间标记为最后两个点的区间
    


    # 统计训练数据每个时间区间的事件和截尾数
    train_event_counts = np.zeros(len(time_labels))
    train_censored_counts = np.zeros(len(time_labels))
    for time_step, event in zip(Y_train_discrete_np, D_train_discrete_np):
        if event == 1:  # 如果发生了事件
            train_event_counts[time_step] += 1
        else:  # 如果是截尾
            train_censored_counts[time_step] += 1

    return Y_train_discrete_np, D_train_discrete_np, Y_val_discrete_np, D_val_discrete_np, label_transform, time_grid_train_np


def pad_col(input, val=0, where='end'):
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

# def predict_survival_probability(model, input_data, batch_size=256, duration_index=None, sub=0):
#     """
#     计算生存概率并返回 Pandas DataFrame 格式
    
#     参数:
#     - model: DeepHit 模型，用于预测
#     - input_data: ndarray, 输入数据，形状为 (num_samples, num_features)
#     - batch_size: int, 批处理大小，默认 256
#     - duration_index: ndarray, 用于 DataFrame 的时间索引，默认 None
#     - interpolation: bool, 是否进行插值，默认 False
#     - sub: int, 插值的子单位数量，用于插值，默认 10
    
#     返回:
#     - surv_df: pd.DataFrame, 每行对应一个时间点的生存概率，每列对应一个样本。形状为 (num_time_points, num_samples)
#     """
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
#     model.to(device)
#     model.eval()

#     # Step 1: 获取 PMF
#     with torch.no_grad():
#         all_preds = []
#         num_samples = input_data[0].shape[0]
        
#         for i in range(0, num_samples, batch_size):
#             if input_data[0] is not None:
#                 batch_tb = input_data[0][i:i + batch_size]
#             else:
#                 batch_tb = None
#             if input_data[1] is not None:
#                 batch_t2 = input_data[1][i:i + batch_size]
#             else:
#                 batch_t2 = None
#             if input_data[2] is not None:
#                 batch_dwi = input_data[2][i:i + batch_size]      
#             else:
#                 batch_dwi = None   
            
#             # print(batch_tb.shape)
#             # print(batch_img.shape)
#             batch_preds,_,_ = model([None,batch_t2,None])  # 模型的输出，形状为 (batch_size, num_time_steps)
#             preds = batch_preds 
#             pmf = torch.softmax(pad_col(preds), 1)[:, :-1]  # PMF
#             # batch_preds_alpha = batch_preds_evidence + 1.0
#             # batch_S = torch.sum(batch_preds_alpha, dim=1, keepdim=True)
#             # batch_preds = batch_preds_alpha / batch_S
#             batch_preds = pmf  # 使用 softmax 转换为每个时间段的概率
            
#             all_preds.append(batch_preds.cpu().numpy())


#     pmf = np.vstack(all_preds)  # 形状为 (num_samples, num_time_steps)

#     # Step 2: 根据 PMF 计算生存概率
#     surv_probs = 1 - np.cumsum(pmf, axis=1)  # 累积和方式计算生存概率，形状为 (num_samples, num_time_steps)

#     # Step 3: 转置矩阵，使得每行对应一个时间点，每列对应一个样本
#     surv_probs = surv_probs.T  # 形状为 (num_time_steps, num_samples)

#     # 如果没有提供 duration_index，则使用默认索引
#     if duration_index is None:
#         duration_index = np.arange(surv_probs.shape[0])

#     # Step 4: 将生存概率转换为 DataFrame 格式
#     surv_df = pd.DataFrame(surv_probs, index=duration_index)

#     # Step 5: 是否进行插值
#     if sub > 1:
#         # 新的插值时间点
#         interp_times = np.linspace(duration_index[0], duration_index[-1], 
#                                    num=len(duration_index) + (len(duration_index) - 1) * sub)

#         # 初始化用于存储插值后的生存概率矩阵
#         num_samples = surv_df.shape[1]
#         interp_survival_probs = np.zeros((len(interp_times), num_samples))

#         # 对每个样本的生存概率进行插值
#         for i in range(num_samples):
#             survival_prob_sample = surv_df.iloc[:, i].values
#             interp_func = interp1d(duration_index, survival_prob_sample, kind='linear', fill_value="extrapolate")
#             interp_survival_probs[:, i] = interp_func(interp_times)

#         # 更新 DataFrame
#         surv_df = pd.DataFrame(interp_survival_probs, index=interp_times)

#     return surv_df

def calculate_c_index(Y,D,surv,duration_index):
    if duration_index is None:
        duration_index = np.arange(surv.shape[0])
    surv = pd.DataFrame(surv, index = duration_index)
    
    eval_pycox = EvalSurv(surv, Y, D)
    c_index = eval_pycox.concordance_td('antolini')
    return c_index



def predict_survival_probability_uncertainty(model, input_data, batch_size=256, duration_index=None, sub=0):
    """
    计算生存概率并返回 Pandas DataFrame 格式
    
    参数:
    - model: DeepHit 模型，用于预测
    - input_data: ndarray, 输入数据，形状为 (num_samples, num_features)
    - batch_size: int, 批处理大小，默认 256
    - duration_index: ndarray, 用于 DataFrame 的时间索引，默认 None
    - interpolation: bool, 是否进行插值，默认 False
    - sub: int, 插值的子单位数量，用于插值，默认 10
    
    返回:
    - surv_df: pd.DataFrame, 每行对应一个时间点的生存概率，每列对应一个样本。形状为 (num_time_points, num_samples)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    model.to(device)
    model.eval()

    # Step 1: 获取 PMF
    with torch.no_grad():
        all_preds = []
        num_samples = batch_size
        
        for i in range(0, num_samples, batch_size):
            if input_data[2] is not None:
                batch_tb = input_data[2][i:i + batch_size].to(device).float()
            else:
                batch_tb = None
            if input_data[0] is not None:
                batch_t2 = input_data[0][i:i + batch_size].to(device).float()
            else:
                batch_t2 = None
            if input_data[1] is not None:
                batch_dwi = input_data[1][i:i + batch_size].to(device).float()      
            else:
                # print("没有DWI数据")
                batch_dwi = None   
            batch_preds_evidence,_,_ = model([batch_t2,batch_dwi,batch_tb])  # 模型的输出，形状为 (batch_size, num_time_steps)
            batch_preds_alpha = batch_preds_evidence + 1.0
            batch_preds_alpha = pad_col(batch_preds_alpha)
            batch_S = torch.sum(batch_preds_alpha, dim=1, keepdim=True)
            batch_preds = batch_preds_alpha / batch_S
            batch_preds = batch_preds[:, :-1]  # 去掉最后一个时间段，因为它是截尾时间段
            
            all_preds.append(batch_preds.cpu().numpy())


    pmf = np.vstack(all_preds)  # 形状为 (num_samples, num_time_steps)

    # Step 2: 根据 PMF 计算生存概率
    surv_probs = 1 - np.cumsum(pmf, axis=1)  # 累积和方式计算生存概率，形状为 (num_samples, num_time_steps)

    # Step 3: 转置矩阵，使得每行对应一个时间点，每列对应一个样本
    surv_probs = surv_probs.T  # 形状为 (num_time_steps, num_samples)

    # 如果没有提供 duration_index，则使用默认索引
    if duration_index is None:
        duration_index = np.arange(surv_probs.shape[0])

    # Step 4: 将生存概率转换为 DataFrame 格式
    surv_df = pd.DataFrame(surv_probs, index=duration_index)

    # Step 5: 是否进行插值
    if sub > 1:
        # 新的插值时间点
        interp_times = np.linspace(duration_index[0], duration_index[-1], 
                                   num=len(duration_index) + (len(duration_index) - 1) * sub)

        # 初始化用于存储插值后的生存概率矩阵
        num_samples = surv_df.shape[1]
        interp_survival_probs = np.zeros((len(interp_times), num_samples))

        # 对每个样本的生存概率进行插值
        for i in range(num_samples):
            survival_prob_sample = surv_df.iloc[:, i].values
            interp_func = interp1d(duration_index, survival_prob_sample, kind='linear', fill_value="extrapolate")
            interp_survival_probs[:, i] = interp_func(interp_times)

        # 更新 DataFrame
        surv_df = pd.DataFrame(interp_survival_probs, index=interp_times)

    return surv_df




def compute_brier_score(surv_df,Y_np,D_np,Y_train,D_train):
    
    
    #surv_df = predict_survival_probability(base_neural_net, X_train_np, batch_size=32, duration_index=time_grid_train_np, sub=0)
    time_grid_train_interp_np = surv_df.index.to_numpy()  # the interpolated time grid is an interpolated version of the *training*  time grid
    surv_np = surv_df.to_numpy().T
    eval = SurvivalEvaluator(surv_np, time_grid_train_interp_np,
                            Y_np, D_np,
                            Y_train, D_train) 
    IBS = eval.integrated_brier_score()
    brier_scores = IBS
    return brier_scores.mean()
    # print(f'IBS: {IBS.mean()}') # 平均Brier score
    
    
    
    
    
def evaluate_model(model, loader_with_dwi, loader_without_dwi, device, batch_size, time_grid_train_np):
    model.to(device)
    model.eval()
    with torch.no_grad():
        all_surv, all_labels, all_D = [], [], []
        all_labels_c = []
        # 处理有DWI的数据
        for i, (t2, dwi, tab, target_c, target, d, id, site) in enumerate(loader_with_dwi):
            tab = tab.to(device).float()
            tab = tab.unsqueeze(1)
            t2 = t2.to(device).float()
            dwi = dwi.to(device).float()
            target = target.to(device)
            target_c = target_c.to(device)
            #t2 = None  # 如果不使用T2，设为None
            #dwi = None  # 如果不使用DWI，设为None
            
            surv = predict_survival_probability_uncertainty(model, [t2,dwi,tab], batch_size=batch_size, duration_index=time_grid_train_np, sub=0)
            all_surv.append(surv)
            all_labels.append(target.cpu().numpy())
            all_labels_c.append(target_c.cpu().numpy())
            all_D.append(d.cpu().numpy())
        
        # 处理无DWI的数据
        for i, (t2, dwi, tab, target_c, target, d, id, site) in enumerate(loader_without_dwi):
            tab = tab.to(device).float()
            tab = tab.unsqueeze(1)
            target = target.to(device)
            target_c = target_c.to(device)
            t2 = t2.to(device).float()
            # t2 = None  # 如果不使用T2，设为None
            dwi = None  # 无DWI数据
            
            surv = predict_survival_probability_uncertainty(model, [t2,dwi,tab], batch_size=batch_size, duration_index=time_grid_train_np, sub=0)
            all_surv.append(surv)
            all_labels.append(target.cpu().numpy())
            all_labels_c.append(target_c.cpu().numpy())
            all_D.append(d.cpu().numpy())
        
        # 合并结果
        all_surv = np.concatenate(all_surv, axis=1)
        all_labels = np.concatenate(all_labels)
        all_labels_c = np.concatenate(all_labels_c)
        all_D = np.concatenate(all_D)
        
        # 计算C-index
        c_index = calculate_c_index(all_labels, all_D, surv=all_surv,duration_index=time_grid_train_np)
    
    return c_index, all_surv, all_labels, all_labels_c, all_D

   
def evaluate_model_wo_missing(model, loader_with_dwi, device, batch_size, time_grid_train_np):
    model.to(device)
    model.eval()
    with torch.no_grad():
        all_surv, all_labels, all_D = [], [], []
        all_labels_c = []
        # 处理有DWI的数据
        for i, (t2, dwi, tab, target_c, target, d, id, site) in enumerate(loader_with_dwi):
            tab = tab.to(device).float()
            tab = tab.unsqueeze(1)
            t2 = t2.to(device).float()
            dwi = dwi.to(device).float()
            target = target.to(device)
            target_c = target_c.to(device)
            #t2 = None  # 如果不使用T2，设为None
            #dwi = None  # 如果不使用DWI，设为None
            
            surv = predict_survival_probability_uncertainty(model, [t2,dwi,tab], batch_size=batch_size, duration_index=time_grid_train_np, sub=0)
            all_surv.append(surv)
            all_labels.append(target.cpu().numpy())
            all_labels_c.append(target_c.cpu().numpy())
            all_D.append(d.cpu().numpy())

        
        # 合并结果
        all_surv = np.concatenate(all_surv, axis=1)
        all_labels = np.concatenate(all_labels)
        all_labels_c = np.concatenate(all_labels_c)
        all_D = np.concatenate(all_D)
        
        # 计算C-index
        c_index = calculate_c_index(all_labels, all_D, surv=all_surv,duration_index=time_grid_train_np)
    
    return c_index, all_surv, all_labels, all_labels_c, all_D
