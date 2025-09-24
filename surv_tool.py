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
    if num_time_steps == 0:
        mask = (D_train == 1)
        label_transform = LabTransDiscreteTime(np.unique(Y_train[mask]))
    else:
        label_transform = LabTransDiscreteTime(num_time_steps, scheme='quantiles')

    Y_train_discrete_np, D_train_discrete_np = label_transform.fit_transform(Y_train, D_train)
    if Y_val is not None and D_val is not None:
        Y_val_discrete_np, D_val_discrete_np = label_transform.transform(Y_val, D_val)

    time_grid_train_np = label_transform.cuts

    output_num_time_steps = len(time_grid_train_np)
    print(f'Number of time steps to be used with DeepHit: {output_num_time_steps}')
    print('Time grid:', time_grid_train_np)

    time_labels = []
    for i in range(len(time_grid_train_np) - 1):
        if np.isinf(time_grid_train_np[i]):
            time_labels.append(f"(-inf - {time_grid_train_np[i+1]}]")
        else:
            time_labels.append(f"({time_grid_train_np[i]} - {time_grid_train_np[i+1]}]")
    time_labels.append(f"({time_grid_train_np[-1]} - inf)")

    train_event_counts = np.zeros(len(time_labels))
    train_censored_counts = np.zeros(len(time_labels))
    for time_step, event in zip(Y_train_discrete_np, D_train_discrete_np):
        if event == 1:
            train_event_counts[time_step] += 1
        else:
            train_censored_counts[time_step] += 1

    return Y_train_discrete_np, D_train_discrete_np, Y_val_discrete_np, D_val_discrete_np, label_transform, time_grid_train_np

def pad_col(input, val=0, where='end'):
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

def calculate_c_index(Y,D,surv,duration_index):
    if duration_index is None:
        duration_index = np.arange(surv.shape[0])
    surv = pd.DataFrame(surv, index = duration_index)
    eval_pycox = EvalSurv(surv, Y, D)
    c_index = eval_pycox.concordance_td('antolini')
    return c_index

def predict_survival_probability_uncertainty(model, input_data, batch_size=256, duration_index=None, sub=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

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
                batch_dwi = None   
            batch_preds_evidence,_,_ = model([batch_t2,batch_dwi,batch_tb])
            batch_preds_alpha = batch_preds_evidence + 1.0
            batch_preds_alpha = pad_col(batch_preds_alpha)
            batch_S = torch.sum(batch_preds_alpha, dim=1, keepdim=True)
            batch_preds = batch_preds_alpha / batch_S
            batch_preds = batch_preds[:, :-1]
            
            all_preds.append(batch_preds.cpu().numpy())

    pmf = np.vstack(all_preds)
    surv_probs = 1 - np.cumsum(pmf, axis=1)
    surv_probs = surv_probs.T

    if duration_index is None:
        duration_index = np.arange(surv_probs.shape[0])

    surv_df = pd.DataFrame(surv_probs, index=duration_index)

    if sub > 1:
        interp_times = np.linspace(duration_index[0], duration_index[-1], 
                                   num=len(duration_index) + (len(duration_index) - 1) * sub)
        num_samples = surv_df.shape[1]
        interp_survival_probs = np.zeros((len(interp_times), num_samples))
        for i in range(num_samples):
            survival_prob_sample = surv_df.iloc[:, i].values
            interp_func = interp1d(duration_index, survival_prob_sample, kind='linear', fill_value="extrapolate")
            interp_survival_probs[:, i] = interp_func(interp_times)
        surv_df = pd.DataFrame(interp_survival_probs, index=interp_times)

    return surv_df

def compute_brier_score(surv_df,Y_np,D_np,Y_train,D_train):
    time_grid_train_interp_np = surv_df.index.to_numpy()
    surv_np = surv_df.to_numpy().T
    eval = SurvivalEvaluator(surv_np, time_grid_train_interp_np,
                            Y_np, D_np,
                            Y_train, D_train) 
    IBS = eval.integrated_brier_score()
    brier_scores = IBS
    return brier_scores.mean()

def evaluate_model(model, loader_with_dwi, loader_without_dwi, device, batch_size, time_grid_train_np):
    model.to(device)
    model.eval()
    with torch.no_grad():
        all_surv, all_labels, all_D = [], [], []
        all_labels_c = []
        for i, (t2, dwi, tab, target_c, target, d, id, site) in enumerate(loader_with_dwi):
            tab = tab.to(device).float()
            tab = tab.unsqueeze(1)
            t2 = t2.to(device).float()
            dwi = dwi.to(device).float()
            target = target.to(device)
            target_c = target_c.to(device)
            surv = predict_survival_probability_uncertainty(model, [t2,dwi,tab], batch_size=batch_size, duration_index=time_grid_train_np, sub=0)
            all_surv.append(surv)
            all_labels.append(target.cpu().numpy())
            all_labels_c.append(target_c.cpu().numpy())
            all_D.append(d.cpu().numpy())
        for i, (t2, dwi, tab, target_c, target, d, id, site) in enumerate(loader_without_dwi):
            tab = tab.to(device).float()
            tab = tab.unsqueeze(1)
            target = target.to(device)
            target_c = target_c.to(device)
            t2 = t2.to(device).float()
            dwi = None
            surv = predict_survival_probability_uncertainty(model, [t2,dwi,tab], batch_size=batch_size, duration_index=time_grid_train_np, sub=0)
            all_surv.append(surv)
            all_labels.append(target.cpu().numpy())
            all_labels_c.append(target_c.cpu().numpy())
            all_D.append(d.cpu().numpy())
        all_surv = np.concatenate(all_surv, axis=1)
        all_labels = np.concatenate(all_labels)
        all_labels_c = np.concatenate(all_labels_c)
        all_D = np.concatenate(all_D)
        c_index = calculate_c_index(all_labels, all_D, surv=all_surv,duration_index=time_grid_train_np)
    return c_index, all_surv, all_labels, all_labels_c, all_D

def evaluate_model_wo_missing(model, loader_with_dwi, device, batch_size, time_grid_train_np):
    model.to(device)
    model.eval()
    with torch.no_grad():
        all_surv, all_labels, all_D = [], [], []
        all_labels_c = []
        for i, (t2, dwi, tab, target_c, target, d, id, site) in enumerate(loader_with_dwi):
            tab = tab.to(device).float()
            tab = tab.unsqueeze(1)
            t2 = t2.to(device).float()
            dwi = dwi.to(device).float()
            target = target.to(device)
            target_c = target_c.to(device)
            surv = predict_survival_probability_uncertainty(model, [t2,dwi,tab], batch_size=batch_size, duration_index=time_grid_train_np, sub=0)
            all_surv.append(surv)
            all_labels.append(target.cpu().numpy())
            all_labels_c.append(target_c.cpu().numpy())
            all_D.append(d.cpu().numpy())
        all_surv = np.concatenate(all_surv, axis=1)
        all_labels = np.concatenate(all_labels)
        all_labels_c = np.concatenate(all_labels_c)
        all_D = np.concatenate(all_D)
        c_index = calculate_c_index(all_labels, all_D, surv=all_surv,duration_index=time_grid_train_np)
    return c_index, all_surv, all_labels, all_labels_c, all_D
