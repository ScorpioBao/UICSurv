import torch
from torch.utils.data import DataLoader
from dataset_without_dwi import data, custom_collate_fn
from uicsurv.models import UICSURV_U_Site_UNETR
import numpy as np
from surv_tool import evaluate_model, compute_brier_score
from loss import deephit_loss_uncertainty
import random
import logging
import os
import pandas as pd


site="beizhong"
mod="w"

import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler(f'./logs/training_log_{mod}.txt',mode=mod), 
        logging.StreamHandler() 
    ]
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to: {seed}")



def train_model(model, train_loader, train_wdwi_loader, optimizer, loss_fn, device, epoch_index, num_epochs, uncertainty_dict,fold_num):
    model.train()
    running_loss = 0.0
    for i, (t2, dwi, tab, target_c, target, d, id, site) in enumerate(train_loader):
        tab = tab.to(device).float().unsqueeze(1)
        t2 = t2.to(device).float()
        dwi = dwi.to(device).float()
        target_c = target_c.to(device).float()
        Y_batch = target.to(device).type(torch.int64)
        D_batch = d.to(device).float()
        site = site.to(device).type(torch.int64)
        if epoch_index == 0:
            uncertainty = torch.ones(Y_batch.shape[0], device=device)
        else:
            uncertainty = torch.tensor([uncertainty_dict[sample_id.item()] for sample_id in id], device=device)
            
        log_hazards_batch, site_logits, evidence_sum, probability, total_constrastive_loss = model([t2, dwi, tab],site_labels=site,target=target,D_batch=D_batch,uncertainty=uncertainty)
        log_hazards_batch = log_hazards_batch + 1.0


        alpha_sum = log_hazards_batch.sum(dim=1)
        uncertainty = log_hazards_batch.shape[1] / alpha_sum
        for idx, sample_id in enumerate(id):
            uncertainty_dict[sample_id.item()] = uncertainty[idx].item()

        if epoch_index > 0:
            # weights = torch.tensor([uncertainty_dict[sample_id.item()] for sample_id in id], device=device)
            # c_loss_weight = min(20, epoch_index * 200 / num_epochs)
            c_loss_weight = min(1.0, epoch_index  / 10)
            # c_loss = contrastive_loss(site_logits, site, Y_batch, D_batch, weights) * c_loss_weight
            c_loss = total_constrastive_loss * c_loss_weight
        else:
            c_loss = 0

        loss_batch = loss_fn(log_hazards_batch, Y_batch, D_batch, evidence_sum, probability, epochs=epoch_index) + 1 * c_loss
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        running_loss += loss_batch.item()
        logging.info(f"Fold {fold_num} - Epoch {epoch_index + 1} - Batch {i + 1}/{len(train_loader)} - Loss: {loss_batch.item():.4f}")

    for i, (t2, dwi, tab, target_c, target, d, id, site) in enumerate(train_wdwi_loader):
        tab = tab.to(device).float().unsqueeze(1)
        t2 = t2.to(device).float()
        target_c = target_c.to(device).float()
        Y_batch = target.to(device).type(torch.int64)
        D_batch = d.to(device).float()
        site = site.to(device).type(torch.int64)
        dwi=None
        log_hazards_batch, evidence_sum, probability = model([t2, dwi, tab])
        log_hazards_batch = log_hazards_batch + 1.0

        loss_batch = loss_fn(log_hazards_batch, Y_batch, D_batch, evidence_sum, probability, epochs=epoch_index)
        optimizer.zero_grad() 
        loss_batch.backward()
        optimizer.step()

        running_loss += loss_batch.item()
        logging.info(f"Epoch {epoch_index + 1} - Batch {i + 1}/{len(train_wdwi_loader)} - Loss: {loss_batch.item():.4f}")

    epoch_loss = running_loss / (len(train_loader) + len(train_wdwi_loader))
    return epoch_loss


def main():
    set_random_seed(13)
    spatial_axes = [1, 1, 1]
    channels = [32, 32, 13]
    table_dir = f'./data/cv_folds1_{site}'
    mri_dirs = {

    }


    num_epochs = 50
    learning_rate = 5e-4
    patience = 15
    decay_gamma = 0.99
    scheduler_patience = 5
    scheduler_factor = 0.5
    batch_size = 32

    fold_results = {
        'train_c_indices': [],
        'val_c_indices': [],
        'test_c_indices': [],
        'train_bs': [],
        'val_bs': [],
        'test_bs': []
    }

    for fold_num in range(1, 6):
        set_random_seed(13)
        logging.info(f"\n ###############Starting fold {fold_num}##################")
        train_data, val_data, test_data, train_wdwi, val_wdwi, test_wdwi, time_grid_train_np = data(table_dir, mri_dirs, fold_num=fold_num)
        time_points = time_grid_train_np.shape[0]
        train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True,collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,collate_fn=custom_collate_fn)
        train_wdwi_loader = DataLoader(train_wdwi, batch_size=batch_size, shuffle=True,collate_fn=custom_collate_fn)
        val_wdwi_loader = DataLoader(val_wdwi, batch_size=batch_size, shuffle=False,collate_fn=custom_collate_fn)
        test_wdwi_loader = DataLoader(test_wdwi, batch_size=batch_size, shuffle=False,collate_fn=custom_collate_fn)

        model = UICSURV_U_Site_UNETR(
            n_modalities=3,
            channel_dims=channels,
            num_spatial_axes=spatial_axes,
            out_dims=time_points,
            l_d=128,
            l_c=128,
            fourier_encode_data=True,
            depth=2, 
            x_heads=8,
            l_heads=8,
            # attn_dropout=0.1,
            # ff_dropout=0.1,
            self_per_cross_attn=1,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=scheduler_patience, factor=scheduler_factor, verbose=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma)
        best_C_index = 0
        epochs_without_improvement = 0
        uncertainty_dict = {}

        for epoch_index in range(num_epochs):
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Epoch {epoch_index + 1} - Current Learning Rate: {current_lr:.6f}")
            epoch_loss = train_model(model, train_loader, train_wdwi_loader, optimizer, deephit_loss_uncertainty, device, epoch_index, num_epochs, uncertainty_dict,fold_num)
            logging.info(f"Epoch {epoch_index + 1} - Average Loss: {epoch_loss:.4f}")

            train_c_index, train_surv, all_train_labels, all_train_labels_c, all_train_D = evaluate_model(model, train_loader, train_wdwi_loader, device, batch_size, time_grid_train_np)
            val_c_index, val_surv, all_val_labels, all_val_labels_c, all_val_D = evaluate_model(model, val_loader, val_wdwi_loader, device, batch_size, time_grid_train_np)
            test_c_index, test_surv, all_test_labels, all_test_labels_c, all_test_D = evaluate_model(model, test_loader, test_wdwi_loader, device, batch_size, time_grid_train_np)
            train_surv_pd = pd.DataFrame(train_surv,index=time_grid_train_np)
            val_surv_pd = pd.DataFrame(val_surv,index=time_grid_train_np)
            test_surv_pd = pd.DataFrame(test_surv,index=time_grid_train_np)
            
            bs_train = compute_brier_score(train_surv_pd, all_train_labels_c, all_train_D,all_train_labels_c, all_train_D)
            bs_val = compute_brier_score(val_surv_pd, all_val_labels_c, all_val_D, all_train_labels_c, all_train_D)
            bs_test = compute_brier_score(test_surv_pd, all_test_labels_c, all_test_D, all_train_labels_c, all_train_D)
            
            logging.info(f"Train C-index: {train_c_index:.4f}, Train Brier Score: {bs_train:.4f}")
            logging.info(f"Validation C-index: {val_c_index:.4f}, Validation Brier Score: {bs_val:.4f}")
            logging.info(f"Test C-index: {test_c_index:.4f}, Test Brier Score: {bs_test:.4f}")
            
    
            if  val_c_index > best_C_index:
                best_C_index = val_c_index
                best_val_c_index = val_c_index
                best_train_c_index = train_c_index
                best_test_c_index = test_c_index
                best_train_bs = bs_train
                best_val_bs = bs_val
                best_test_bs = bs_test
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f'./checkpoints/fold_{fold_num}_best_model_{mod}_{site}.pth')
                logging.info(f"*********************************************************New best model saved with C-index: {best_C_index:.4f}")
                # logging.info(f"*********************************************************Best Test-C-index: {best_test_c_index:.4f}")
            else:
                epochs_without_improvement += 1

            logging.info(f"Epochs without improvement: {epochs_without_improvement}/{patience}")
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping at epoch {epoch_index + 1}")
                logging.info(f"*********************************************************Best C-index: {best_C_index:.4f}")
                break

            # scheduler.step(val_c_index)
            scheduler.step()

        fold_results['train_c_indices'].append(best_train_c_index)
        fold_results['val_c_indices'].append(best_val_c_index)
        fold_results['test_c_indices'].append(best_test_c_index)
        fold_results['train_bs'].append(best_train_bs)
        fold_results['val_bs'].append(best_val_bs)
        fold_results['test_bs'].append(best_test_bs)

  
    train_mean = np.mean(fold_results['train_c_indices'])
    train_std = np.std(fold_results['train_c_indices'])
    val_mean = np.mean(fold_results['val_c_indices'])
    val_std = np.std(fold_results['val_c_indices'])
    test_mean = np.mean(fold_results['test_c_indices'])
    test_std = np.std(fold_results['test_c_indices'])
    train_bs_mean = np.mean(fold_results['train_bs'])
    train_bs_std = np.std(fold_results['train_bs'])
    val_bs_mean = np.mean(fold_results['val_bs'])
    val_bs_std = np.std(fold_results['val_bs'])
    test_bs_mean = np.mean(fold_results['test_bs'])
    test_bs_std = np.std(fold_results['test_bs'])
    logging.info(f"Train C-index: {fold_results}, Train Brier Score: {train_bs_mean:.4f}")
    logging.info(f"Train C-index: Mean = {train_mean:.4f}, Std = {train_std:.4f}, Brier Score: Mean = {train_bs_mean:.4f}, Std = {train_bs_std:.4f}")
    logging.info(f"Validation C-index: Mean = {val_mean:.4f}, Std = {val_std:.4f}, Validation Brier Score: Mean = {val_bs_mean:.4f}, Std = {val_bs_std:.4f}")
    logging.info(f"Test C-index: Mean = {test_mean:.4f}, Std = {test_std:.4f}, Test Brier Score: Mean = {test_bs_mean:.4f}, Std = {test_bs_std:.4f}")

if __name__ == "__main__":
    main()