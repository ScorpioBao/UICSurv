import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
import random
from surv_tool import time_interview
import monai
from scipy.stats import zscore
from monai.transforms import (
    Compose,
    RandRotate,
    RandFlip,
    RandZoom,
    RandGaussianNoise,
    AdjustContrast,
    RandAdjustContrast,
    RandShiftIntensity,
    RandBiasField,
    ToTensor,
)
import matplotlib.pyplot as plt


device = torch.device('cpu')

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_random_seed(42)

time_point = 10

def zscore_standardize_tensor(tensor):
    if tensor is None:
        return None
    tensor_np = tensor.cpu().numpy()
    flattened_tensor = tensor_np.reshape(-1)
    if np.std(flattened_tensor) == 0:
        return tensor
    standardized_np = zscore(flattened_tensor)
    standardized_tensor = torch.from_numpy(standardized_np).reshape(tensor.shape).float()
    if tensor.is_cuda:
        standardized_tensor = standardized_tensor.cuda()
    return standardized_tensor

class MultiModalDataset(Dataset):
    def __init__(self, mri_dirs, table_csv, transform=None):
        self.mri_dirs = mri_dirs
        self.table_csv = table_csv
        self.transform = transform
        self.table_csv['PatientID'] = self.table_csv['PatientID'].astype(str).apply(lambda x: x.replace('.0', ''))
        self.patient_ids = self.table_csv['PatientID'].values
        self.censored = self.table_csv['DFS_Status'].values
        self.loaded_data_cache = {}
        self.t2_expected_shape = (256, 256, 20)
        self.dwi_expected_shape = (256, 256, 20)
        
    def __len__(self):
        return len(self.patient_ids)
    
    def load_mri_data(self, patient_id):
        if patient_id in self.loaded_data_cache:
            return self.loaded_data_cache[patient_id]
        t2 = None
        dwi = None
        for t2_dir_key in [key for key in self.mri_dirs.keys() if 't2' in key.lower()]:
            t2_dir = self.mri_dirs[t2_dir_key]
            t2_file = os.path.join(t2_dir, patient_id, 'T2_with_mask.npy')
            if os.path.exists(t2_file):
                t2 = np.load(t2_file)
                t2 = torch.tensor(t2, dtype=torch.float32)
                break
        if t2 is None:
            t2 = None
        for dwi_dir_key in [key for key in self.mri_dirs.keys() if 'dwi' in key.lower()]:
            dwi_dir = self.mri_dirs[dwi_dir_key]
            dwi_file = os.path.join(dwi_dir, patient_id, 'DWI_with_mask.npy')
            if os.path.exists(dwi_file):
                dwi = np.load(dwi_file)
                dwi = torch.tensor(dwi, dtype=torch.float32)
                break
        if dwi is None:
            dwi = None
        self.loaded_data_cache[patient_id] = (t2, dwi)
        return t2, dwi

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        t2, dwi = self.load_mri_data(patient_id)
        if self.transform:
            if (t2 is not None) & (dwi is not None):
                combined = torch.cat([t2.unsqueeze(0), dwi.unsqueeze(0)], dim=0)
                combined_augmented = self.transform(combined)
                t2,dwi = torch.split(combined_augmented, 1, dim=0)
                t2 = t2.squeeze(0)
                dwi = dwi.squeeze(0)
            if t2 is None:
                dwi = self.transform(dwi)
            if dwi is None:
                t2 = self.transform(t2)
        if t2 is not None:
            t2 = zscore_standardize_tensor(t2)
            t2 = t2.unsqueeze(-1)
            t2 = t2.permute(2, 0, 1, 3)
        if dwi is not None:
            dwi = zscore_standardize_tensor(dwi)
            dwi = dwi.unsqueeze(-1)
            dwi = dwi.permute(2, 0, 1, 3)
        table_row = self.table_csv[self.table_csv['PatientID'] == patient_id]
        table_features = table_row.drop(columns='PatientID').values.squeeze().astype('float32')
        table_features = torch.tensor(table_features, dtype=torch.float32)
        patient_id = int(patient_id)
        return {
            'T2': t2,
            'dwi': dwi,
            'table': table_features,
            'patient_id': patient_id,
        }

class Transform:
    def __init__(self, augment=True):
        self.augment = augment
        self.transform = Compose([
            RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
            RandFlip(spatial_axis=[1], prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.3, prob=0.5),
            RandGaussianNoise(prob=0.5),
            RandBiasField(prob=0.5),
            ToTensor(),
        ])
        self.transform2 = Compose([
            RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
            RandFlip(spatial_axis=[0], prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.3, prob=0.5),
            RandGaussianNoise(prob=0.5),
            RandBiasField(prob=0.5),
            ToTensor(),
        ])

    def __call__(self, image):
        if self.augment:
            if image.shape[0] == 2:
                image = self.transform(image)
            else:
                image = self.transform2(image)
        return image


def load_data_from_csv(csv_file):
    return pd.read_csv(csv_file)

def generate_dataset_from_csv(fold_dir, mri_dirs):
    train_csv_file = os.path.join(fold_dir, 'X_train_processed.csv')
    val_csv_file = os.path.join(fold_dir, 'X_val_processed.csv')
    test_csv_file = os.path.join(fold_dir, 'X_test_processed.csv')
    X_train_np = load_data_from_csv(train_csv_file)
    X_val_np = load_data_from_csv(val_csv_file)
    X_test_np = load_data_from_csv(test_csv_file)
    trian_transform = Transform(augment=False)
    val_transform = Transform(augment=False)
    train_data = MultiModalDataset(mri_dirs, X_train_np, transform=trian_transform)
    val_data = MultiModalDataset(mri_dirs, X_val_np, transform=val_transform)
    test_data = MultiModalDataset(mri_dirs, X_test_np, transform=val_transform)
    return train_data, val_data, test_data


def data(table_dir, mri_dirs, fold_num):
    fold_dirs = [os.path.join(table_dir, f'fold_{i+1}') for i in range(5)]
    fold_dir = fold_dirs[fold_num - 1]
    print(f"Processing fold {fold_num} in {fold_dir}", flush=True)
    train_data, val_data, test_data = generate_dataset_from_csv(fold_dir, mri_dirs)
    print(f"Total number of samples in train dataset: {len(train_data)}", flush=True)
    print(f"Total number of samples in validation dataset: {len(val_data)}", flush=True)
    print(f"Total number of samples in test dataset: {len(test_data)}", flush=True)

    def extract_data(dataset):
        X_T2 = []
        X_dwi = []
        X_table = []
        Y = []
        D = []
        id = []
        site = []
        for i in range(len(dataset)):
            item = dataset[i]
            X_T2.append(item['T2'])
            X_dwi.append(item['dwi'])
            X_table.append(item['table'][0:-3])
            Y.append(item['table'][-3])
            D.append(item['table'][-2])
            id.append(item['patient_id'])
            site.append(item['table'][-1])
        return np.stack(X_T2), X_dwi, np.stack(X_table), np.stack(Y), np.stack(D), np.stack(id), np.stack(site)

    def create_tensor_data(X_T2, X_dwi, X_table, Y, Y_d, D, id, site):
        X_T2 = torch.tensor(X_T2, dtype=torch.float32, device=device)
        X_table = torch.tensor(X_table, dtype=torch.float32, device=device)
        Y = torch.tensor(Y, dtype=torch.int64, device=device)
        Y_d = torch.tensor(Y_d, dtype=torch.int64, device = device)
        D = torch.tensor(D, dtype=torch.int32, device=device)
        id = torch.tensor(id, dtype=torch.int32, device=device)
        site = torch.tensor(site, dtype=torch.int32, device=device)
        X_dwi_tensor = []
        for dwi in X_dwi:
            if dwi is not None:
                X_dwi_tensor.append(dwi.clone().detach().to(dtype=torch.float32, device=device))
            else:
                X_dwi_tensor.append(None)
        return list(zip(X_T2, X_dwi_tensor, X_table, Y, Y_d, D, id, site))

    def split_data_by_dwi(X_T2, X_dwi, X_table, Y, Y_d, D, id, site):
        has_dwi_indices = [i for i, dwi in enumerate(X_dwi) if dwi is not None]
        missing_dwi_indices = [i for i, dwi in enumerate(X_dwi) if dwi is None]
        X_T2_has_dwi = X_T2[has_dwi_indices]
        X_dwi_has_dwi = [X_dwi[i] for i in has_dwi_indices]
        X_table_has_dwi = X_table[has_dwi_indices]
        Y_has_dwi = Y[has_dwi_indices]
        Y_d_has_dwi = Y_d[has_dwi_indices]
        D_has_dwi = D[has_dwi_indices]
        id_has_dwi = id[has_dwi_indices]
        site_has_dwi = site[has_dwi_indices]
        X_T2_missing = X_T2[missing_dwi_indices]
        X_dwi_missing = [None] * len(missing_dwi_indices)
        X_table_missing = X_table[missing_dwi_indices]
        Y_missing = Y[missing_dwi_indices]
        Y_d_missing = Y_d[missing_dwi_indices]
        D_missing = D[missing_dwi_indices]
        id_missing = id[missing_dwi_indices]
        site_missing = site[missing_dwi_indices]
        return (
            (X_T2_has_dwi, X_dwi_has_dwi, X_table_has_dwi, Y_has_dwi, Y_d_has_dwi, D_has_dwi, id_has_dwi, site_has_dwi),
            (X_T2_missing, X_dwi_missing, X_table_missing, Y_missing, Y_d_missing ,D_missing, id_missing, site_missing)
        )

    X_train_T2, X_train_dwi, X_train_table, Y_train_np, D_train_np, train_id, train_site = extract_data(train_data)
    X_val_T2, X_val_dwi, X_val_table, Y_val_np, D_val_np, val_id, val_site = extract_data(val_data)
    X_test_T2, X_test_dwi, X_test_table, Y_test_np, D_test_np, test_id, test_site = extract_data(test_data)

    Y_train_discrete_np, D_train_discrete_np, Y_val_discrete_np, D_val_discrete_np, label_transform, time_grid_train_np =\
        time_interview(time_point, Y_train_np, D_train_np, Y_val=Y_val_np, D_val=D_val_np)
    Y_test_discrete_np, D_test_discrete_np = label_transform.transform(Y_test_np, D_test_np)

    (X_train_T2_has_dwi, X_train_dwi_has_dwi, X_train_table_has_dwi, Y_train_has_dwi, Y_train_discrete_has_dwi, D_train_has_dwi, train_id_has_dwi, train_site_has_dwi),\
    (X_train_T2_missing, X_train_dwi_missing, X_train_table_missing, Y_train_missing, Y_train_discrete_missing, D_train_missing, train_id_missing, train_site_missing) =\
        split_data_by_dwi(X_train_T2, X_train_dwi, X_train_table, Y_train_np, Y_train_discrete_np, D_train_discrete_np, train_id, train_site)

    train_data = create_tensor_data(X_train_T2_has_dwi, X_train_dwi_has_dwi, X_train_table_has_dwi, Y_train_has_dwi, Y_train_discrete_has_dwi,D_train_has_dwi, train_id_has_dwi, train_site_has_dwi)

    X_train_wdwi = list(zip(
        torch.tensor(X_train_T2_missing, dtype=torch.float32, device=device),
        X_train_dwi_missing,
        torch.tensor(X_train_table_missing, dtype=torch.float32, device=device),
        torch.tensor(Y_train_missing, dtype=torch.int64, device=device),
        torch.tensor(Y_train_discrete_missing,dtype=torch.int64, device=device),
        torch.tensor(D_train_missing, dtype=torch.int32, device=device),
        torch.tensor(train_id_missing, dtype=torch.int32, device=device),
        torch.tensor(train_site_missing, dtype=torch.int32, device=device)
    ))

    (X_val_T2_has_dwi, X_val_dwi_has_dwi, X_val_table_has_dwi, Y_val_has_dwi, Y_val_discrete_has_dwi, D_val_has_dwi, val_id_has_dwi, val_site_has_dwi),\
    (X_val_T2_missing, X_val_dwi_missing, X_val_table_missing, Y_val_missing, Y_val_discrete_missing,D_val_missing, val_id_missing, val_site_missing) =\
        split_data_by_dwi(X_val_T2, X_val_dwi, X_val_table, Y_val_np ,Y_val_discrete_np, D_val_discrete_np, val_id, val_site)

    val_data = create_tensor_data(X_val_T2_has_dwi, X_val_dwi_has_dwi, X_val_table_has_dwi, Y_val_has_dwi, Y_val_discrete_has_dwi,D_val_has_dwi, val_id_has_dwi, val_site_has_dwi)

    X_val_wdwi = list(zip(
        torch.tensor(X_val_T2_missing, dtype=torch.float32, device=device),
        X_val_dwi_missing,
        torch.tensor(X_val_table_missing, dtype=torch.float32, device=device),
        torch.tensor(Y_val_missing, dtype=torch.int64, device=device),
        torch.tensor(Y_val_discrete_missing,dtype=torch.int64, device=device),
        torch.tensor(D_val_missing, dtype=torch.int32, device=device),
        torch.tensor(val_id_missing, dtype=torch.int32, device=device),
        torch.tensor(val_site_missing, dtype=torch.int32, device=device)
    ))

    (X_test_T2_has_dwi, X_test_dwi_has_dwi, X_test_table_has_dwi, Y_test_has_dwi, Y_test_discrete_has_dwi, D_test_has_dwi, test_id_has_dwi, test_site_has_dwi),\
    (X_test_T2_missing, X_test_dwi_missing, X_test_table_missing, Y_test_missing, Y_test_discrete_missing, D_test_missing, test_id_missing, test_site_missing) =\
        split_data_by_dwi(X_test_T2, X_test_dwi, X_test_table, Y_test_np, Y_test_discrete_np, D_test_discrete_np, test_id, test_site)

    test_data = create_tensor_data(X_test_T2_has_dwi, X_test_dwi_has_dwi, X_test_table_has_dwi, Y_test_has_dwi, Y_test_discrete_has_dwi, D_test_has_dwi, test_id_has_dwi, test_site_has_dwi)

    X_test_wdwi = list(zip(
        torch.tensor(X_test_T2_missing, dtype=torch.float32, device=device),
        X_test_dwi_missing,
        torch.tensor(X_test_table_missing, dtype=torch.float32, device=device),
        torch.tensor(Y_test_missing, dtype=torch.int64, device=device),
        torch.tensor(Y_test_discrete_missing, dtype=torch.int64, device=device),
        torch.tensor(D_test_missing, dtype=torch.int32, device=device),
        torch.tensor(test_id_missing, dtype=torch.int32, device=device),
        torch.tensor(test_site_missing, dtype=torch.int32, device=device)
    ))
    
    print("Training data with DWI:")
    X_train_dwi_has_dwi = torch.stack(X_train_dwi_has_dwi).numpy()
    print(X_train_T2_has_dwi.shape, X_train_dwi_has_dwi.shape, X_train_table_has_dwi.shape, Y_train_has_dwi.shape, D_train_has_dwi.shape, train_id_has_dwi.shape, train_site_has_dwi.shape)
    print("Training data without DWI:")
    print(X_train_T2_missing.shape, len(X_train_dwi_missing), X_train_table_missing.shape, Y_train_missing.shape, D_train_missing.shape, train_id_missing.shape, train_site_missing.shape)
    print("Validation data with DWI:")
    X_val_dwi_has_dwi = torch.stack(X_val_dwi_has_dwi).numpy()
    print(X_val_T2_has_dwi.shape, X_val_dwi_has_dwi.shape, X_val_table_has_dwi.shape, Y_val_has_dwi.shape, D_val_has_dwi.shape, val_id_has_dwi.shape, val_site_has_dwi.shape)
    print("Validation data without DWI:")
    print(X_val_T2_missing.shape, len(X_val_dwi_missing), X_val_table_missing.shape, Y_val_missing.shape, D_val_missing.shape, val_id_missing.shape, val_site_missing.shape)
    print("Test data with DWI:")
    X_test_dwi_has_dwi = torch.stack(X_test_dwi_has_dwi).numpy()
    print(X_test_T2_has_dwi.shape, X_test_dwi_has_dwi.shape, X_test_table_has_dwi.shape, Y_test_has_dwi.shape, D_test_has_dwi.shape, test_id_has_dwi.shape, test_site_has_dwi.shape)
    print("Test data without DWI:")
    print(X_test_T2_missing.shape, len(X_test_dwi_missing), X_test_table_missing.shape, Y_test_missing.shape, D_test_missing.shape, test_id_missing.shape, test_site_missing.shape)
  
    

    return train_data, val_data, test_data, X_train_wdwi, X_val_wdwi, X_test_wdwi ,time_grid_train_np



def custom_collate_fn(batch):

    batch = [item for item in batch if item is not None]
    

    if len(batch) == 0:
        return None
    
    T2 = [item[0] for item in batch]
    dwi_shape = torch.zeros(10, 126, 126, 1) 
    DWI = [item[1] if item[1] is not None else torch.zeros_like(dwi_shape) for item in batch]  
    table = [item[2] for item in batch]
    Y = [item[3] for item in batch]
    Y_d = [item[4] for item in batch]
    D = [item[5] for item in batch]
    id = [item[6] for item in batch]
    site = [item[7] for item in batch]


    T2 = torch.stack(T2)
    DWI = torch.stack(DWI)
    table = torch.stack(table)
    Y = torch.stack(Y)
    Y_d = torch.stack(Y_d)
    D = torch.stack(D)
    id = torch.stack(id)
    site = torch.stack(site)

    return T2, DWI, table, Y,Y_d, D, id, site
