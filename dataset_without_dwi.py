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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定随机种子以确保结果可复现
def set_random_seed(seed: int):
    """
    固定随机种子以确保结果可复现。
    
    参数:
    seed (int): 用于随机数生成的种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(f"Random seed set to {seed}", flush=True)
    
# 调用该函数并传入一个种子值
set_random_seed(42)  # 使用种子值42

time_point = 10

def zscore_standardize_tensor(tensor):
    """
    对 PyTorch 张量进行 Z-score 标准化 (沿着指定的轴)。
    假设张量的形状是 (H, W, C) 或 (B, H, W, C)

    Args:
        tensor (torch.Tensor): 要标准化的 PyTorch 张量。

    Returns:
        torch.Tensor: 标准化后的张量。  如果输入是 None，则返回 None。
        如果张量所有值都相同，返回原始张量（不做标准化）。
    """
    if tensor is None:
        return None

    # 将 PyTorch 张量转换为 NumPy 数组
    tensor_np = tensor.cpu().numpy()

    #展平
    flattened_tensor = tensor_np.reshape(-1)

    # 检查标准差是否为零
    if np.std(flattened_tensor) == 0:
        print("警告：张量中的所有值都相同，无法进行 Z-score 标准化。返回原始张量。")
        return tensor

    # 对 NumPy 数组进行 Z-score 标准化
    standardized_np = zscore(flattened_tensor)

    # 转换回 PyTorch 张量,并调整形状
    standardized_tensor = torch.from_numpy(standardized_np).reshape(tensor.shape).float()

    # 如果原始张量在 GPU 上，则将结果移回 GPU
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
        # 将 patient_id 转为 str
        if patient_id in self.loaded_data_cache:
            return self.loaded_data_cache[patient_id]
        
        t2 = None
        dwi = None
        
        # 在所有 T2 目录中搜索 T2 数据
        for t2_dir_key in [key for key in self.mri_dirs.keys() if 't2' in key.lower()]:
            t2_dir = self.mri_dirs[t2_dir_key]
            t2_file = os.path.join(t2_dir, patient_id, 'T2_with_mask.npy')
            if os.path.exists(t2_file):
                t2 = np.load(t2_file)
                t2 = torch.tensor(t2, dtype=torch.float32)
                break  # 找到后停止搜索
        
        # 如果未找到 T2 数据，设置为 None
        if t2 is None:
            t2 = None
        
        # 在所有 DWI 目录中搜索 DWI 数据
        for dwi_dir_key in [key for key in self.mri_dirs.keys() if 'dwi' in key.lower()]:
            dwi_dir = self.mri_dirs[dwi_dir_key]
            dwi_file = os.path.join(dwi_dir, patient_id, 'DWI_with_mask.npy')
            if os.path.exists(dwi_file):
                dwi = np.load(dwi_file)
                dwi = torch.tensor(dwi, dtype=torch.float32)
                break  # 找到后停止搜索
        
        # 如果未找到 DWI 数据，设置为 None
        if dwi is None:
            dwi = None
        
        # 缓存数据以备后续访问
        self.loaded_data_cache[patient_id] = (t2, dwi)
        return t2, dwi

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        t2, dwi = self.load_mri_data(patient_id)
               
        
        # 对 T2 和 DWI 应用相同的随机反转
        if self.transform:
            # 为当前样本生成随机种子，确保 T2 和 DWI 增强一致
            if (t2 is not None)&(dwi is not None):
                combined = torch.cat([t2.unsqueeze(0), dwi.unsqueeze(0)], dim=0)
                combined_augmented = self.transform(combined)
                t2,dwi = torch.split(combined_augmented, 1, dim=0)
                t2 = t2.squeeze(0)
                dwi = dwi.squeeze(0)
            if t2 is None:
                dwi = self.transform(dwi)
            if dwi is None:
                t2 = self.transform(t2)
        
        # 对 T2 数据进行归一化和维度调整
        if t2 is not None:
            # t2 = t2 / 2000.0
            t2 = zscore_standardize_tensor(t2)
            t2 = t2.unsqueeze(-1)
            t2 = t2.permute(2, 0, 1, 3)
        
        # 对 DWI 数据进行归一化和维度调整
        if dwi is not None:
            #dwi = dwi / 2000.0
            dwi = zscore_standardize_tensor(dwi)
            dwi = dwi.unsqueeze(-1)
            dwi = dwi.permute(2, 0, 1, 3)
        
        # 提取表格特征
        table_row = self.table_csv[self.table_csv['PatientID'] == patient_id]
        table_features = table_row.drop(columns='PatientID').values.squeeze().astype('float32')
        table_features = torch.tensor(table_features, dtype=torch.float32)
        patient_id = int(patient_id)
        
        return {
            'T2': t2,
            'dwi': dwi,  # 如果 DWI 缺失，这里会是 None
            'table': table_features,
            'patient_id': patient_id,
        }

# 使用 MONAI 的 RandFlip 进行随机反转
class Transform:
    def __init__(self, augment=True):
        self.augment = augment
        # 定义 MONAI 的随机反转操作
        self.transform = Compose([
            RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),  # Random rotation
            RandFlip(spatial_axis=[1], prob=0.5),                      # Random flip
            RandZoom(min_zoom=0.9, max_zoom=1.3, prob=0.5),               # Random zoom
            RandGaussianNoise(prob=0.5),                                  # Random Gaussian noise
            RandBiasField(prob=0.5),                                      # Random bias field
            ToTensor(),                                                   # Convert to tensor
])
        self.transform2 = Compose([
            RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),  # Random rotation
            RandFlip(spatial_axis=[0], prob=0.5),                      # Random flip
            RandZoom(min_zoom=0.9, max_zoom=1.3, prob=0.5),               # Random zoom
            RandGaussianNoise(prob=0.5),                                  # Random Gaussian noise
            RandBiasField(prob=0.5),                                      # Random bias field
            ToTensor(),                                                   # Convert to tensor
])

    def __call__(self, image):
        # 如果image的shape为(2, 256, 256, 20)
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
    # test_transform = Transform(augment=False)

    train_data = MultiModalDataset(mri_dirs, X_train_np, transform=trian_transform)
    val_data = MultiModalDataset(mri_dirs, X_val_np, transform=val_transform)
    test_data = MultiModalDataset(mri_dirs, X_test_np, transform=val_transform)

    return train_data, val_data, test_data


def data(table_dir, mri_dirs, fold_num):
    fold_dirs = [os.path.join(table_dir, f'fold_{i+1}') for i in range(5)]
    fold_dir = fold_dirs[fold_num - 1]
    print(f"Processing fold {fold_num} in {fold_dir}", flush=True)

    # 加载数据集
    train_data, val_data, test_data = generate_dataset_from_csv(fold_dir, mri_dirs)

    print(f"Total number of samples in train dataset: {len(train_data)}", flush=True)
    print(f"Total number of samples in validation dataset: {len(val_data)}", flush=True)
    print(f"Total number of samples in test dataset: {len(test_data)}", flush=True)

    def extract_data(dataset):
        """
        从数据集中提取T2、DWI、表格数据、标签Y、事件状态D和患者ID。
        如果DWI缺失，则DWI为None。
        """
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
            X_dwi.append(item['dwi'])  # 如果DWI缺失，这里会是None
            X_table.append(item['table'][0:-3])
            Y.append(item['table'][-3])
            D.append(item['table'][-2])
            id.append(item['patient_id'])
            site.append(item['table'][-1])
        
        return np.stack(X_T2), X_dwi, np.stack(X_table), np.stack(Y), np.stack(D), np.stack(id), np.stack(site)

    def create_tensor_data(X_T2, X_dwi, X_table, Y, Y_d, D, id, site):
        """
        将数据转换为PyTorch张量。
        如果DWI缺失，则保持为None。
        """
        X_T2 = torch.tensor(X_T2, dtype=torch.float32, device=device)
        X_table = torch.tensor(X_table, dtype=torch.float32, device=device)
        Y = torch.tensor(Y, dtype=torch.int64, device=device)
        Y_d = torch.tensor(Y_d, dtype=torch.int64, device = device)
        D = torch.tensor(D, dtype=torch.int32, device=device)
        id = torch.tensor(id, dtype=torch.int32, device=device)
        site = torch.tensor(site, dtype=torch.int32, device=device)
        
        # 处理DWI数据，如果为None则保持为None
        X_dwi_tensor = []
        for dwi in X_dwi:
            if dwi is not None:
                X_dwi_tensor.append(dwi.clone().detach().to(dtype=torch.float32, device=device))
            else:
                X_dwi_tensor.append(None)
        
        return list(zip(X_T2, X_dwi_tensor, X_table, Y, Y_d, D, id, site))

    def split_data_by_dwi(X_T2, X_dwi, X_table, Y, Y_d, D, id, site):
        """
        将数据分为有DWI和缺失DWI的两部分。
        """
        has_dwi_indices = [i for i, dwi in enumerate(X_dwi) if dwi is not None]
        missing_dwi_indices = [i for i, dwi in enumerate(X_dwi) if dwi is None]

        # 有DWI的数据
        X_T2_has_dwi = X_T2[has_dwi_indices]
        X_dwi_has_dwi = [X_dwi[i] for i in has_dwi_indices]
        X_table_has_dwi = X_table[has_dwi_indices]
        Y_has_dwi = Y[has_dwi_indices]
        Y_d_has_dwi = Y_d[has_dwi_indices]
        D_has_dwi = D[has_dwi_indices]
        id_has_dwi = id[has_dwi_indices]
        site_has_dwi = site[has_dwi_indices]

        # 缺失DWI的数据
        X_T2_missing = X_T2[missing_dwi_indices]
        X_dwi_missing = [None] * len(missing_dwi_indices)  # DWI字段设置为None
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

    # 提取训练、验证和测试数据
    X_train_T2, X_train_dwi, X_train_table, Y_train_np, D_train_np, train_id, train_site = extract_data(train_data)
    X_val_T2, X_val_dwi, X_val_table, Y_val_np, D_val_np, val_id, val_site = extract_data(val_data)
    X_test_T2, X_test_dwi, X_test_table, Y_test_np, D_test_np, test_id, test_site = extract_data(test_data)


    # 处理时间离散化
    Y_train_discrete_np, D_train_discrete_np, Y_val_discrete_np, D_val_discrete_np, label_transform, time_grid_train_np =\
        time_interview(time_point, Y_train_np, D_train_np, Y_val=Y_val_np, D_val=D_val_np)
    Y_test_discrete_np, D_test_discrete_np = label_transform.transform(Y_test_np, D_test_np)


    # 将训练集分为有DWI和缺失DWI的两部分
    (X_train_T2_has_dwi, X_train_dwi_has_dwi, X_train_table_has_dwi, Y_train_has_dwi, Y_train_discrete_has_dwi, D_train_has_dwi, train_id_has_dwi, train_site_has_dwi),\
    (X_train_T2_missing, X_train_dwi_missing, X_train_table_missing, Y_train_missing, Y_train_discrete_missing, D_train_missing, train_id_missing, train_site_missing) =\
        split_data_by_dwi(X_train_T2, X_train_dwi, X_train_table, Y_train_np, Y_train_discrete_np, D_train_discrete_np, train_id, train_site)

    # 创建有DWI的训练数据集
    train_data = create_tensor_data(X_train_T2_has_dwi, X_train_dwi_has_dwi, X_train_table_has_dwi, Y_train_has_dwi, Y_train_discrete_has_dwi,D_train_has_dwi, train_id_has_dwi, train_site_has_dwi)

    # 创建缺失DWI的训练数据集
    X_train_wdwi = list(zip(
        torch.tensor(X_train_T2_missing, dtype=torch.float32, device=device),
        X_train_dwi_missing,  # DWI字段为None
        torch.tensor(X_train_table_missing, dtype=torch.float32, device=device),
        torch.tensor(Y_train_missing, dtype=torch.int64, device=device),
        torch.tensor(Y_train_discrete_missing,dtype=torch.int64, device=device),
        torch.tensor(D_train_missing, dtype=torch.int32, device=device),
        torch.tensor(train_id_missing, dtype=torch.int32, device=device),
        torch.tensor(train_site_missing, dtype=torch.int32, device=device)
    ))

    # 将验证集分为有DWI和缺失DWI的两部分
    (X_val_T2_has_dwi, X_val_dwi_has_dwi, X_val_table_has_dwi, Y_val_has_dwi, Y_val_discrete_has_dwi, D_val_has_dwi, val_id_has_dwi, val_site_has_dwi),\
    (X_val_T2_missing, X_val_dwi_missing, X_val_table_missing, Y_val_missing, Y_val_discrete_missing,D_val_missing, val_id_missing, val_site_missing) =\
        split_data_by_dwi(X_val_T2, X_val_dwi, X_val_table, Y_val_np ,Y_val_discrete_np, D_val_discrete_np, val_id, val_site)

    # 创建有DWI的验证数据集
    val_data = create_tensor_data(X_val_T2_has_dwi, X_val_dwi_has_dwi, X_val_table_has_dwi, Y_val_has_dwi, Y_val_discrete_has_dwi,D_val_has_dwi, val_id_has_dwi, val_site_has_dwi)

    # 创建缺失DWI的验证数据集
    X_val_wdwi = list(zip(
        torch.tensor(X_val_T2_missing, dtype=torch.float32, device=device),
        X_val_dwi_missing,  # DWI字段为None
        torch.tensor(X_val_table_missing, dtype=torch.float32, device=device),
        torch.tensor(Y_val_missing, dtype=torch.int64, device=device),
        torch.tensor(Y_val_discrete_missing,dtype=torch.int64, device=device),
        torch.tensor(D_val_missing, dtype=torch.int32, device=device),
        torch.tensor(val_id_missing, dtype=torch.int32, device=device),
        torch.tensor(val_site_missing, dtype=torch.int32, device=device)
    ))

    # 将测试集分为有DWI和缺失DWI的两部分
    (X_test_T2_has_dwi, X_test_dwi_has_dwi, X_test_table_has_dwi, Y_test_has_dwi, Y_test_discrete_has_dwi, D_test_has_dwi, test_id_has_dwi, test_site_has_dwi),\
    (X_test_T2_missing, X_test_dwi_missing, X_test_table_missing, Y_test_missing, Y_test_discrete_missing, D_test_missing, test_id_missing, test_site_missing) =\
        split_data_by_dwi(X_test_T2, X_test_dwi, X_test_table, Y_test_np, Y_test_discrete_np, D_test_discrete_np, test_id, test_site)

    # 创建有DWI的测试数据集
    test_data = create_tensor_data(X_test_T2_has_dwi, X_test_dwi_has_dwi, X_test_table_has_dwi, Y_test_has_dwi, Y_test_discrete_has_dwi, D_test_has_dwi, test_id_has_dwi, test_site_has_dwi)

    # 创建缺失DWI的测试数据集
    X_test_wdwi = list(zip(
        torch.tensor(X_test_T2_missing, dtype=torch.float32, device=device),
        X_test_dwi_missing,  # DWI字段为None
        torch.tensor(X_test_table_missing, dtype=torch.float32, device=device),
        torch.tensor(Y_test_missing, dtype=torch.int64, device=device),
        torch.tensor(Y_test_discrete_missing, dtype=torch.int64, device=device),
        torch.tensor(D_test_missing, dtype=torch.int32, device=device),
        torch.tensor(test_id_missing, dtype=torch.int32, device=device),
        torch.tensor(test_site_missing, dtype=torch.int32, device=device)
    ))
    

    # 打印数据形状
    print("Training data with DWI:")
    # 将X_train_dwi_has_dwi从list转为tensor
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
    """
    自定义 collate_fn，处理包含 None 值的批次数据。
    """
    # 过滤掉 None 值
    batch = [item for item in batch if item is not None]
    
    # 如果批次为空，返回 None
    if len(batch) == 0:
        return None
    
    # 分别提取 T2、DWI、table、Y、D 和 id
    T2 = [item[0] for item in batch]
    dwi_shape = torch.zeros(10, 126, 126, 1)  # 用于创建全零张量
    DWI = [item[1] if item[1] is not None else torch.zeros_like(dwi_shape) for item in batch]  # 如果 DWI 为 None，用全零张量代替
    table = [item[2] for item in batch]
    Y = [item[3] for item in batch]
    Y_d = [item[4] for item in batch]
    D = [item[5] for item in batch]
    id = [item[6] for item in batch]
    site = [item[7] for item in batch]

    # 将列表堆叠为张量
    T2 = torch.stack(T2)
    DWI = torch.stack(DWI)
    table = torch.stack(table)
    Y = torch.stack(Y)
    Y_d = torch.stack(Y_d)
    D = torch.stack(D)
    id = torch.stack(id)
    site = torch.stack(site)

    return T2, DWI, table, Y,Y_d, D, id, site
