import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import torchvision
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from dataclasses import dataclass
import os
import sys

root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/conditionalDM/stable-diffusion-from-scratch-main/"
os.chdir(root_path)
sys.path.append(root_path)

from src.data.data_common import DataPack

class CifarTransformation:    
    def __call__(self, tensor: torch.Tensor):
        return (tensor * 50 + 50).long().clip(0,100).permute(1,2,0).detach().cpu().numpy()

def preprocess_numpy(image_np):
    # Convert to [0, 1] range
    image_np = image_np /100.0
    # Normalize
    mean = np.array([0.5])
    std = np.array([0.5])
    image_np = (image_np - mean) / std
    return image_np

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert to PyTorch tensor
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]

def get_combine_data(data):

    data = data[:,1:,:]
    # Perform sampling with replacement 7000 times, each time drawing one sample (shape 2, 31)
    samp_number = 7000
    group_number = samp_number//14
    # sampled_data = np.zeros((samp_number, 2, 31))
    # for i in range(samp_number):
    #     index = np.random.choice(data.shape[0], size=1, replace=True)
    #     sampled_data[i] = data[index[0]]  # Directly draw samples from data

    index = np.random.choice(data.shape[0], size=samp_number, replace=True)
    sampled_data =  data[index]

    # Divide 7000 samples into 500 groups, each group contains 14 samples
    result = sampled_data.reshape(group_number, 14, 2, 31)

    # Combine samples in each group along the first dimension, forming the final shape (500, 28, 31)
    final_result = result.reshape(group_number, 28, 31)
    return final_result
def get_on_off_nm_data(batch_size: int = 256):
    # Assume we have 20,000 (30, 2) NumPy arrays
    # data = np.random.rand(20000, 30, 2)  # Example data
    # Create Dataset and DataLoader
    # [128,1,28,28]
    # dataset = MyDataset(data)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)  # Batch size 64, using 4 threads
    transform_to_tensor = transforms.ToTensor()

    json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/on_off_nm/(on_off_nm)_(4000).json" #5000
    with open(json_file, 'r') as f:
        json_data = json.load(f)[:2000]  # Parse JSON file

    # data_li = [transform_to_tensor(np.array(json_data[i]["counts"]).astype("float")) for i in range(len(json_data))]
    data_li = [get_combine_data(np.array(json_data[i]["counts"]).astype("float")) for i in range(len(json_data))]
    data_np = np.vstack(data_li)

    valid_data_np = np.expand_dims(data_np, axis=1)[:,:,:28,:28]

    data_np = preprocess_numpy(data_np)
    new_data_np = np.expand_dims(data_np, axis=1)[:,:,:28,:28]
            
    label_li = []
    parameter_repeat =500 # Because each set of parameters corresponds to 1000 samples
    for i in range(len(json_data)):
        label = json_data[i]["parameter"]
        label = np.tile(label, (parameter_repeat, 1))
        label_li.append(label)
    label_np = np.vstack(label_li)
    print("label shape is",label_np.shape[1])
    transform_to_pil = CifarTransformation()
    dataset = MyDataset(new_data_np,label_np)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # Batch size 64, using 4 threads
    train_dataset = dataset
    train_dataloader = dataloader
    dataset = MyDataset(valid_data_np,label_np)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)  
    
    val_dataset=dataset
    val_dataloader=dataloader,
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Batch size 64, using 4 threads
    return DataPack(
        train_dataset=train_dataset,
        train_loader=train_dataloader,
        val_dataset=val_dataset,
        val_loader=val_dataloader,
        transform_to_tensor=transform_to_tensor,
        transform_to_pil=transform_to_pil,
        in_channels=1,
        out_channels=1,
        num_classes=label_np.shape[1],
        recommended_steps=(1,2,4),
        recommended_attn_step_indexes=[1]
    )

# get_on_off_nm_data()

# for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
#     print(f"Batch {batch_idx + 1}:")
#     print(f"Data shape: {batch_data.shape}, Labels shape: {batch_labels.shape}")
#     print(f"Data: {batch_data}, Labels: {batch_labels}")
#     if batch_idx == 1:  # Exit after testing two batches
#         break

