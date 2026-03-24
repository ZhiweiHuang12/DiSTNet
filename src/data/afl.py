import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import torchvision
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from dataclasses import dataclass
from src.data.data_common import DataPack
from src.data.data_common import *
class CifarTransformation:
    def __init__(self,max_value):
        # Define an attribute and assign it a value of 10
        self.max_value = max_value    
    def __call__(self, tensor: torch.Tensor):
        return (tensor * self.max_value/2 + self.max_value/2).long().clip(0,self.max_value).permute(1,2,0).detach().cpu().numpy()

def preprocess_numpy(image_np,max_value):
    # Convert to [0, 1] range
    image_np = image_np /max_value
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

# Since we are applying a handwritten digit generation framework, the input should be n*28*28 where 28 represents the image size
def get_combine_data(data):
    # Perform sampling with replacement 7000 times, each time drawing one sample (shape 2, 31)
    if data.shape[1]>2:
        data = data[:,1:,:]
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
# def get_afl_data(batch_size: int = 256,data_path=""):
#     # # Assume we have 20,000 (30, 2) NumPy arrays
#     # data = np.random.rand(20000, 30, 2)  # Example data
#     # # Create Dataset and DataLoader
#     # # [128,1,28,28]
#     # dataset = MyDataset(data)
#     # dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)  # Batch size 64, using 4 threads
#     transform_to_tensor = transforms.ToTensor()

#     json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/autoFeedbackLoop/(autoFeedbackLoop)_(5000).json" #5000
#     with open(json_file, 'r') as f:
#         json_data = json.load(f)[:3000]  # Parse JSON file

#     # data_li = [transform_to_tensor(np.array(json_data[i]["counts"]).astype("float")) for i in range(len(json_data))]
#     data_li = [get_combine_data(np.array(json_data[i]["counts"]).astype("float")) for i in range(len(json_data))]
#     data_np = np.vstack(data_li)

#     valid_data_np = np.expand_dims(data_np, axis=1)[:,:,:28,:28]

#     data_np = preprocess_numpy(data_np)
#     new_data_np = np.expand_dims(data_np, axis=1)[:,:,:28,:28]
            
#     label_li = []
#     parameter_repeat =500 # Because each set of parameters corresponds to 1000 samples
#     for i in range(len(json_data)):
#         label = json_data[i]["parameter"]
#         label = np.tile(label, (parameter_repeat, 1))
#         label_li.append(label)
#     label_np = np.vstack(label_li)
#     transform_to_pil = CifarTransformation()
#     dataset = MyDataset(new_data_np,label_np)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # Batch size 64, using 4 threads
#     train_dataset = dataset
#     train_dataloader = dataloader
#     dataset = MyDataset(valid_data_np,label_np)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)  
#     val_dataset=dataset
#     val_dataloader=dataloader,
#     # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Batch size 64, using 4 threads
#     return DataPack(
#         train_dataset=train_dataset,
#         train_loader=train_dataloader,
#         val_dataset=val_dataset,
#         val_loader=val_dataloader,
#         transform_to_tensor=transform_to_tensor,
#         transform_to_pil=transform_to_pil,
#         in_channels=1,
#         out_channels=1,
#         num_classes=label_np.shape[1],
#         recommended_steps=(1,2,4),
#         recommended_attn_step_indexes=[1]
#     )

def get_dataloader(json_data=[],batch_size=256,train_flag=True,max_value=100):
    if len(json_data)==0:
        json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/autoFeedbackLoop/(autoFeedbackLoop)_(5000).json" #5000
        with open(json_file, 'r') as f:
            json_data = json.load(f)[:3000]  # Parse JSON file
    data_li = [get_combine_data(np.array(json_data[i]["counts"]).astype("float")) for i in range(len(json_data))]
    data_np = np.vstack(data_li)
    
    if train_flag:
        data_np = preprocess_numpy(data_np,max_value=max_value)
    data_np = np.expand_dims(data_np, axis=1)[:,:,:28,:28]
    label_li = []
    parameter_repeat =500 # Because each set of parameters corresponds to 500 samples
    for i in range(len(json_data)):
        label = json_data[i]["parameter"]
        label = np.tile(label, (parameter_repeat, 1))
        label_li.append(label)
    label_np = np.vstack(label_li)
    dataset = MyDataset(data_np,label_np)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train_flag, num_workers=4)  
    return dataset,dataloader


def get_afl_loader_and_transform(batch_size: int = 256,json_data=[],max_value=100):
    transform_to_tensor = transforms.ToTensor()
    transform_to_pil = CifarTransformation(max_value=max_value)

    # json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/autoFeedbackLoop/(autoFeedbackLoop)_(5000).json" #5000
    # with open(json_file, 'r') as f:
    #     json_data = json.load(f)# Parse JSON file

    train_dataset,train_dataloader = get_dataloader(json_data[:1000],max_value=max_value)
    val_dataset,val_dataloader = get_dataloader(json_data[-1000:],train_flag=False,max_value=max_value)
    # val_dataset,val_dataloader = get_dataloader(json_data[:1000],train_flag=False,max_value=max_value)

    return DataPack(
        train_dataset=train_dataset,
        train_loader=train_dataloader,
        val_dataset=val_dataset,
        val_loader=val_dataloader,
        transform_to_tensor=transform_to_tensor,
        transform_to_pil=transform_to_pil,
        in_channels=1,
        out_channels=1,
        num_classes=train_dataset[0][1].shape[0],
        recommended_steps=(1,2,4),
        recommended_attn_step_indexes=[1]
    )
