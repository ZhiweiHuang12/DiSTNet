from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Any, Tuple, List

@dataclass
class DataPack:
    train_dataset: Dataset
    train_loader: DataLoader
    val_dataset: Dataset
    val_loader: DataLoader
    transform_to_tensor: Any
    transform_to_pil: Any
    in_channels: int
    out_channels: int
    num_classes: int
    recommended_steps: Tuple[int]
    recommended_attn_step_indexes: List[int]




def get_dataloader(json_data=[],train_flag=True,max_value=100):
    if len(json_data)==0:
        json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/autoFeedbackLoop/(autoFeedbackLoop)_(5000).json" #5000
        with open(json_file, 'r') as f:
            json_data = json.load(f)[:3000]  # Parse JSON file
    data_li = [get_combine_data(np.array(json_data[i]["counts"]).astype("float"),max_value = max_value) for i in range(len(json_data))]
    data_np = np.vstack(data_li)
    if train_flag:
        data_np = preprocess_numpy(data_np)
    data_np = np.expand_dims(data_np, axis=1)[:,:,:28,:28]
    label_li = []
    parameter_repeat =500 # Because each set of parameters corresponds to 500 samples
    for i in range(len(json_data)):
        label = json_data[i]["parameter"]
        label = np.tile(label, (parameter_repeat, 1))
        label_li.append(label)
    label_np = np.vstack(label_li)
    dataset = MyDataset(new_data_np,label_np)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)  
    return dataset,dataloader


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

def get_combine_data(data):
    # Perform sampling with replacement 7000 times, each time drawing one sample (shape 2, 31)
    samp_number = 7000
    group_number = samp_number//14
    index = np.random.choice(data.shape[0], size=samp_number, replace=True)
    sampled_data =  data[index]
    # Divide 7000 samples into 500 groups, each group contains 14 samples
    result = sampled_data.reshape(group_number, 14, 2, 31)
    # Combine samples in each group along the first dimension, forming the final shape (500, 28, 31)
    final_result = result.reshape(group_number, 28, 31)
    return final_result