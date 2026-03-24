import os
import sys
import numpy as np
import json
from scipy.stats import entropy
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/conditionalDM/stable-diffusion-from-scratch-main/"
os.chdir(root_path)
sys.path.append(root_path)
from src.condition_diffusion.ddpm import DDPM
from src.condition_diffusion.unet import UNet
from src.data.afl import get_afl_loader_and_transform
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import torch.backends

def sample(config_dict,cond_sample,weight_path,transform_to_pil):
    n_samples = config_dict["n_samples"]
    ddpm = DDPM(
        T = config_dict["T"],
        p_cond=config_dict["p_cond"],
        eps_model=UNet(
            in_channels=config_dict["in_channels"],
            out_channels=config_dict["out_channels"],
            T=config_dict["T"]+1,
            num_classes=config_dict["num_classes"],
            steps=config_dict["recommended_steps"],
            attn_step_indexes=config_dict["attn_step_indexes"]
        ), device=config_dict["device"]    )

    ddpm.load_state_dict(torch.load(weight_path, map_location=config_dict["device"]))
   
    x_t = ddpm.sample(
        n_samples=n_samples,
        size=config_dict["data_size"],
        classes=list(range(n_samples)),
        w=0,
        cond_sample=cond_sample
    )
    result = []
    for i in range(x_t.shape[0]):
        result.append(transform_to_pil(x_t[i]))
    squeezed_arrays = [np.squeeze(array, axis=-1) for array in result]  
    # Step 2: Concatenate along the first dimension using np.concatenate
    predicted_sample = np.concatenate(squeezed_arrays, axis=0)
    predicted_sample = predicted_sample.reshape(predicted_sample.shape[0]//2,2,predicted_sample.shape[1])

    result = predicted_sample[:,1::2,:].squeeze(1)
    return result

class CifarTransformation:
    def __init__(self,max_value):
        # Define an attribute and assign it a value of 10
        self.max_value = max_value    
    def __call__(self, tensor: torch.Tensor):
        return (tensor * self.max_value/2 + self.max_value/2).long().clip(0,self.max_value).permute(1,2,0).detach().cpu().numpy()


if __name__ == '__main__':
    # in_channels,out_channels,num_classes,recommended_steps,attn_step_indexes = 1,1,4,(1,2,4),[1]
    config_dict = {
        "in_channels":1,
        "out_channels":1,
        "num_classes":4,
        "recommended_steps":(1,2,4),
        "attn_step_indexes":[1],
        "T":200,
        "data_size":[1, 28, 28],
        "max_value":100.0,
        "p_cond":0.2,
        "n_samples":  100,
        "T": 200,
        "dataset": "afl" 
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict["device"] = device
    transform_to_pil = CifarTransformation(max_value=config_dict["max_value"])
    weight_dir = "weights/{}/".format(config_dict["dataset"])
    weight_path = weight_dir + "model_{}.pth".format(28) # input path for ready model

    # data_dir = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/"
    # json_file = data_dir + "autoFeedbackLoop/(autoFeedbackLoop)_(4).json" #5000
    # with open(json_file, 'r') as f:
    #     json_data = json.load(f)# Parse JSON file
    # data = get_afl_loader_and_transform(json_data = json_data,max_value=max_value)
    # single_parameter = json_data[0]["parameter"]

    single_parameter = [1.025, 1.025, 2.525, 25.025]
    cond_sample = torch.tensor(single_parameter).repeat(config_dict["n_samples"], 1)
    result = sample(config_dict,cond_sample,weight_path,transform_to_pil)
    print(result)
    print("done")




