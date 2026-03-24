import numpy as np
from scipy.optimize import differential_evolution
import numpy as np
from scipy.stats import poisson
from scipy.special import rel_entr
import pickle
import os
import torch
import json
import pandas as pd
import sys
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/conditionalDM/stable-diffusion-from-scratch-main/"
os.chdir(root_path)
sys.path.append(root_path)

from sampler import sample
from src.data.afl import get_afl_loader_and_transform
from src.data.afl import CifarTransformation


def filter_p_sample(samples_p,max_val):
    samples_p = samples_p[samples_p<=max_val]
    samples_p[samples_p<0]=0
    sample_counts = np.bincount(samples_p,minlength=max_val+1)
    sample_p = sample_counts / len(samples_p)
    return sample_p
def simulate_p(args,theta,weight_path,max_val_li):
    theta = torch.tensor(theta).repeat(args["n_samples"], 1)
    transform_to_pil = CifarTransformation(max_value=args["max_value"])
    result = sample(args,theta,weight_path,transform_to_pil)

    # result = sample(args,[theta],fold_weights=fold_weights,weight_id=weight_id)[0][:,0,4:,1].cpu().numpy()
    result_prob = [filter_p_sample(result[:,i],max_val_li[i]) for i in range(result.shape[1])]
    return result_prob
# Calculate KL divergence
def kl_divergence(q, p):
    return np.sum(q * np.log((q+0.0001) / (p+0.0001)))


def infer(args,weight_path,param_data,result_dir="result/inference/",epochs = 20):
    
    model_name = "autoFeedbackLoop"
    rub = np.array([1, 0.1, 5, 20])
    rlb = np.array([0.01, 0.01, 0.01, 0.01])
    lr = np.array([0.01, 0.01, 0.5, 5])
    epsilon = 1e-10
    
    sample_number = 3
    r0 = [0.08, 0.05, 0.2, 15]
    rate_samples_all = []
    true_val_all = []
    for data_index in range(sample_number):
        
        observed_data = np.squeeze(np.array(param_data[data_index]["counts"])[:,1::2,:28],axis=1)
        max_val_li = observed_data.max(axis=0).astype("int")
        true_val = np.array(param_data[data_index]["parameter"])
        print("the true val is",true_val)
        rate = r0
        rate_samples = []
        standar_normal_val_li = [np.random.standard_normal(4) for i in range(epochs)]
        for i in range(epochs):
            x = observed_data
            # standar_normal_val = np.random.standard_normal(4)
            dr = lr * standar_normal_val_li[i]
            print("the dr is",dr)
            new_rate = rate + dr
            
            out_idx = new_rate > rub
            new_rate[out_idx] = rub[out_idx]
            out_idx = new_rate < rlb
            new_rate[out_idx] = rlb[out_idx]
            r = rate
            nr = new_rate
            prob = simulate_p(args,r,weight_path = weight_path,max_val_li=max_val_li)
            nprob = simulate_p(args,nr,weight_path = weight_path,max_val_li=max_val_li)
            observed_prob = np.array([prob[i][observed_data[:,i].astype("int")]+epsilon for i in range(2,len(prob))]) 
            observed_nprob = np.array([nprob[i][observed_data[:,i].astype("int")]+epsilon for i in range(2,len(nprob))]) 

            gamma = observed_nprob / observed_prob
            a = gamma[gamma>1].shape[0]
            b = gamma[gamma<1].shape[0]
            pth = a / b if b > 0 else 1
            pselect = np.random.rand()
            if pselect < pth:
                rate = new_rate

            print(f"epoch: {i}, rate: {rate}")
            rate_samples.append(rate)
        true_val_all.append(true_val)
        rate_samples_all.append(rate_samples)

    print(rate_samples)
    np.savez(result_dir+"/{}_autoreg_inference_rate_sample".format(model_name), data=np.array(rate_samples_all),true_val=np.array(true_val_all))


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

    data_dir = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/"
    json_file = data_dir + "autoFeedbackLoop/(autoFeedbackLoop)_(4).json" #5000
    with open(json_file, 'r') as f:
        param_data = json.load(f)# Parse JSON file
    infer(config_dict,weight_path,param_data,result_dir="result/inference/",epochs = 100)