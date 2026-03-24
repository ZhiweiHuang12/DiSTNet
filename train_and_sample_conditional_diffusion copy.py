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
from src.data.mnist import get_mnist_loader_and_transform
from src.data.cifar10 import get_cifar10_loader_and_transform
from src.data.afl import get_afl_loader_and_transform
from src.data.on_off_nm import get_on_off_nm_data
from src.condition_diffusion.utils_plot import *

from torchvision.utils import save_image, make_grid
from src.condition_diffusion.train import train
import matplotlib.pyplot as plt

import torch.backends
# import torch.backends.mps
def plot_compare_prob(predicted_sample,tagert_sample_np,figure_name):
    time_instants = [1,2,3,5,7]
    plt.clf()
    # fig, ax = generate_canvas([1,5],width_size = 1.2,height_size=0.3)
    fig, ax = generate_canvas([1,5],width_size = 1.2,height_size=0.15)

    time_index = 0   
    for time_instant in time_instants: 
        nn_ssa_sample = np.vstack((predicted_sample[:,time_instant],tagert_sample_np[:,time_instant])).T
        # nn_ssa_sample = np.vstack((tagert_sample_np[:,time_instant],tagert_sample_np[:,time_instant])).T

        # temp = ax[d].hist(nn_ssa_sample, bins = bins, stacked=False, density=True, color=colors)
        hist_plot(nn_ssa_sample,ax[time_index])
        time_index = time_index + 1
    plt.tight_layout()
    # figure_name = "figures/hist_{}_rescaled_comparison.png".format(dataset)
    plt.savefig(figure_name)
    plt.close()

    predicted_mean = np.mean(predicted_sample, axis=0)
    target_mean = np.mean(tagert_sample_np, axis=0)
    correlation_matrix = np.corrcoef(predicted_mean, target_mean)
    # Extract correlation coefficient between the two arrays
    correlation = correlation_matrix[0, 1]
    print("Correlation coefficient:", correlation)

def calculate_kl(sample1,sample2):
    # Calculate frequency of each integer value in the sample set
    values = np.arange(max(np.max(sample1), np.max(sample2)) + 1)
    pmf1 = np.array([np.sum(sample1 == v) for v in values]) / len(sample1)
    pmf2 = np.array([np.sum(sample2 == v) for v in values]) / len(sample2)
    # Avoid division by zero
    pmf1 += 1e-10
    pmf2 += 1e-10
    # Calculate KL divergence
    kl_divergence = entropy(pmf1, pmf2)
    return kl_divergence

T = 200
dataset = "cifar10" # can be "cifar10" or "mnist"
dataset = "afl" 
dataset = "on_off_nm" 
# for dataset in ["afl","on_off_nm","nm_nm"]:
# for dataset in ["afl"]:
dataset = "on_off_nm"
dataset = "nm_nm"

# for dataset in ["afl","on_off_nm","nm_nm"]:
for dataset in ["toggle_switch","center_dogma"]:

    weight_dir = "weights/{}/".format(dataset)
    os.makedirs(weight_dir, exist_ok=True)
    PATH_TO_READY_MODEL = None # input path for ready model
    PATH_TO_READY_MODEL = weight_dir + "model_{}.pth".format(38) # input path for ready model

    PATH_TO_SAVE_MODEL = weight_dir + "model_{}.pth".format(dataset)
    EPOCHS = 40 # for cifar10 it should be more than 1000, but for mnist 20-100 should be okay
    EPOCHS = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for dataset in ["afl","on_off_nm"]:

    if dataset == "mnist":
        data = get_mnist_loader_and_transform()
    elif dataset == "cifar10":
        data = get_cifar10_loader_and_transform()
    elif dataset=="afl":
        # data = get_afl_data()
        max_value = 100.0
        data_dir = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/"
        json_file = data_dir + "autoFeedbackLoop/(autoFeedbackLoop)_(5000).json" #5000
        with open(json_file, 'r') as f:
            json_data = json.load(f)# Parse JSON file
        data = get_afl_loader_and_transform(json_data = json_data,max_value=max_value)

        # from src.data.afl import *

        # dataset,train_dataloader = get_dataloader(json_data[:100])
        print("sucessful load data")
    elif dataset=="nm_nm":
        # data = get_on_off_nm_data()
        json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/nm_nm_model/(nm_nm_model)_(2001).json"
        with open(json_file, 'r') as f:
            json_data = json.load(f)  # Parse JSON file
        max_value = 90.0
        data = get_afl_loader_and_transform(json_data = json_data,max_value=max_value)
        PATH_TO_READY_MODEL = weight_dir + "model_{}.pth".format(32) # input path for ready model

    elif dataset=="on_off_nm":
        # data = get_on_off_nm_data()
        json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/on_off_nm/(on_off_nm)_(4000).json"
        with open(json_file, 'r') as f:
            json_data = json.load(f)  # Parse JSON file

        max_value = 180.0
        data = get_afl_loader_and_transform(json_data = json_data,max_value=max_value)
        PATH_TO_READY_MODEL = weight_dir + "model_{}.pth".format(38) # input path for ready model

    elif dataset=="center_dogma":
        json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/center_dogma/(center_dogma)_(3500).json"
        with open(json_file, 'r') as f:
            json_data = json.load(f)  # Parse JSON file
        max_li = []
        for i in range(len(json_data)):
            max_li.append(np.max(np.array(json_data[i]["counts"])))
        max_value = max(max_li)
        data = get_afl_loader_and_transform(json_data = json_data,max_value=max_value)
        PATH_TO_READY_MODEL = weight_dir + "model_{}.pth".format(4)
        PATH_TO_READY_MODEL = None
    elif dataset=="toggle_switch":
        json_file = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/MasterDPM/data/toggle_switch/(toggle_switch)_(1500).json"
        with open(json_file, 'r') as f:
            json_data = json.load(f)  # Parse JSON file
        max_li = []
        for i in range(len(json_data)):
            max_li.append(np.max(np.array(json_data[i]["counts"])))
        max_value = max(max_li)
        data = get_afl_loader_and_transform(json_data = json_data,max_value=max_value)
        PATH_TO_READY_MODEL = weight_dir + "model_{}.pth".format(4)
        # PATH_TO_READY_MODEL = None

    ddpm = DDPM(
        T = T,
        p_cond=0.2,
        eps_model=UNet(
            in_channels=data.in_channels,
            out_channels=data.out_channels,
            T=T+1,
            num_classes=data.num_classes,
            steps=data.recommended_steps,
            attn_step_indexes=data.recommended_attn_step_indexes
        ),
        device=device
    )

    if PATH_TO_READY_MODEL is not None:
        ddpm.load_state_dict(torch.load(PATH_TO_READY_MODEL, map_location=device))
    else:
        PATH_TO_READY_MODEL =  weight_dir + "model_{}.pth".format(4)
        if os.path.exists(PATH_TO_READY_MODEL):
            ddpm.load_state_dict(torch.load(PATH_TO_READY_MODEL, map_location=device))
            print("continue train")
        _, val_losses = train(
            model=ddpm,
            optimizer=torch.optim.Adam(params=ddpm.parameters(), lr=2e-4),
            epochs=EPOCHS,
            device=device,
            train_dataloader=data.train_loader,
            val_dataloader=data.val_loader,
            save_dir = weight_dir,
            json_data = json_data[:-100],
            max_value=max_value
        )

        path = PATH_TO_SAVE_MODEL if PATH_TO_SAVE_MODEL is not None else "model.pth"
        torch.save(ddpm.state_dict(), path)


    step=500
    n_samples = 500
    tagert_sample_np_li = []
    predicted_sample_li = []
    kl_li = []

    for j in range(20): #40
        print("Sampling samples",j)
        cond_sample  = [data.val_dataset[i+j*step][1] for i in range(n_samples)]
        cond_sample = torch.stack(cond_sample)
        tagert_sample= [data.val_dataset[i+j*step][0].squeeze(0) for i in range(n_samples)]
        
        # Step 2: Concatenate along the first dimension using torch.cat
        # tagert_sample_np = torch.cat(tagert_sample, dim=0).numpy()[ 1::2,:]
        tagert_sample_np = torch.cat(tagert_sample, dim=0).numpy()
        tagert_sample_np = tagert_sample_np.reshape(tagert_sample_np.shape[0]//2,2,tagert_sample_np.shape[1])

        tagert_sample_np_li.append(tagert_sample_np)

        x_t = ddpm.sample(
            n_samples=n_samples,
            size=data.train_dataset[0][0].shape,
            classes=list(range(n_samples)),
            # w=0.2,
            w=0,
            cond_sample=cond_sample
        )
        result = []
        for i in range(x_t.shape[0]):
            result.append(data.transform_to_pil(x_t[i]))
        squeezed_arrays = [np.squeeze(array, axis=-1) for array in result]  
        # Step 2: Concatenate along the first dimension using np.concatenate
        # predicted_sample = np.concatenate(squeezed_arrays, axis=0)[1::2,:] 
        predicted_sample = np.concatenate(squeezed_arrays, axis=0)
        predicted_sample = predicted_sample.reshape(predicted_sample.shape[0]//2,2,predicted_sample.shape[1])

        predicted_sample_li.append(predicted_sample)

        # figure_name =  "figures/hist_{}_rescaled_comparison_{}.png".format(dataset,j)
        # plot_compare_prob(predicted_sample,tagert_sample_np,figure_name)

    np.savez('result/sample_{}.npz'.format(dataset), predicted_sample=predicted_sample_li, target_sample=tagert_sample_np_li)
# data_sample = np.load('result/sample_{}.npz'.format(dataset))

# kl_dist = np.empty((10, 1,28))
# for i in range(10):
#     for j in range(28):
#         kl_dist[i,0,j] = calculate_kl(tagert_sample_np_li[i][:,j],predicted_sample_li[i][:,j])

# np.save("kl_dist.npy", kl_dist)