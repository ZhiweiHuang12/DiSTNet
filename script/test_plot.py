import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.special import rel_entr
import matplotlib.font_manager as fm
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
import os
import sys
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/conditionalDM/stable-diffusion-from-scratch-main/"
os.chdir(root_path)
sys.path.append(root_path)
from script.utils_plot import *

def plot_autoFeedbackLoop_result():

    species_labels = ["G","P"]
    model_name = "afl"
    # model_name = "on_off_nm"
    data_sample = np.load('result/sample_{}.npz'.format(model_name))
    
    # 2000,30,2
    nn_samples_np,ssa_samples_np = data_sample["predicted_sample"].transpose(0, 1, 3, 2),data_sample["target_sample"].transpose(0, 1, 3, 2)
    figure_path = "figures/{}/".format(model_name)
    os.makedirs(figure_path, exist_ok=True)

    param_sample_number = nn_samples_np.shape[0] #param number
    traj_length = nn_samples_np.shape[1] #nb trajs per point
    time_length = nn_samples_np.shape[2] #time length
    feature_number = nn_samples_np.shape[3] #feature

    wass_dist = np.empty((param_sample_number, feature_number,time_length))
    kl_dist = np.empty((param_sample_number, feature_number,time_length))

    param_sample_number_li = list(range(param_sample_number))
    # A k represents a set of parameters
    for k in param_sample_number_li:
        nn_samples = nn_samples_np[k]
        ssa_samples = ssa_samples_np[k]

        # ssa_samples 2000,30,2
        time_instants = [1,2,3,5,7]
        plt.clf()
        fig, ax = generate_canvas([feature_number,5],width_size = 1.2,height_size=0.3)
        time_index = 0   
        for time_instant in time_instants: 
            for d in range(feature_number):
                nn_ssa_sample = np.vstack((nn_samples[:, time_instant, d], ssa_samples[:, time_instant, d])).T
                hist_plot(nn_ssa_sample,ax[d,time_index])
            time_index = time_index + 1
        plt.tight_layout()
        figure_name = figure_path+"hist_{}_comparison_{}th.png".format(model_name,k)
        plt.savefig(figure_name)
        plt.savefig(figure_name.replace("png","pdf"))

        plt.close()

        # plot compared mean of each time point
        ssa_samples_mean = np.mean(ssa_samples,axis=0)
        nn_samples_mean = np.mean(nn_samples,axis=0)

        fig, ax = generate_canvas([1,1],width_size=0.4,height_size=0.45)
        t_value = list(range(ssa_samples_mean.shape[0]))
        for species_index in range(ssa_samples_mean.shape[1]):
            # plt.plot(t_value, ssa_samples_mean[:,species_index],  linewidth=3, label="SSA_{}".format(species_index), alpha=0.7)
            # plt.plot(t_value, nn_samples_mean[:,species_index], marker='o', linestyle='None', markersize=10, alpha=0.5)
            line_scatter_plot(ssa_samples_mean[:,species_index],nn_samples_mean[:,species_index],ax,line_label=species_labels[species_index])
        figure_name = figure_path+"mean_{}_rescaled_comparison_{}th.png".format(model_name,k,time_instant)
        plt.tight_layout()
        plt.savefig(figure_name,dpi=300)
        plt.savefig(figure_name.replace("png","pdf"))
        plt.close()

        for m in range(feature_number):
            for t in range(time_length):    
                wass_dist[k,m,t] = wasserstein_distance(ssa_samples[:,t,m], nn_samples[:,t,m])
                max_values = max(ssa_samples[:,t,m])
                # kl_dist[k,m,t] = calculate_kl(ssa_samples[:,t,m], nn_samples[:,t,m],max_values = max_values)
                kl_dist[k,m,t] = calculate_kl(ssa_samples[:,t,m], nn_samples[:,t,m])

    plt.clf()
    fig, ax = generate_canvas([1,1],height_size=0.45,width_size=0.4)
    data = wass_dist[:,1,:].T
    data = kl_dist[:,1,1:].T
    line_with_shadow_plot(data,ax,fill_flag=False)

    figure_name = figure_path+"distribution_{}_comparison.png".format(model_name)
    plt.tight_layout()
    plt.savefig(figure_name,dpi=300)
    plt.savefig(figure_name.replace("png","pdf"))  
    plt.close()

def plot_on_off_nm():
    species_labels = ["nRNA","mRNA"]
    
    model_name = "on_off_nm"
    # model_name = "nm_nm"

    species_labels = ["mRNA","Protein"]
    model_name = "center_dogma"
    
    species_labels = ["Protein A","Protein B"]
    model_name = "toggle_switch"

    species_labels = ["I","R"]
    model_name = "SIRS"
    model_name = "delay_SIRS"

    data_sample = np.load('result/sample_{}.npz'.format(model_name))
    # 2000,30,2
    nn_samples_np,ssa_samples_np = data_sample["predicted_sample"].transpose(0, 1, 3, 2),data_sample["target_sample"].transpose(0, 1, 3, 2)
    figure_path = "figures/{}/".format(model_name)
    os.makedirs(figure_path, exist_ok=True)

    param_sample_number = nn_samples_np.shape[0] #param number
    traj_length = nn_samples_np.shape[1] #nb trajs per point
    time_length = nn_samples_np.shape[2] #time length
    feature_number = nn_samples_np.shape[3] #feature

    wass_dist = np.empty((param_sample_number, feature_number,time_length))
    kl_dist = np.empty((param_sample_number, feature_number,time_length))

    param_sample_number_li = list(range(param_sample_number))

    colors = ["#D22427","#D22427",'#2186AB']
    colors = ['#2186AB',"#D22427",'#2186AB']

    # A k represents a set of parameters
    # for k in range(param_sample_number):
    for k in param_sample_number_li:
        nn_samples = nn_samples_np[k]
        ssa_samples = ssa_samples_np[k]        # bins = 50
        # plot compared mean of each time point
        ssa_samples_mean = np.mean(ssa_samples,axis=0)
        nn_samples_mean = np.mean(nn_samples,axis=0)
        fig, ax = generate_canvas([1,1],width_size=0.35,height_size=0.48)
        t_value = list(range(ssa_samples_mean.shape[0]))
        for species_index in range(ssa_samples_mean.shape[1]):
            line_scatter_plot(ssa_samples_mean[:,species_index],nn_samples_mean[:,species_index],ax,line_label=species_labels[species_index],color=colors[species_index])
        figure_name = figure_path+"mean_{}_rescaled_comparison_{}th.png".format(model_name,k)
        plt.tight_layout()
        plt.savefig(figure_name,dpi=300)
        plt.savefig(figure_name.replace("png","pdf"))
        plt.close()

        for m in range(feature_number):
            for t in range(time_length):    
                wass_dist[k,m,t] = wasserstein_distance(ssa_samples[:,t,m], nn_samples[:,t,m])
                kl_dist[k,m,t] = calculate_kl(ssa_samples[:,t,m], nn_samples[:,t,m])
    fig, ax = generate_canvas([1,1],height_size=0.48,width_size=0.35)
    # data = wass_dist[:,1,:]
    # data = kl_dist[:,1,:].T
    # line_with_shadow_plot(kl_dist[:,0,:].T,ax,color=colors[1],line_label="Mean")
    # line_with_shadow_plot(kl_dist[:,1,:].T,ax,color=colors[2],line_label="Mean")
    line_with_shadow_plot(kl_dist[:,0,:].T,ax,color=colors[0],line_label="nRNA")
    line_with_shadow_plot(kl_dist[:,1,:].T,ax,color=colors[1],line_label="mRNA")
    figure_name = figure_path+"distribution_{}_comparison.png".format(model_name)
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.savefig(figure_name.replace("png","pdf"))
    plt.close()

    time_points = [5,10,15,20]
    # plot joint_prob
    for file_index in param_sample_number_li:
        nn_samples = nn_samples_np[file_index]
        ssa_samples = ssa_samples_np[file_index] 
        data = nn_samples[:,:,[0,1]][:,time_points,:]
        noise = np.random.random(data.shape)
        noised_data = (data + noise).T
        data1_nn_x_c,data1_nn_y_c = noised_data[:,0,:]
        data2_nn_x_c,data2_nn_y_c = noised_data[:,1,:]
        data3_nn_x_c,data3_nn_y_c = noised_data[:,2,:]
        data4_nn_x_c,data4_nn_y_c = noised_data[:,3,:]

        data = ssa_samples[:,:,[0,1]][:,time_points,:]
        noised_data = (data + noise).T
        data1_ssa_x_c,data1_ssa_y_c = noised_data[:,0,:]
        data1_xy_lim = np.ceil(np.max(noised_data[:,0,:],axis=1))

        data2_ssa_x_c,data2_ssa_y_c = noised_data[:,1,:]
        data2_xy_lim = np.ceil(np.max(noised_data[:,1,:],axis=1))

        data3_ssa_x_c,data3_ssa_y_c = noised_data[:,2,:]
        data3_xy_lim = np.ceil(np.max(noised_data[:,2,:],axis=1))

        data4_ssa_x_c,data4_ssa_y_c = noised_data[:,3,:]
        data4_xy_lim= np.ceil(np.max(noised_data[:,3,:],axis=1))

        label_names = ["${}$".format(el) for el in species_labels]

        try:
            # Use Seaborn to draw joint probability density plots
            # Set figure size
            width_mm=183*0.7
            # width_mm = width_mm * 0.5
            width_inch = width_mm * 0.0393701
            height_inch = width_inch * 0.45
            fig, axs = plt.subplots(2, 4, figsize=(width_inch, height_inch))
            fig = plt.figure(figsize=(width_inch,height_inch))
            gs = GridSpec(2, 4)

            # cmap_name = "GnBu"
            cmap_name = "Blues"
            fontsize = 6
            # xy_ticks = [[0,2,4,6,8,10],[0,4,8,12,16]]
            xy_ticks = [[],[]]
            g0 = plot_jointplot(data1_nn_x_c,data1_nn_y_c,xy_ticks,axs[0, 0],lims=data1_xy_lim,label_names = label_names,data_nn=True)
            g1 = plot_jointplot(data1_ssa_x_c,data1_ssa_y_c,xy_ticks,axs[0, 1],lims=data1_xy_lim,label_names = label_names)

            g2 = plot_jointplot(data2_nn_x_c,data2_nn_y_c,xy_ticks,axs[0, 2],lims=data2_xy_lim,label_names = label_names,data_nn=True)
            g3 = plot_jointplot(data2_ssa_x_c,data2_ssa_y_c,xy_ticks,axs[0, 3],lims=data2_xy_lim,label_names = label_names)

            g4 = plot_jointplot(data3_nn_x_c,data3_nn_y_c,xy_ticks,axs[1, 0],lims=data3_xy_lim,label_names = label_names,data_nn=True)
            g5 = plot_jointplot(data3_ssa_x_c,data3_ssa_y_c,xy_ticks,axs[1,1],lims=data3_xy_lim,label_names = label_names)

            g6 = plot_jointplot(data4_nn_x_c,data4_nn_y_c,xy_ticks,axs[1, 2],lims=data4_xy_lim,label_names = label_names,data_nn=True)
            g7 = plot_jointplot(data4_ssa_x_c,data4_ssa_y_c,xy_ticks,axs[1,3],lims=data4_xy_lim,label_names = label_names)

            mg0 = SeabornFig2Grid(g1, fig, gs[0])
            mg1 = SeabornFig2Grid(g3, fig, gs[1])
            mg2 = SeabornFig2Grid(g5, fig, gs[2])
            mg3 = SeabornFig2Grid(g7, fig, gs[3])
            mg4 = SeabornFig2Grid(g0, fig, gs[4])
            mg5 = SeabornFig2Grid(g2, fig, gs[5])
            mg6 = SeabornFig2Grid(g4, fig, gs[6])
            mg7 = SeabornFig2Grid(g6, fig, gs[7])
            gs.tight_layout(fig)

            # Display the graph
            plt.savefig(figure_path + "joint_plot_{}_{}.jpg".format(model_name,file_index))
            plt.savefig(figure_path + "joint_plot_{}_{}.pdf".format(model_name,file_index),bbox_inches="tight",dpi=600)
            plt.savefig(figure_path + "joint_plot_{}_{}.eps".format(model_name,file_index),bbox_inches="tight")
        except Exception as e:
            print("error",e)
if __name__ == '__main__':

    # plot_autoFeedbackLoop_result()
    plot_on_off_nm()
