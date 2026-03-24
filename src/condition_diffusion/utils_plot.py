import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams


# Set font file path
font_path = '/GPUFS/sysu_jjzhang_1/hzw/data/arial.ttf'

# Create font object
arial_font = fm.FontProperties(fname=font_path)

# Apply font as global default font
rcParams['font.family'] = arial_font.get_name()
rcParams['font.family'] = "Helvetica"
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
rcParams['text.usetex'] = False



def generate_canvas(rows_cols,width_size=1,height_size = 0.7):
    width_inch = 183 * 0.0393701*width_size
    height_inch = width_inch *height_size 
    fig, axes = plt.subplots(nrows=rows_cols[0], ncols=rows_cols[1], figsize=(width_inch, height_inch))
    return fig,axes


def line_scatter_plot(data1,data2,ax,line_label = "",labels=["Time","Average count"], color='#4d9ebc'):
    t_value = list(range(len(data1)))
    ax.plot(t_value,data1,  linewidth=0.5, label=line_label,color=color)
    ax.plot(t_value, data2, marker='o', linestyle='None', markersize=1.5,color=color)
    legend = ax.legend(loc='upper right', fontsize='4.8', title_fontsize='6')
    set_font_label(ax,x_label=labels[0],y_label=labels[1])

def line_with_shadow_plot(data,ax,labels=['Time Point','KL distance'],line_label="Mean", color='#4d9ebc',fill_flag=False):
    means = np.mean(data, axis=1)
    mins = np.min(data, axis=1)
    maxs = np.max(data, axis=1)
    ax.plot(means, label=line_label, linewidth=0.5,color=color)
    ax.plot(means, marker='o', linestyle='None', markersize=1.5,color=color,markeredgewidth=0.5)
    if fill_flag:
        ax.fill_between(range(data.shape[0]), mins, maxs, color='#cfebf7', label='Min-Max Range')
    legend = ax.legend(loc='upper right', fontsize='4.8', title_fontsize='6')
    set_font_label(ax,x_label=labels[0],y_label=labels[1])

# def set_font_label(ax,x_label,y_label,font_size=7,font_name='arial'):
# def set_font_label(ax,x_label,y_label,font_size=7,font_name='Helvetica'):
#     ax.set_xlabel(x_label, fontsize=font_size, fontname=font_name,labelpad=1)
#     ax.set_ylabel(y_label, fontsize=font_size, fontname=font_name,labelpad=1)
#     ax.tick_params(axis='both', which='major', labelsize=6.5, length=1, pad=1)
#     ax.tick_params(axis='both', which='minor', labelsize=6.5, length=1,pad=1)
#     return ax

# def set_font_label(ax,x_label,y_label,font_size=7):
#     ax.set_xlabel(x_label, fontsize=font_size, labelpad=1)
#     ax.set_ylabel(y_label, fontsize=font_size, labelpad=1)
#     ax.tick_params(axis='both', which='major', labelsize=6.5, length=1, pad=1)
#     ax.tick_params(axis='both', which='minor', labelsize=6.5, length=1,pad=1)
#     return ax
def set_font_label(ax,x_label,y_label,font_size=6,font_name='Arial',labelsize=5,tick_width=0.4): # 6.5
    ax.set_xlabel(x_label, fontsize=font_size, fontname=font_name,labelpad=1)
    ax.set_ylabel(y_label, fontsize=font_size, fontname=font_name,labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, length=1.6, pad=1,width=tick_width)
    ax.tick_params(axis='both', which='minor', labelsize=labelsize, length=1.6,pad=1,width=tick_width)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    return ax

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


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def plot_jointplot(data1,data2,xy_ticks,ax,lims,label_names,fontsize = 6,cmap_name = "Blues",fill_flag=True,color="#d37166",data_nn=False):
    if data_nn:
        label_names = [el + " (ABMGPT)" for el in label_names]
    else:
        label_names = [el + " (SSA)" for el in label_names]
    cmap_name = "Blues"
    # g0 = sns.jointplot(x=data1, y=data2, kind="hist",  fill=fill_flag,cmap=cmap_name, color=color,ax=ax,cbar=True)
    g0 = sns.jointplot(x=data1, y=data2, kind="kde",  fill=fill_flag,cmap=cmap_name, color=color,ax=ax,cbar=True)
    if len(lims)>0:
        g0.ax_joint.set_xlim(0, lims[0]+2)
        g0.ax_joint.set_ylim(0, lims[1]+2)
    # Set axis labels and font size
    g0.ax_joint.set_xlabel(label_names[0], fontsize=fontsize,labelpad=1)
    g0.ax_joint.set_ylabel(label_names[1], fontsize=fontsize,labelpad=1)

    length = 1
    g0.ax_joint.tick_params(axis='both', which='major', labelsize=fontsize, length=length,pad=1)
    g0.ax_joint.tick_params(axis='both', which='minor', labelsize=fontsize, length=length,pad=1)
    g0.ax_marg_x.tick_params(axis='x', which='major', labelsize=6, length=length)  # Set tick length for top histogram
    g0.ax_marg_y.tick_params(axis='y', which='major', labelsize=6, length=length)  # Set tick length for right histogram
    if len(xy_ticks[0])>0:
        g0.ax_joint.set_xticks(xy_ticks[0])
        g0.ax_joint.set_yticks(xy_ticks[1])
    # plt.colorbar(g0.ax_joint.collections[0], ax=g0.ax_joint,location='right')
    return g0


def convert_counts_data(file_path1,file_path2,sub_col_index=[1,4],path_flag=True):
    data1_ssa = pd.read_csv(file_path1)
    data1_nn = pd.read_csv(file_path2)
    sub_col = [data1_ssa.columns[sub_col_index[0]],data1_ssa.columns[sub_col_index[1]]]

    data1_ssa = data1_ssa[sub_col][:10000]
    data1_nn = data1_nn[sub_col][:10000]

    data1 =  data1_nn.iloc[:,0].values 
    # noise1 = np.random.uniform(-0.5, 0.5, size=len(data1))
    # noise2 = np.random.uniform(-0.5, 0.5, size=len(data1))
    noise1 = np.random.uniform(0, 1, size=len(data1))
    noise2 = np.random.uniform(0, 1, size=len(data1))

    data1_nn_x_continuous = data1_nn.iloc[:,0].values + noise1
    data1_nn_y_continuous = data1_nn.iloc[:,1].values + noise2

    data1_ssa_x_continuous = data1_ssa.iloc[:,0].values + noise1
    data1_ssa_y_continuous = data1_ssa.iloc[:,1].values + noise2
    x_lim = math.ceil(max(data1_ssa_x_continuous))
    y_lim = math.ceil(max(data1_ssa_y_continuous))
    return data1_nn_x_continuous,data1_nn_y_continuous,data1_ssa_x_continuous,data1_ssa_y_continuous,x_lim,y_lim


def plot_density(ax,values,weights,x_position,x_label,label_value = 'True value'):
    kde = gaussian_kde(values, weights=weights)
    x = np.linspace(min(values)*0.8, max(values), 1000)
    y = kde(x)
    # Draw weighted KDE curve

    ax.plot(x, y)

    set_font_label(ax,x_label,'Posterior prob.')

    ax.vlines(x=x_position, ymin=min(y), ymax=max(y), color='r', linestyle='--', linewidth=2, label=label_value)
    # x0=min(x)
    # ax.vlines(x=[x0,x_position], ymin=min(y), ymax=max(y), color='r', linestyle='--', linewidth=2)

    # ax.axvline(x=x_position, color='red', linestyle='--', label=label_value)

    ax.legend(prop={'size': 5})
    ax.set_xlim([min(values)*0.8,max(values)])


def plot_density(ax,values,weights,x_position,x_label,label_value = 'True value',max_param_value=-1):
    kde = gaussian_kde(values, weights=weights)
    x = np.linspace(min(values)*0.8, max(values), 1000)
    y = kde(x)
    # Draw weighted KDE curve
    ax.plot(x, y,linewidth=1)
    ax.axvline(x=x_position, color='red', linestyle='--', label=label_value)
    ax.fill_between(x, y, color='blue', alpha=0.05)
    if max_param_value>=0:
        ax.axvline(x=max_param_value, color='green', linestyle='--', label=" max density param")
    set_font_label(ax,x_label,'Posterior prob.')
    ax.legend(prop={'size': 5})

def hist_plot(data,ax,bins=50,colors = ['#C9CACA', '#2186AB']):
    labels = ["Protein counts","Probability"]
    bins = int(np.max(data))+1
    temp = ax.hist(data, bins = bins, stacked=False, density=True, color=colors,label=["NN","SSA"])
    legend = ax.legend(loc='upper right', fontsize='4.8', title_fontsize='6')
    set_font_label(ax,x_label=labels[0],y_label=labels[1])

