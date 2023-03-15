import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as plt
from PIL import Image
sys.path.append('src')
from utils import *
import seaborn as sns
from matplotlib.lines import Line2D

# _, _, x_param, x_test, _, _, _, targets_c_i, param_trials, test_trials = load_nsd(vector="c_img_0", loader=False, average=True)

# folder_list = []
# for it in os.scandir("/logs"):
#     if it.is_dir():
#         folder_list.append(it.name)

var = np.empty((78, 10))
brain_correlation = np.empty((78, 10))
# Encoding vectors for 2819140 images
for i in tqdm(range(78)):
    var[i] = np.load("logs/SCS 10:250:5 HS V1234567 AE/" + str(i) + "_var_list.npy")
    brain_correlation[i] = np.load("logs/SCS 10:250:5 HS V1234567 AE/" + str(i) + "_score_list.npy")

var = np.average(var, axis=0)
brain_correlation = np.average(brain_correlation, axis=0)
strength = np.arange(1.0, 0.5, -0.05)
iteration = np.arange(0.0, 10.0)
# print(var.dtype, brain_correlation.dtype, strength.dtype, iteration.dtype)

df = pd.DataFrame(np.stack((iteration, brain_correlation, var, strength), axis=1), columns = ['iteration', 'brain_correlation','var','strength'])

g = sns.lineplot(data=df['brain_correlation'], color="g")
sns.lineplot(data=df['var'], color="b", ax=g.axes.twinx())
sns.lineplot(data=df['strength'], color="r", ax=g.axes.twinx())
g.axes.spines.right.set_position(("axes", 1.2))
g.legend(handles=[Line2D([], [], marker='_', color="g", label='brain_correlation'), Line2D([], [], marker='_', color="b", label='var'), Line2D([], [], marker='_', color="r", label='strength')])


# ax = plt.figure(figsize=(10,6), tight_layout=True)
#plotting
# ax = df.plot(x='iteration', y='brain_correlation', legend=False, color="b")
# # sns.lineplot(data=df['brain_correlation'], linewidth=2, color="b")
# ax2 = ax.twinx()
# df.plot(x='iteration', y='var', ax=ax2, legend=False, color="r")
# ax3 = ax.twinx()
# df.plot(x='iteration', y='strength', ax=ax3, legend=False, color="g")
# ax.figure.legend()
# sns.lineplot(data=df['var'], linewidth=2, color="g")
#customization
g.set(xlabel='Search Iteration', ylabel='Brain Pearson Correlation', title='Encoded Brain Pearson Correlation in Early Visual Cortex ', xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# ax.legend(title='Players', title_fontsize = 13)
# plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# plt.xlabel('Search Iteration')
# plt.ylabel('Brain Correlation')
# plt.title('Encoded Brain Pearson Correlation in Early Visual Cortex ')
# plt.legend(title='Players', title_fontsize = 13, labels=['L. Messi', 'Cristiano Ronaldo', 'K. De Bruyne', 'V. van Dijk', 'K. Mbappé'])
plt.savefig("charts/brain_correlation_plot2.png")