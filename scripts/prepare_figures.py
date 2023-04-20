import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append('src')
from utils import *
import seaborn as sns
from matplotlib.lines import Line2D
import cv2


# folder_list = []
# for it in os.scandir("/logs"):
#     if it.is_dir():
#         folder_list.append(it.name)
# count = 0

# for i in range(25):
#     ground_truth   = cv2.imread('/export/raid1/home/ojeda040/Second-Sight/reconstructions/SCS VD 10:250:5 HS nsd_general AE/' + str(i) + '/Ground Truth.png')
#     reconstruction = cv2.imread('/export/raid1/home/ojeda040/Second-Sight/reconstructions/SCS VD 10:250:5 HS nsd_general AE/' + str(i) + '/Search Reconstruction.png')
    
#     ground_truth = cv2.resize(ground_truth, (425, 425))
#     reconstruction = cv2.resize(reconstruction, (425, 425))

#     ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
#     reconstruction = cv2.cvtColor(reconstruction, cv2.COLOR_BGR2GRAY)

#     count += ssim_scs(ground_truth, reconstruction)
    
# print(count / 25)


# brain_correlation_V1            = np.empty((25, 10))
# brain_correlation_V2            = np.empty((25, 10))
# brain_correlation_V3            = np.empty((25, 10))
# brain_correlation_V4            = np.empty((25, 10))
# brain_correlation_early_visual  = np.empty((25, 10))
# brain_correlation_higher_visual = np.empty((25, 10))
# brain_correlation_unmasked      = np.empty((25, 10))


# # Encoding vectors for 2819140 images
# for i in tqdm(range(25)):
    
#     brain_correlation_V1[i]            = np.load("logs/SCS 10:250:5 HS nsd_general AE/" + str(i) + "_score_list_V1.npy")
#     brain_correlation_V2[i]            = np.load("logs/SCS 10:250:5 HS nsd_general AE/" + str(i) + "_score_list_V2.npy")
#     brain_correlation_V3[i]            = np.load("logs/SCS 10:250:5 HS nsd_general AE/" + str(i) + "_score_list_V3.npy")
#     brain_correlation_V4[i]            = np.load("logs/SCS 10:250:5 HS nsd_general AE/" + str(i) + "_score_list_V4.npy")
#     brain_correlation_early_visual[i]  = np.load("logs/SCS 10:250:5 HS nsd_general AE/" + str(i) + "_score_list_early_visual.npy")
#     brain_correlation_higher_visual[i] = np.load("logs/SCS 10:250:5 HS nsd_general AE/" + str(i) + "_score_list_higher_visual.npy")
#     brain_correlation_unmasked[i]      = np.load("logs/SCS 10:250:5 HS nsd_general AE/" + str(i) + "_score_list.npy")


# brain_correlation_V1            = np.average(brain_correlation_V1, axis=0)
# brain_correlation_V2            = np.average(brain_correlation_V2, axis=0)
# brain_correlation_V3            = np.average(brain_correlation_V3, axis=0)
# brain_correlation_V4            = np.average(brain_correlation_V4, axis=0)
# brain_correlation_unmasked      = np.average(brain_correlation_unmasked, axis=0)
# brain_correlation_early_visual  = np.average(brain_correlation_early_visual, axis=0)
# brain_correlation_higher_visual = np.average(brain_correlation_higher_visual, axis=0)


# strength  = np.arange(1.0, 0.5, -0.05)
# iteration = np.arange(0.0, 10.0)

# # print(var.dtype, brain_correlation.dtype, strength.dtype, iteration.dtype)

# df = pd.DataFrame(np.stack((iteration, brain_correlation_V1, brain_correlation_V2, 
#                             brain_correlation_V3, brain_correlation_V4, brain_correlation_early_visual,
#                             brain_correlation_higher_visual, brain_correlation_unmasked), axis=1), 
#                             columns = ['iteration', 'brain_correlation_V1', 'brain_correlation_V2', 'brain_correlation_V3', 
#                                        'brain_correlation_V4', 'brain_correlation_early_visual', 'brain_correlation_higher_visual', 
#                                        'brain_correlation_visual_cortex'])

# # df = pd.DataFrame(np.stack((iteration, brain_correlation_V1, brain_correlation_V2, 
# #                             brain_correlation_V3, brain_correlation_V4, brain_correlation_early_visual,
# #                             brain_correlation_unmasked), axis=1), 
# #                             columns = ['iteration', 'brain_correlation_V1', 'brain_correlation_V2', 'brain_correlation_V3', 
# #                                        'brain_correlation_V4', 'brain_correlation_early_visual', 
# #                                        'brain_correlation_visual_cortex'])

# g = sns.lineplot(data=df['brain_correlation_V1'], color = '#FF5733')
# sns.lineplot(data=df['brain_correlation_V2'], color = '#EA891B', ax=g.axes.twinx())
# sns.lineplot(data=df['brain_correlation_V3'], color = '#FDFD03', ax=g.axes.twinx())
# sns.lineplot(data=df['brain_correlation_V4'], color = '#3B8809', ax=g.axes.twinx())
# sns.lineplot(data=df['brain_correlation_early_visual'], color = '#00FCCB', ax=g.axes.twinx())
# sns.lineplot(data=df['brain_correlation_visual_cortex'], color = '#001FFC', ax=g.axes.twinx())
# sns.lineplot(data=df['brain_correlation_higher_visual'], color = '#A800FC', ax=g.axes.twinx())

# # sns.lineplot(data=df['var'], color="b", ax=g.axes.twinx())
# # sns.lineplot(data=df['strength'], color="r", ax=g.axes.twinx())
# g.axes.spines.right.set_position(("axes", 1.2))
# g.legend(handles=[Line2D([], [], marker='_', color = '#FF5733', label='brain_correlation_V1'), Line2D([], [], marker='_', color = '#EA891B', label='brain_correlation_v2'), 
#                   Line2D([], [], marker='_', color = '#FDFD03', label='brain_correlation_V3'), Line2D([], [], marker='_', color = '#3B8809', label='brain_correlation_V4'), 
#                   Line2D([], [], marker='_', color = '#00FCCB', label='brain_correlation_early_visual'), Line2D([], [], marker='_', color = '#A800FC', label='brain_correlation_higher_visual'),
#                   Line2D([], [], marker='_', color = '#001FFC', label='brain_correlation_visual_cortex')])


# # ax = plt.figure(figsize=(10,6), tight_layout=True)
# #plotting
# # ax = df.plot(x='iteration', y='brain_correlation', legend=False, color="b")
# # # sns.lineplot(data=df['brain_correlation'], linewidth=2, color="b")
# # ax2 = ax.twinx()
# # df.plot(x='iteration', y='var', ax=ax2, legend=False, color="r")
# # ax3 = ax.twinx()
# # df.plot(x='iteration', y='strength', ax=ax3, legend=False, color="g")
# # ax.figure.legend()
# # sns.lineplot(data=df['var'], linewidth=2, color="g")
# #customization
# g.set(xlabel='Search Iteration', ylabel='Brain Pearson Correlation', title='Encoded Brain Pearson Correlation in Early Visual Cortex ', xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# # ax.legend(title='Players', title_fontsize = 13)
# # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # plt.xlabel('Search Iteration')
# # plt.ylabel('Brain Correlation')
# # plt.title('Encoded Brain Pearson Correlation in Early Visual Cortex ')
# # plt.legend(title='Players', title_fontsize = 13, labels=['L. Messi', 'Cristiano Ronaldo', 'K. De Bruyne', 'V. van Dijk', 'K. Mbappé'])
# plt.savefig("charts/brain_correlation_plot_twin.png")

Library_L2 = np.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/library_decoder_scores/c_img_uc_gnetEncoder_L2.npy")
Library_PeC = np.load("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/library_decoder_scores/c_img_uc_gnetEncoder_PeC.npy")

# print(Library_L2)
# print(Library_PeC)
print(np.argmin(Library_L2[1:1000]))
print(np.argmax(Library_PeC[1:1000]))

plt.plot(Library_PeC[30:1000])
plt.savefig("charts/Library_decoder_gnet_topn_1000_PeC.png")