# Only GPU's in use
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
from torchmetrics.functional import pearson_corrcoef
from torch.autograd import Variable
import numpy as np
from nsd_access import NSDAccess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
sys.path.append('src')
from utils import *
import copy
from tqdm import tqdm
import nibabel as nib

# _, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=1, loader=False, average=False)
# _, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=2, loader=False, average=False)
# _, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=5, loader=False, average=False)
# _, _, _, _, _, _, _ = load_nsd(vector="c_img_uc", subject=7, loader=False, average=False)
# vdvae_73k = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/preprocessed_data/z_vdvae_73k.pt")
# torch.save(torch.mean(vdvae_73k, dim=0), "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/vdvae/train_mean.pt")
# torch.save(torch.std(vdvae_73k, dim=0), "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/vdvae/train_std.pt")
# vdvae_27k = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/preprocessed_data/subject1/z_vdvae.pt")
# for i in range(vdvae_73k.shape[0]):
#     print(i, torch.sum(torch.count_nonzero(vdvae_73k[i])), torch.sum(torch.count_nonzero(vdvae_27k[i])))
# process_raw_tensors(vector="z_vdvae")
# process_masks(subject=1, big=True)
# subjects = [7]
# for subject in subjects:
#     # create_whole_region_unnormalized(subject=subject, big=True)
#     # create_whole_region_normalized(subject=subject, big=True)
#     # process_data(subject=subject, vector="c_img_uc")
#     # process_data(subject=subject, vector="images")
#     # process_data(subject=subject, vector="z_vdvae")
#     process_masks(subject=subject, big=False)
#     process_masks(subject=subject, big=True)

# mask_path = "masks/subject1/nsdgeneral_big.nii.gz"
# # mask_path = "/export/raid1/home/styvesg/data/nsd/masks/subj01/func1pt8mm/brainmask_inflated_1.0.nii"
# # mask_path_v = "/home/naxos2-raid25/kneel027/home/kneel027/home/styvesg/data/nsd/masks/subj01/func1pt8mm/roi/prf-visualrois.nii.gz"
# nsd_general = nib.load(mask_path).get_fdata()
# nsd_general = np.nan_to_num(nsd_general)#.astype(bool)
# nsd_general = np.where(nsd_general==1.0, True, False)
# # nsd_general = np.where(nsd_general==1.0, True, False)
# print("NSD_GENERAL S1: ", np.unique(nsd_general, return_counts=True))

# visual_rois = nib.load(mask_path_v).get_fdata()
# V1L = np.where(visual_rois==1.0, True, False)
# V1R = np.where(visual_rois==2.0, True, False)
# V1 = torch.from_numpy(V1L[nsd_general] + V1R[nsd_general])
# print("V1 S1: ", np.unique(V1, return_counts=True))

# encoderWeights = torch.load("masks/subject{}/{}_encoder_prediction_weights.pt".format(1, "gnetEncoder_clipEncoder"))
# V1 = torch.load("masks/subject1/V1.pt")
# early_vis = torch.load("masks/subject1/early_vis.pt")
# higher_vis = torch.load("masks/subject1/higher_vis.pt")
# print(torch.mean(encoderWeights[0, V1], dim=0), torch.mean(encoderWeights[1, V1], dim=0))
# print(torch.mean(encoderWeights[0, early_vis], dim=0), torch.mean(encoderWeights[1, early_vis], dim=0))
# print(torch.mean(encoderWeights[0, higher_vis], dim=0), torch.mean(encoderWeights[1, higher_vis], dim=0))
# thresh = encoderWeights[0] > 0.5
# print(torch.sum(thresh))
nsda = NSDAccess('/home/naxos2-raid25/kneel027/home/surly/raid4/kendrick-data/nsd', '/home/naxos2-raid25/kneel027/home/kneel027/nsd_local')
subj1 = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
subj1 = subj1.sort_values(by='subject1_rep0')

# idx = convert_indices(idx=[2, 7, 8, 10, 22, 28, 44, 61, 77, 90, 104, 110, 114, 121, 122, 159, 169, 185, 209, 210, 215, 225, 233, 255, 265, 325, 342, 351, 394, 401, 412, 414, 427, 439, 442, 451, 466, 467, 479, 487, 488, 500, 503, 504, 517, 519, 523, 531, 547, 579, 607, 609, 612, 616, 689, 735, 740, 776, 777, 779, 838, 874, 882, 887, 891, 916, 922, 928, 946, 960, 964, 970])
# print(idx)
# subj_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
# sample_count = 0
# index_list = []
# for i in range(subj_test.shape[0]):
#     heldout = True
#     for j in range(3):
#         scanId = subj_test.iloc[i]['subject{}_rep{}'.format(1, j)]
#         if scanId < 27750:
#             heldout = False
#     if heldout == False:
#         nsdId = subj1.iloc[i]['nsdId']
#         img = nsda.read_images([nsdId], show=True)
#         Image.fromarray(img[0]).save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/shared1000_images_nsdId/" + str(sample_count) + ".png")
#         index_list.append(sample_count)
#         sample_count += 1
#     else:
#         index_list.append(-1)
# for i, index in enumerate(idx):
#     print(i)
#     nsdId = subj1.iloc[i]['nsdId']

#     img = nsda.read_images([nsdId], show=True)
#     Image.fromarray(img[0]).save("/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/shared1000_images_nsdId/" + str(i) + ".png")
<<<<<<< HEAD
# gt_images = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/shared1000_images_nsdId/"
# rec_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/Brain Diffuser raw/"
# new_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/Brain Diffuser/"
# converted_indicies = convert_indices(idx=[i for i in range(1000)], reverse=True)
# converted_indicies = remove_heldout_indices(converted_indicies, scanId_sorted=False)
# for i in tqdm(range(982)):
#     index = converted_indicies[i]
#     tqdm.write(str(index))
#     os.makedirs(new_folder + str(index) + "/", exist_ok=True)
#     gt = Image.open(gt_images + "{}.png".format(str(index)))
#     gt.save(new_folder + str(index) + "/Ground Truth.png")
    
#     image = Image.open(rec_folder + "{}.png".format(str(i)))
#     image.save(new_folder + str(index) + "/0.png")
        
=======
gt_images = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/shared1000_images_nsdId/"
rec_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject7/Tagaki raw/"
new_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject7/Tagaki/"
os.makedirs(new_folder, exist_ok=True)
converted_indicies = convert_indices(idx=[i for i in range(1000)], reverse=True)
converted_indicies = remove_heldout_indices(converted_indicies, scanId_sorted=False)
for i in tqdm(range(982)):
    index = converted_indicies[i]
    tqdm.write(str(index))
    os.makedirs(new_folder + str(index) + "/", exist_ok=True)
    gt = Image.open(gt_images + "{}.png".format(str(index)))
    gt.save(new_folder + str(index) + "/Ground Truth.png")
    for j in range(5):
        image = Image.open(rec_folder + "{:05d}_00{}_zc.png".format(i, j))
        image.save(new_folder + str(index) + "/{}.png".format(j))
>>>>>>> 8fe4a577cddf6ef3bd5a329212bd12861edb8d9c
        
# SAVE CORTICAL CONVOLUTION IMAGES   
# subj_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
# # subj_test = subj_test.sort_values(by='subject1_rep0')
# sample_count = 0
# index_list = []
# for i in range(subj_test.shape[0]):
#     heldout = True
#     for j in range(3):
#         scanId = subj_test.iloc[i]['subject{}_rep{}'.format(1, j)]
#         if scanId < 27750:
#             heldout = False
#     if heldout == False:
#         index_list.append(sample_count)
#         sample_count += 1
#     else:
#         index_list.append(-1)
# gt_images = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/logs/shared1000_images_nsdId/"
# rec_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject2/Cortical Convolutions/"
# image_block = np.load(rec_folder + "meshpool_adamw_lr1e-04_dc1e-01_dp5e-01_fd32_ind32_layer3_rw1e-07_ifw0e+00_ofw1e+00_kldw1e-08_ae10_kldse0_vsf1e+00_fixb2f_mupred_imgs.npy")
# print(image_block.shape)
# for index in tqdm(range(1000)):
#     if index_list[index] != -1:
#         i = index_list[index]
#         os.makedirs(rec_folder + str(i) + "/", exist_ok=True)
#         gt = Image.open(gt_images +"{}.png".format(str(i)))
#         gt.save(rec_folder + str(i) + "/Ground Truth.png")
#         # for j in range(5):
#         # print(image_block[index])
#         image = Image.fromarray((image_block[index] * 255).astype(np.uint8))
#         image.save(rec_folder + str(i) + "/0.png")

# rec_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/Mind Diffuser raw/"
# out_folder = "/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/reconstructions/subject1/Mind Diffuser/"
# for i in tqdm(range(982)):
#     os.makedirs(out_folder + str(i) + "/", exist_ok=True)
#     gt = Image.open(gt_images +"{}.png".format(str(i)))
#     gt.save(out_folder + str(i) + "/Ground Truth.png")
#     # for j in range(5):
#     # print(image_block[index])
<<<<<<< HEAD
#     image = Image.open(rec_folder + "{}.png".format(str(i)))
#     image.save(out_folder + str(i) + "/" + str(i) + ".png")

first_list = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 
                64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 108, 
                109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
                147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 
                182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 205, 206, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 
                221, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 
                258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 270, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 
                297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 
                333, 334, 335, 336, 337, 338, 339, 341, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 364, 365, 366, 367, 368, 369, 370, 
                371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 393, 394, 395, 396, 397, 398, 400, 401, 402, 403, 404, 406, 407, 408, 
                409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 
                447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 481, 482, 
                483, 484, 485, 486, 487, 488, 489, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 
                521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549, 551, 552, 553, 554, 555, 556, 557, 
                558, 559, 560, 561, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 
                594, 595, 596, 597, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 616, 617, 618, 619, 620, 621, 622, 624, 625, 626, 627, 628, 629, 630, 631, 632, 
                633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 655, 656, 657, 659, 661, 662, 663, 664, 666, 667, 668, 669, 670, 671, 
                672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 694, 695, 696, 698, 699, 700, 701, 702, 703, 704, 705, 706, 708, 709, 
                710, 711, 712, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 
                747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 782, 783, 
                784, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819,
                820, 821, 822, 823, 824, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 838, 839, 840, 841, 842, 843, 844, 845, 847, 848, 849, 851, 852, 854, 855, 856, 857, 858, 859,
                861, 862, 863, 864, 865, 866, 867, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 892, 893, 894, 895, 896, 897, 
                898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 
                934, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 
                971, 974, 976, 977, 978, 979, 980, 981]

second_list = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 205, 206, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 270, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 393, 394, 395, 396, 397, 398, 400, 401, 402, 403, 404, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 616, 617, 618, 619, 620, 621, 622, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 655, 656, 657, 659, 661, 662, 663, 664, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 694, 695, 696, 698, 699, 700, 701, 702, 703, 704, 705, 706, 708, 709, 710, 711, 712, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 782, 783, 784, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 838, 839, 840, 841, 842, 843, 844, 845, 847, 848, 849, 851, 852, 854, 855, 856, 857, 858, 859, 861, 862, 863, 864, 865, 866, 867, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 974, 976, 977, 978, 979, 980, 981]


difference_result = [item for item in first_list if item not in second_list]
print(difference_result)
=======
#     image = Image.open(rec_folder + "{}.png".format(str(i))).convert('RGB')
#     image.save(out_folder + str(i) + "/0.png")
>>>>>>> 8fe4a577cddf6ef3bd5a329212bd12861edb8d9c
