import os, sys
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as plt
from PIL import Image
sys.path.append('src')
from utils import *

def collect_subject_images(idx):
    base_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject"
    base_path2 = "/Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter/"
    output_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/comparisons/collected_subject_samples_appendix/"
    for index in tqdm(idx):
        image = Image.open("{}1{}{}/Ground Truth.png".format(base_path, base_path2, index))
        image.save("{}{}_gt.png".format(output_path, index))
        image = Image.open("{}1{}{}/Search Reconstruction.png".format(base_path, base_path2, index))
        image.save("{}{}_s1.png".format(output_path, index))
        image = Image.open("{}2{}{}/Search Reconstruction.png".format(base_path, base_path2, index))
        image.save("{}{}_s2.png".format(output_path, index))
        image = Image.open("{}5{}{}/Search Reconstruction.png".format(base_path, base_path2, index))
        image.save("{}{}_s5.png".format(output_path, index))
        image = Image.open("{}7{}{}/Search Reconstruction.png".format(base_path, base_path2, index))
        image.save("{}{}_s7.png".format(output_path, index))
        
def collect_paper_images(idx):
    minddiffuser_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Mind Diffuser/"
    braindiffuser_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Brain Diffuser/"
    tagaki_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Tagaki/"
    cortical_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Cortical Convolutions/"
    second_sight_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter/"
    output_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/comparisons/collected_paper_samples/"
    for index in tqdm(idx):
            image = Image.open("{}{}/Ground Truth.png".format(second_sight_path, index))
            image.save("{}{}_gt.png".format(output_path, index))
            image = Image.open("{}{}/Search Reconstruction.png".format(second_sight_path, index))
            image.save("{}{}_ss.png".format(output_path, index))
            image = Image.open("{}{}/Library Reconstruction.png".format(second_sight_path, index))
            image.save("{}{}_library.png".format(output_path, index))
            image = Image.open("{}{}/0.png".format(braindiffuser_path, index))
            image.save("{}{}_bd.png".format(output_path, index))
            image = Image.open("{}{}/0.png".format(tagaki_path, index))
            image.save("{}{}_tagaki.png".format(output_path, index))
            image = Image.open("{}{}/0.png".format(cortical_path, index))
            image.save("{}{}_cc.png".format(output_path, index))
            image = Image.open("{}{}/0.png".format(minddiffuser_path, index))
            image.save("{}{}_md.png".format(output_path, index))
            
        
def collect_iteration_images(idx):
    base_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter/"
    output_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/comparisons/collected_iteration_samples/"
    for index in tqdm(idx):
        image = Image.open("{}{}/Ground Truth.png".format(base_path, index))
        image.save("{}{}_gt.png".format(output_path, index))
        image = Image.open("{}{}/Search Reconstruction.png".format(base_path, index))
        image.save("{}{}_sr.png".format(output_path, index))
        image = Image.open("{}{}/Decoded CLIP Only.png".format(base_path, index))
        image.save("{}{}_dc.png".format(output_path, index))
        image = Image.open("{}{}/Decoded VDVAE.png".format(base_path, index))
        image.save("{}{}_dv.png".format(output_path, index))
        image = Image.open("{}{}/Decoded CLIP+VDVAE.png".format(base_path, index))
        image.save("{}{}_dcv.png".format(output_path, index))
        image = Image.open("{}{}/iter_0.png".format(base_path, index))
        image.save("{}{}_i0.png".format(output_path, index))
        image = Image.open("{}{}/iter_1.png".format(base_path, index))
        image.save("{}{}_i1.png".format(output_path, index))
        image = Image.open("{}{}/iter_2.png".format(base_path, index))
        image.save("{}{}_i2.png".format(output_path, index))
        image = Image.open("{}{}/iter_3.png".format(base_path, index))
        image.save("{}{}_i3.png".format(output_path, index))
        image = Image.open("{}{}/iter_4.png".format(base_path, index))
        image.save("{}{}_i4.png".format(output_path, index))
        image = Image.open("{}{}/iter_5.png".format(base_path, index))
        image.save("{}{}_i5.png".format(output_path, index))

def make_comparison_figures():
    minddiffuser_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Mind Diffuser/"
    braindiffuser_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Brain Diffuser/"
    tagaki_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Tagaki/"
    cortical_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Cortical Convolutions/"
    second_sight_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter/"
    output_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject1/comparisons/paper comparisons/"
    idx = sorted([int(i) for i in os.listdir(minddiffuser_path)])
    heldout = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 49, 70, 76, 78, 105, 112, 123, 146, 187, 197, 204, 208, 223, 255, 260, 269, 271, 276, 315, 340, 342, 363, 392, 399, 405, 416, 434, 446, 480, 490, 491, 503, 546, 550, 562, 598, 599, 615, 623, 654, 658, 660, 665, 693, 697, 707, 713, 724, 762, 781, 785, 825, 837, 846, 850, 853, 860, 868, 891, 914, 935, 946, 972, 973, 975]
    # print(len(idx), idx)
    for index in tqdm(idx):
        if index not in heldout and index > 239:
            images = []
            captions = []
            images.append(Image.open("{}{}/Ground Truth.png".format(second_sight_path, index)))
            captions.append("Ground Truth")
            images.append(Image.open("{}{}/Search Reconstruction.png".format(second_sight_path, index)))
            captions.append("Second Sight (Ours)")
            images.append(Image.open("{}{}/0.png".format(braindiffuser_path, index)))
            captions.append("Brain Diffuser")
            images.append(Image.open("{}{}/0.png".format(tagaki_path, index)))
            captions.append("Tagaki")
            images.append(Image.open("{}{}/0.png".format(cortical_path, index)))
            captions.append("Cortical")
            images.append(Image.open("{}{}/0.png".format(minddiffuser_path, index)))
            captions.append("Mind Diffuser")
        
            output = tileImages("Paper Comparisons", images, captions, 3, 2)
            output.save("{}{}.png".format(output_path, index))
            
def make_subject_figures():
    base_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/subject"
    base_path2 = "/Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter/"
    output_path = "/export/raid1/home/kneel027/Second-Sight/reconstructions/comparisons/subject comparisons/"
    idx = sorted([int(i.name[:-4]) for i in os.scandir("{}7{}".format(base_path, base_path2)) if i.is_file()])
    heldout = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 49, 70, 76, 78, 105, 112, 123, 146, 187, 197, 204, 208, 223, 255, 260, 269, 271, 276, 315, 340, 342, 363, 392, 399, 405, 416, 434, 446, 480, 490, 491, 503, 546, 550, 562, 598, 599, 615, 623, 654, 658, 660, 665, 693, 697, 707, 713, 724, 762, 781, 785, 825, 837, 846, 850, 853, 860, 868, 891, 914, 935, 946, 972, 973, 975]# print(len(idx), idx)
    for index in tqdm(idx):
        if index not in heldout:
            images = []
            captions = []
            images.append(Image.open("{}1{}{}/Ground Truth.png".format(base_path, base_path2, index)))
            captions.append("Ground Truth")
            images.append(Image.new('RGB', (768, 768), color='white'))
            captions.append("")
            images.append(Image.open("{}1{}{}/Search Reconstruction.png".format(base_path, base_path2, index)))
            captions.append("Subject 1")
            images.append(Image.open("{}2{}{}/Search Reconstruction.png".format(base_path, base_path2, index)))
            captions.append("Subject 2")
            images.append(Image.open("{}5{}{}/Search Reconstruction.png".format(base_path, base_path2, index)))
            captions.append("Subject 5")
            images.append(Image.open("{}7{}{}/Search Reconstruction.png".format(base_path, base_path2, index)))
            captions.append("Subject 7")
        
            output = tileImages("Subject Comparisons", images, captions, 3, 2)
            output.save("{}{}.png".format(output_path, index))

# collect_subject_images([24, 38, 107, 108, 110, 114, 150, 156, 170, 175, 176, 183, 185, 202, 211, 212, 213, 215, 216, 233, 234, 242, 244, 248, 257, 265, 267, 277, 284, 312, 319, 388, 427, 431, 437, 442, 451, 453, 488, 500, 501, 504, 506, 531, 547, 593, 609, 612, 616, 626, 730, 735, 765, 776, 779, 793, 830, 838, 848, 877, 882, 955, 964])
# collect_subject_images([90, 115, 122, 130, 159, 186, 488, 740])
# collect_iteration_images([226, 234, 404, 593, 626, 689, 830, 890])
# collect_paper_images([22, 169, 209, 325, 394, 412, 777, 874])
# make_comparison_figures()
make_subject_figures()