import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
from utils import *
from tqdm import tqdm
from autoencoder import AutoEncoder
from vdvae import VDVAE
from stochastic_search import StochasticSearch
from torchvision import transforms
import math
import argparse
    
if __name__ == "__main__":
    
    # Create the parser and add arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--output', 
                        help="output directory for the generated samples",
                        type=str,
                        default="output/")
    
    parser.add_argument('--idx', 
                        help="list of indicies to be generated for each subject.", 
                        type=str,
                        #LAION heldout
                        # default="20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 205, 206, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 270, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 393, 394, 395, 396, 397, 398, 400, 401, 402, 403, 404, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 616, 617, 618, 619, 620, 621, 622, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 655, 656, 657, 659, 661, 662, 663, 664, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 694, 695, 696, 698, 699, 700, 701, 702, 703, 704, 705, 706, 708, 709, 710, 711, 712, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 782, 783, 784, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 838, 839, 840, 841, 842, 843, 844, 845, 847, 848, 849, 851, 852, 854, 855, 856, 857, 858, 859, 861, 862, 863, 864, 865, 866, 867, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 974, 976, 977, 978, 979, 980, 981")
                        #Full set of 982:
                        default=range(1000))
    parser.add_argument('-l',
                        '--log', 
                        help="boolean flag, if passed, will save all intermediate images for each iteration of the algorithm, as well as intermediate encoded brain scans. WARNING: This saves a lot of data, only enable if you have terabytes of disk space to throw at it.",
                        action='store_true')
    
    parser.add_argument('--mode', 
                        help="trial modality, either nsd_vision for main shared1000 data, vision, or imagery.",
                        type=str,
                        default="nsd_vision")
    
    parser.add_argument('--basemodel', 
                        help="base decoding model to load seeds from",
                        type=str,
                        default="mindeye")
    
    parser.add_argument('-s',
                        '--subjects', 
                        help="list of subjects to run the algorithm on, if not specified, will run on all subjects",
                        type=str,
                        default="1,2,5,7")
    
    parser.add_argument('-n',
                        '--num_samples', 
                        help="number of library images generated at every interation of the algorithm.",
                        type=int,
                        default=100)
    
    parser.add_argument('-i',
                        '--iterations', 
                        help="number of interations for the search algorithm.",
                        type=int,
                        default=10)
    
    parser.add_argument('-b',
                        '--branches', 
                        help="number of additional suboptimal paths to explore during the search.",
                        type=int,
                        default=4)
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0")
    

    # Parse and print the results
    args = parser.parse_args()
    if isinstance(args.idx, str):
        idx_list = [int(sub) for sub in args.idx.strip().split(",")]
    else:
        idx_list = args.idx
    subject_list = [int(sub) for sub in args.subjects.strip().split(",")]
    print(idx_list)
    ae = False
    mode = args.mode
    basemodel = args.basemodel
    
    for subject in subject_list:
        with torch.no_grad():
            subject_path = "{}subject{}/".format(args.output, subject)
            os.makedirs(subject_path, exist_ok=True)
            # Load data and targets
            if mode == "nsd_vision":
                _, _, x_test, _, _, target_images, trials = load_nsd(vector="images", subject=subject, loader=False, average=True, nest=False, include_heldout=True)

            else:
                x_test, target_images = load_nsd_mental_imagery(vector = "images", subject=subject, mode=mode, stimtype="all", average=True, nest=False)
                # basemodel_path = f"output/mental_imagery_paper/{mode}/mindeye/"
                # basemodel_z_path = "/home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/seeds/{}/".format(mode)
            SCS = StochasticSearch(modelParams=["gnet"],
                                    subject=subject,
                                    device=args.device,
                                    log=args.log,
                                    ae=ae,
                                    n_iter=args.iterations,
                                    n_samples=args.num_samples,
                                    n_branches=args.branches)

            os.makedirs(subject_path + "iteration_figures/", exist_ok=True)
                    
            for i, val in enumerate(tqdm(idx_list, desc="Reconstructing samples")):
                sample_path = "{}{}/".format(subject_path, val)
                best_dist_path = "{}best_distribution/".format(sample_path)
                os.makedirs(best_dist_path, exist_ok=True)
                os.makedirs(best_dist_path + "images/", exist_ok=True)
                os.makedirs(best_dist_path + "beta_primes/", exist_ok=True)
                # Generate target reconstructions
                # Generate and save output best initial guess reconstructions in a distribution format
                
                beta = x_test[val]
                
                if basemodel == "mindeye":
                    if mode != "nsd_vision":
                        init_img = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/seeds_40_sessions/{}/subject{}/{}/low_level.png".format(mode, subject, val))
                        decoded_clip_vision = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/seeds_40_sessions/{}subject{}/{}/clip_best.pt".format(mode, subject, val), map_location=args.device)
                        decoded_clip_text = None
                        basemodel_recon = Image.open("{}/subject{}/{}/0.png".format(basemodel_path, subject, val))
                    else:
                        init_img = Image.open("/home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/seeds_40_sessions/{}/subject{}/{}/low_level.png".format(mode, subject, val))
                        decoded_clip_vision = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/seeds_40_sessions/{}subject{}/{}/clip_best.pt".format(mode, subject, val), map_location=args.device)
                        decoded_clip_text = None
                        basemodel_recon = Image.open("{}/subject{}/{}/0.png".format(basemodel_path, subject, val))
                elif basemodel == "braindiffuser":
                    if mode != "nsd_vision":
                        init_img = Image.open(f"/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/mental_imagery_paper/{mode}/braindiffuser/subject{subject}/{val}/low_level.png")
                        basemodel_recon = Image.open(f"/home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/mental_imagery_paper/{mode}/braindiffuser/subject{subject}/{val}/0.png")
                        decoded_clip_vision = torch.from_numpy(np.load(f"/home/naxos2-raid25/kneel027/home/kneel027/brain-diffuser_mi/data/predicted_features_mi/subj0{subject}/nsd_clipvision_predtest_nsdgeneral_{mode}.npy")[val])
                        decoded_clip_text = torch.from_numpy(np.load(f"/home/naxos2-raid25/kneel027/home/kneel027/brain-diffuser_mi/data/predicted_features_mi/subj0{subject}/nsd_cliptext_predtest_nsdgeneral_{mode}.npy")[val])
                    else:
                        raise ValueError("Invalid modality for braindiffuser")
                else:
                    raise ValueError("Invalid basemodel")
                # Perform search
                scs_reconstruction, best_distribution_params, image_list, score_list, var_list, basemodel_recon_bc = SCS.search(sample_path=sample_path, beta=beta, c_i=decoded_clip_vision, c_t= decoded_clip_text, init_img=init_img, basemodel_recon=basemodel_recon)
                
                # Save final distribution parameters for search
                f = open("{}strength.txt".format(best_dist_path), "w")
                f.write("{}\n".format(best_distribution_params["strength"]))
                f.close()
                torch.save(decoded_clip_vision, "{}clip_vision.pt".format(sample_path))
                if decoded_clip_text is not None:
                    torch.save(decoded_clip_text, "{}clip_text.pt".format(sample_path))
                if best_distribution_params["z_img"] is not None:
                    best_distribution_params["z_img"].save("{}z_img.png".format(best_dist_path))
                np.save("{}score_list.npy".format(sample_path), np.array(score_list))
                np.save("{}var_list.npy".format(sample_path), np.array(var_list))
                for j, (im, beta_prime) in enumerate(zip(best_distribution_params["images"], best_distribution_params["beta_primes"])):
                    im.save("{}{}.png".format(best_dist_path + "images/", j))
                    torch.save(beta_prime, "{}{}.pt".format(best_dist_path + "beta_primes/", j))
                    
                ground_truth = Image.fromarray(target_images[val].numpy().reshape((425, 425, 3)).astype(np.uint8)).resize((768, 768), resample=Image.Resampling.LANCZOS)
                # THIS NEEDS TO BE FIXED TO WORK WITH BOTH METHODS
                images = [ground_truth, scs_reconstruction, init_img, basemodel_recon]
                rows = int(math.ceil(len(image_list)/2 + len(images)/2))
                columns = 2
                captions = ["ground_truth", "search_reconstruction", "low_level", f"Base Model: {basemodel_recon_bc}"]
                for j in range(len(image_list)):
                    images.append(image_list[j])
                    captions.append("BC: {:.3f}, Var: {:.5f}".format(float(score_list[j]), var_list[j]))
                print(len(captions), len(images), rows, columns, captions)
                figure = tileImages("index: {}".format(val), images, captions, rows, columns)
                
                # Save relevant output images for evaluation
                figure.save("{}/iteration_diagram.png".format(sample_path))
                figure.save("{}iteration_figures/{}.png".format(subject_path, val))
                count = 0
                for j in range(len(images)):
                    if("BC" in captions[j]):
                        images[j].save("{}/iter_{}.png".format(sample_path, count))
                        count +=1
                    elif("Base Model:" in captions[j]):
                        images[j].save("{}/{}.png".format(sample_path, "basemodel_recon"))
                    else:
                        images[j].save("{}/{}.png".format(sample_path, captions[j]))
                        
                extra_beta_primes = SCS.EncModels[0].predict([ground_truth])
                torch.save(extra_beta_primes[0], "{}/ground_truth_beta_prime.pt".format(sample_path))