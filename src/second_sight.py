import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from utils import *
from tqdm import tqdm
from library_assembler import LibraryAssembler
from vdvae import VDVAE
from stochastic_search import StochasticSearch
from diffusers import StableUnCLIPImg2ImgPipeline
from torchmetrics import PearsonCorrCoef
from decoder_uc import Decoder_UC
import wandb
import math
import argparse
    
    #ALL SAMPLES EXCEPT FIRST 64
    # generateTestSamples(experiment_title="Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter", 
    #                     idx=[93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 106, 107, 108, 109, 111, 113, 115, 116, 117, 118, 119, 120, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 205, 206, 207, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 258, 259, 261, 262, 263, 264, 266, 267, 268, 270, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 343, 344, 345, 346, 347, 348, 349, 350, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 393, 395, 396, 397, 398, 400, 401, 402, 403, 404, 406, 407, 408, 409, 410, 411, 413, 414, 415, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 428, 429, 430, 431, 432, 433, 435, 436, 437, 438, 440, 441, 443, 444, 445, 447, 448, 449, 450, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 481, 482, 483, 484, 485, 486, 489, 492, 493, 494, 495, 496, 497, 498, 499, 501, 502, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 518, 519, 520, 521, 522, 524, 525, 526, 527, 528, 529, 530, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 548, 549, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 600, 601, 602, 603, 604, 605, 606, 607, 608, 610, 611, 613, 614, 617, 618, 619, 620, 621, 622, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 655, 656, 657, 659, 661, 662, 663, 664, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 690, 691, 692, 694, 695, 696, 698, 699, 700, 701, 702, 703, 704, 705, 706, 708, 709, 710, 711, 712, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 736, 737, 738, 739, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 778, 780, 782, 783, 784, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 839, 840, 841, 842, 843, 844, 845, 847, 848, 849, 851, 852, 854, 855, 856, 857, 858, 859, 861, 862, 863, 864, 865, 866, 867, 869, 870, 871, 872, 873, 875, 876, 877, 878, 879, 880, 881, 883, 884, 885, 886, 888, 889, 890, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 915, 917, 918, 919, 920, 921, 923, 924, 925, 926, 927, 929, 930, 931, 932, 933, 934, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 961, 962, 963, 965, 966, 967, 968, 969, 971, 974, 976, 977, 978, 979, 980, 981], 
    #                     modelParams=["gnetEncoder", "clipEncoder"],
    #                     subject=7,
    #                     num_samples=100,
    #                     n_iter=6,
    #                     n_branches=4,
    #                     ae=True,  
    #                     refine_clip = True,
    #                     refine_z = True)    
    
    #FIRST 64 ACROSS ALL SUBJECTS BATCH 1
    # generateTestSamples(experiment_title="Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter", 
    #                     idx=[20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92],
    #                     modelParams=["gnetEncoder", "clipEncoder"],
    #                     subject=2,
    #                     num_samples=100,
    #                     n_iter=6,
    #                     n_branches=4,
    #                     ae=True,  
    #                     refine_clip = True,
    #                     refine_z = True)  
    # generateTestSamples(experiment_title="Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter", 
    #                     idx=[20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 52, 53, 54],
    #                     modelParams=["gnetEncoder", "clipEncoder"],
    #                     subject=5,
    #                     num_samples=100,
    #                     n_iter=6,
    #                     n_branches=4,
    #                     ae=True,  
    #                     refine_clip = True,
    #                     refine_z = True) 
    #FIRST 64 ACROSS ALL SUBJECTS BATCH 2
    # generateTestSamples(experiment_title="Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter", 
    #                     idx=[55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92],
    #                     modelParams=["gnetEncoder", "clipEncoder"],
    #                     subject=5,
    #                     num_samples=100,
    #                     n_iter=6,
    #                     n_branches=4,
    #                     ae=True,  
    #                     refine_clip = True,
    #                     refine_z = True) 
    # generateTestSamples(experiment_title="Final Run: SCS UC LD 6:100:4 Dual Guided clip_iter", 
    #                     idx=[20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92],
    #                     modelParams=["gnetEncoder", "clipEncoder"],
    #                     subject=7,
    #                     num_samples=100,
    #                     n_iter=6,
    #                     n_branches=4,
    #                     ae=True,  
    #                     refine_clip = True,
    #                     refine_z = True)  
    

if __name__ == "__main__":
    
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('output', 
                        help="output directory for the generated samples",
                        type=str)
    
    parser.add_argument('--idx', 
                        help="list of indicies to be generated for each subject.", 
                        type=list,
                        default=[20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 106, 107, 108, 109, 111, 113, 115, 116, 117, 118, 119, 120, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 205, 206, 207, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 258, 259, 261, 262, 263, 264, 266, 267, 268, 270, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 343, 344, 345, 346, 347, 348, 349, 350, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 393, 395, 396, 397, 398, 400, 401, 402, 403, 404, 406, 407, 408, 409, 410, 411, 413, 414, 415, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 428, 429, 430, 431, 432, 433, 435, 436, 437, 438, 440, 441, 443, 444, 445, 447, 448, 449, 450, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 481, 482, 483, 484, 485, 486, 489, 492, 493, 494, 495, 496, 497, 498, 499, 501, 502, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 518, 519, 520, 521, 522, 524, 525, 526, 527, 528, 529, 530, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 548, 549, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 600, 601, 602, 603, 604, 605, 606, 607, 608, 610, 611, 613, 614, 617, 618, 619, 620, 621, 622, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 655, 656, 657, 659, 661, 662, 663, 664, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 690, 691, 692, 694, 695, 696, 698, 699, 700, 701, 702, 703, 704, 705, 706, 708, 709, 710, 711, 712, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 736, 737, 738, 739, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 778, 780, 782, 783, 784, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 839, 840, 841, 842, 843, 844, 845, 847, 848, 849, 851, 852, 854, 855, 856, 857, 858, 859, 861, 862, 863, 864, 865, 866, 867, 869, 870, 871, 872, 873, 875, 876, 877, 878, 879, 880, 881, 883, 884, 885, 886, 888, 889, 890, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 915, 917, 918, 919, 920, 921, 923, 924, 925, 926, 927, 929, 930, 931, 932, 933, 934, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 961, 962, 963, 965, 966, 967, 968, 969, 971, 974, 976, 977, 978, 979, 980, 981])
    
    parser.add_argument('-l',
                        '--log', 
                        help="boolean flag, if true, will save all intermediate images for each iteration of the algorithm, as well as intermediate encoded brain scans. WARNING: This saves a lot of data, only enable if you have terabytes of disk space to throw at it.",
                        type=bool,
                        default=False)
    
    parser.add_argument('-s',
                        '--subjects', 
                        help="list of subjects to run the algorithm on, if not specified, will run on all subjects",
                        type=list,
                        default=[1, 2, 5, 7])
    
    parser.add_argument('-n',
                        '--num_samples', 
                        help="number of library images generated at every interation of the algorithm.",
                        type=int,
                        default=100)
    
    parser.add_argument('-i',
                        '--iterations', 
                        help="number of interations for the search algorithm.",
                        type=int,
                        default=6)
    
    parser.add_argument('-b',
                        '--branches', 
                        help="number of additional suboptimal paths to explore during the search.",
                        type=int,
                        default=4)
    

    # Parse and print the results
    args = parser.parse_args()
    print(args.log)
    
    for subject in args.subjects:
        with torch.no_grad():
            os.makedirs("{}/subject{}/".format(args.output, subject), exist_ok=True)
            if(args.log):
                os.makedirs("logs/{}/subject{}/".format(args.output, subject), exist_ok=True)
                # Load data and targets
            _, _, x_test, _, _, targets_clips, trials = load_nsd(vector="c_i", subject=subject, loader=False, average=False, nest=True)
            _, _, x_test_avg, _, _, targets_vdvae, _ = load_nsd(vector="z_vdvae", subject=subject, loader=False, average=True, nest=False)
            x_test = x_test[args.idx]
            x_test_avg = x_test_avg[args.idx]
            
            targets_vdvae = normalize_vdvae(targets_vdvae[args.idx]).reshape((len(args.idx), 1, 91168))
            targets_clips = targets_clips[args.idx].reshape((len(args.idx), 1, 1024))
            
            LD = LibraryAssembler(configList=["gnetEncoder", "clipEncoder"],
                                subject=subject,
                                ae=True,
                                device="cuda")
            output_images  = LD.predict(x_test, vector="images", topn=1)
            output_clips = LD.predict(x_test, vector="c_i", topn=100).reshape((len(args.idx), 1, 1024))
            del LD
            LD_v = LibraryAssembler(configList=["gnetEncoder"],
                                subject=subject,
                                ae=True,
                                mask=torch.load("masks/subject{}/early_vis_big.pt".format(subject)),
                                device="cuda")
            output_vdvae = LD_v.predict(x_test, vector="z_vdvae", topn=25)
            output_vdvae = normalize_vdvae(output_vdvae).reshape((len(args.idx), 1, 91168))
            del LD_v
            # Initialize Models
            V = VDVAE()
            SCS = StochasticSearch(modelParams=["gnetEncoder", "clipEncoder"],
                                    subject=subject,
                                    device="cuda:0",
                                    n_iter=args.iterations,
                                    n_samples=args.num_samples,
                                    n_branches=args.branches)
                    
            PeC = PearsonCorrCoef(num_outputs=len(args.idx)).to("cpu")
            PeC1 = PearsonCorrCoef(num_outputs=1).to("cpu")
            
            for i, val in enumerate(tqdm(args.idx, desc="Reconstructing samples")):
                sample_path = "{}/subject{}/{}/".format(args.output, subject, val)
                os.makedirs(sample_path, exist_ok=True)
                if(args.log):
                    os.makedirs("{}/clip_distribution/".format(sample_path), exist_ok=True)
                    os.makedirs("{}/clip+vdvae_distribution/".format(sample_path), exist_ok=True)
                    os.makedirs("{}/vdvae_distribution/".format(sample_path), exist_ok=True)
                    torch.save(output_clips[i], sample_path + "decoded_clip.pt")
                # Generate target reconstructions
                target_v = V.reconstruct(targets_vdvae[i])
                target_c = SCS.R.reconstruct(image_embeds=targets_clips[i], negative_prompt="text, caption", strength=1.0)
                target_cv = SCS.R.reconstruct(image=target_v, image_embeds=targets_clips[i], negative_prompt="text, caption", strength=0.9)
                # Generate and save output reconstructions in a distribution
                for j in range(12):
                    output_v = V.reconstruct(output_vdvae[i])
                    output_c = SCS.R.reconstruct(image_embeds=output_clips[i], negative_prompt="text, caption", strength=1.0)
                    output_cv = SCS.R.reconstruct(image=output_v, image_embeds=output_clips[i], negative_prompt="text, caption", strength=0.9)
                    if(args.log):
                        output_c.save("{}/clip_distribution/{}.png".format(sample_path, j))
                        output_v.save("{}/vdvae_distribution/{}.png".format(sample_path, j))
                        output_cv.save("{}/clip+vdvae_distribution/{}.png".format(sample_path, j))
                
                scs_reconstruction, image_list, score_list, var_list = SCS.search(sample_path=sample_path, beta=x_test[i], c_i=output_clips[i], init_img=output_v)

                # Format output diagram
                nsdId = trials[val]
                ground_truth_np_array = read_images(image_index=[nsdId], show=True)
                ground_truth = Image.fromarray(ground_truth_np_array[0])
                ground_truth = ground_truth.resize((768, 768), resample=Image.Resampling.LANCZOS)
                rows = int(math.ceil(len(image_list)/2 + 4))
                columns = 2
                images = [ground_truth, scs_reconstruction, target_v, output_v, target_c, output_c, target_cv, output_cv]
                captions = ["Ground Truth", "Search Reconstruction", "Ground Truth VDVAE", "Decoded VDVAE", "Ground Truth CLIP", "Decoded CLIP Only", "Ground Truth CLIP+VDVAE", "Decoded CLIP+VDVAE"]
                intermediate_clip_scores = []
                for j in range(len(image_list)):
                    images.append(image_list[j])
                    intermediate_clip = SCS.R.encode_image_raw(image_list[j]).reshape((1024,)).to("cpu")
                    intermediate_clip_scores.append(float(PeC1(intermediate_clip.to("cpu"), targets_clips[i].reshape((1024,)).to("cpu").detach())))
                    captions.append("BC: {} CLIP: {}".format(round(score_list[j], 3), round(intermediate_clip_scores[j], 3)))
                    
                figure = tileImages("idx :{}".format(val), images, captions, rows, columns)
                scs_reconstruction.save("{}/Search Reconstruction.png".format(sample_path))
                figure.save("{}/iteration_diagram.png".format(sample_path))
                count = 0
                images.append(process_image(output_images[i]))
                captions.append("Library Reconstruction")
                if(args.log):
                    for j in range(len(images)):
                        if("BC" in captions[j]):
                            images[j].save("{}/iter_{}.png".format(sample_path, count))
                            count +=1
                        else:
                            images[j].save("{}/{}.png".format(sample_path, captions[j]))