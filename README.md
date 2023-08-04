# Second Sight: Using brain-optimized encoding models to align image distributions with human brain activity


![](data/charts/Pipeline_Diagram.png)<br>

Project page: 

arXiv preprint: https://arxiv.org/abs/2306.00927

## Installation instructions

1. Download this repository:
 ```
 git clone https://github.com/reesekneeland/Second-Sight.git
``` 

2. Create and activate conda environment:
```
cd Second-Sight
conda env create -f environment.yml
conda activate SS
```

3. Aquire a copy of the Natural Scenes Dataset, you can download the relevant files for this project using aws via our setup script:
```
python data/download_nsddata.py
```
Or by downloading/moving your own copy to the ```/data``` directory. To download your own copy, start by agreeing to the Natural Scenes Dataset's [Terms and Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) and fill out the [NSD Data Access form.](https://forms.gle/xue2bCdM9LaFNMeb7)

4. CONTINUE SETUP

## General information

This repository contains scripts for 

1. Training the required models for Second Sight (```src/clip_encoder.py```, ```src/autoencoder.py```)
2. Reconstructing images from brain activity using the model architecture (```src/second_sight.py```)
3. Evaluating reconstructions against the ground truth images according to low- and high-level image metrics (**TODO**) 

### Pre-trained Subject models

You can skip training MindEye yourself and instead run the rest of the notebooks on Subject 1 of NSD by downloading our pre-trained models available on [huggingface](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/mindeye_models) and putting these folders containing model checkpoints inside the ```models``` folder, or by running our download script to perform this step automatically:
```
python data/download_weights.py

Options:
  --gnet                flag to download only the GNet model (1.18GB), as all other models are trainable.

  --subj [1,2,5,7]      list of subjects to download models for, if not specified, will run on all subjects
```


## Training MindEye (high-level pipeline)

Train MindEye via ``Train_MindEye.py``.

- Set ``data_path`` to the folder containing the Natural Scenes Dataset (will download there if not found; >30Gb per subject, only downloads data for the current subject).
- Set ``model_name`` to what you want to name the model, used for saving.
- Set ``--no-hidden --no-norm_embs`` if you want to map to the final layer of CLIP for LAION-5B retrieval or to reconstruct via Stable Diffusion (Image Variations). Otherwise, Versatile Diffusion uses the default ``--hidden --norm_embs``.

Various arguments can be set (see below) for training; the default is to train MindEye to the last hidden layer of CLIP ViT-L/14 using the same settings as our paper, for Subject 1 of NSD.

Trained model checkpoints will be saved inside a folder "fMRI-reconstruction-NSD/train_logs". All other outputs get saved inside "fMRI-reconstruction-NSD/src" folder.

```bash
$ python Train_MindEye.py --help
```
```
usage: Train_MindEye.py [-h] [--model_name MODEL_NAME] [--data_path DATA_PATH]
                        [--subj {1,2,5,7}] [--batch_size BATCH_SIZE]
                        [--hidden | --no-hidden]
                        [--clip_variant {RN50,ViT-L/14,ViT-B/32,RN50x64}]
                        [--wandb_log | --no-wandb_log]
                        [--resume_from_ckpt | --no-resume_from_ckpt]
                        [--wandb_project WANDB_PROJECT]
                        [--mixup_pct MIXUP_PCT] [--norm_embs | --no-norm_embs]
                        [--use_image_aug | --no-use_image_aug]
                        [--num_epochs NUM_EPOCHS] [--prior | --no-prior]
                        [--v2c | --no-v2c] [--plot_umap | --no-plot_umap]
                        [--lr_scheduler_type {cycle,linear}]
                        [--ckpt_saving | --no-ckpt_saving]
                        [--ckpt_interval CKPT_INTERVAL]
                        [--save_at_end | --no-save_at_end] [--seed SEED]
                        [--max_lr MAX_LR] [--n_samples_save {0,1}]
                        [--use_projector | --no-use_projector]
                        [--vd_cache_dir VD_CACHE_DIR]

Model Training Configuration

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        name of model, used for ckpt saving and wandb logging
                        (if enabled)
  --data_path DATA_PATH
                        Path to where NSD data is stored / where to download
                        it to
  --subj {1,2,5,7}
  --batch_size BATCH_SIZE
                        Batch size can be increased by 10x if only training
                        v2c and not diffusion prior
  --hidden, --no-hidden
                        if True, CLIP embeddings will come from last hidden
                        layer (e.g., 257x768 - Versatile Diffusion), rather
                        than final layer (default: True)
  --clip_variant {RN50,ViT-L/14,ViT-B/32,RN50x64}
                        OpenAI clip variant
  --wandb_log, --no-wandb_log
                        whether to log to wandb (default: False)
  --resume_from_ckpt, --no-resume_from_ckpt
                        if not using wandb and want to resume from a ckpt
                        (default: False)
  --wandb_project WANDB_PROJECT
                        wandb project name
  --mixup_pct MIXUP_PCT
                        proportion of way through training when to switch from
                        BiMixCo to SoftCLIP
  --norm_embs, --no-norm_embs
                        Do l2-norming of CLIP embeddings (default: True)
  --use_image_aug, --no-use_image_aug
                        whether to use image augmentation (default: True)
  --num_epochs NUM_EPOCHS
                        number of epochs of training
  --prior, --no-prior   if False, will only use CLIP loss and ignore diffusion
                        prior (default: True)
  --v2c, --no-v2c       if False, will only use diffusion prior loss (default:
                        True)
  --plot_umap, --no-plot_umap
                        Plot UMAP plots alongside reconstructions (default:
                        False)
  --lr_scheduler_type {cycle,linear}
  --ckpt_saving, --no-ckpt_saving
  --ckpt_interval CKPT_INTERVAL
                        save backup ckpt and reconstruct every x epochs
  --save_at_end, --no-save_at_end
                        if True, saves best.ckpt at end of training. if False
                        and ckpt_saving==True, will save best.ckpt whenever
                        epoch shows best validation score (default: False)
  --seed SEED
  --max_lr MAX_LR
  --n_samples_save {0,1}
                        Number of reconstructions for monitoring progress, 0
                        will speed up training
  --use_projector, --no-use_projector
                        Additional MLP after the main MLP so model can
                        separately learn a way to minimize NCE from prior loss
                        (BYOL) (default: True)
  --vd_cache_dir VD_CACHE_DIR
                        Where is cached Versatile Diffusion model; if not
                        cached will download to this path
```

## Reconstructing from pre-trained MindEye

Now that you have pre-trained model ckpts in your "train_logs" folder, either from running ``Train_MindEye.py`` or by downloading our pre-trained Subject 1 models from [huggingface](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/mindeye_models), we can proceed to reconstructing images from the test set of held-out brain activity. 

``Reconstructions.py`` defaults to outputting Versatile Diffusion reconstructions as a torch .pt file, without img2img and without second-order selection (recons_per_sample=1).

- Set ``data_path`` to the folder containing the Natural Scenes Dataset (will download there if not found; >30Gb per subject, only downloads data for the current subject).
- Set ``model_name`` to the name of the folder contained in "fMRI-reconstruction-NSD/train_logs" that contains the ckpt mapping brain activity to the last hidden layer of CLIP.
- If you want to use img2img, set ``autoencoder_name`` to the name of the folder contained in "fMRI-reconstruction-NSD/train_logs" that contains the ckpt mapping brain activity to the variational autoencoder of Stable Diffusion. 
- If you are using img2img, set ``img2img_strength`` to the level of guidance you prefer, where 1=no img2img and 0=outputs solely from the low-level pipeline.

```bash
$ python Reconstructions.py --help
```
```
usage: Reconstructions.py [-h] [--model_name MODEL_NAME]
                          [--autoencoder_name AUTOENCODER_NAME] [--data_path DATA_PATH]
                          [--subj {1,2,5,7}] [--img2img_strength IMG2IMG_STRENGTH]
                          [--recons_per_sample RECONS_PER_SAMPLE]
                          [--vd_cache_dir VD_CACHE_DIR]

Model Training Configuration

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        name of trained model
  --autoencoder_name AUTOENCODER_NAME
                        name of trained autoencoder model
  --data_path DATA_PATH
                        Path to where NSD data is stored (see README)
  --subj {1,2,5,7}
  --img2img_strength IMG2IMG_STRENGTH
                        How much img2img (1=no img2img; 0=outputting the low-level image
                        itself)
  --recons_per_sample RECONS_PER_SAMPLE
                        How many recons to output, to then automatically pick the best
                        one (MindEye uses 16)
  --vd_cache_dir VD_CACHE_DIR
                        Where is cached Versatile Diffusion model; if not cached will
                        download to this path
```

## Image/Brain Retrieval (inc. LAION-5B image retrieval)

To evaluate image/brain retrieval using the NSD test set then use the Jupyter notebook ``Retrievals.ipynb`` and follow the code blocks under the "Image/Brain Retrieval" heading.

Running ``Retrievals.py`` will retrieve the top 16 nearest neighbors in LAION-5B based on the MindEye variant where brain activity is mapped to the final layer of CLIP. This is followed by second-order selection where the 16 retrieved images are converted to CLIP last hidden layer embeddings and compared to the MindEye outputs from the core model where brain activity is mapped to the last hidden layer of CLIP. The highest CLIP similarity retrieved image will be chosen, with all top-1 retrievals saved to a torch .pt file.

- Set ``data_path`` to the folder containing the Natural Scenes Dataset (will download there if not found; >30Gb per subject, only downloads data for the current subject).
- Set ``model_name`` to the name of the folder contained in "fMRI-reconstruction-NSD/train_logs" that contains the ckpt mapping brain activity to the last hidden layer of CLIP.
- Set ``model_name2`` to the name of the folder contained in "fMRI-reconstruction-NSD/train_logs" that contains the ckpt mapping brain activity to the final layer of CLIP.

```bash
$ python Retrievals.py --help
```
```
usage: Retrievals.py [-h] [--model_name MODEL_NAME]
                               [--model_name2 MODEL_NAME2] [--data_path DATA_PATH]
                               [--subj {1,2,5,7}]

Model Training Configuration

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        name of 257x768 model, used for everything except LAION-5B
                        retrieval
  --model_name2 MODEL_NAME2
                        name of 1x768 model, used for LAION-5B retrieval
  --data_path DATA_PATH
                        Path to where NSD data is stored (see README)
  --subj {1,2,5,7}
```


## Evaluating Reconstructions

After you have saved a .pt file from running ``Reconstructions.py`` or ``Retrievals.py``, you can use ``Reconstruction_Metrics.py`` to evaluate reconstructed images using the same low- and high-level image metrics used in the paper.

- Set ``recon_path`` to the name of the file in "fMRI-reconstruction-NSD/src" that was output from ``Reconstructions.py`` (should be ```{model_name}_recons_img2img{img2img_strength}_{recons_per_sample}samples.pt```). 
- Alternatively, to evaluate LAION-5B retrievals, you can replace recon_path with the name of the .pt file output from ```Retrievals.py``` (should be ```{model_name}_laion_retrievals_top16.pt```).
- Set ``all_images_path`` to the all_images.pt file in "fMRI-reconstruction-NSD/src" that was output from either ``Reconstructions.py`` or ``Retrievals.py`` (should be ```all_images.pt```). 

```bash
$ python Reconstruction_Metrics.py --help
```
```
usage: Reconstruction_Metrics.py [-h] [--recon_path RECON_PATH]
                                 [--all_images_path ALL_IMAGES_PATH]

Model Training Configuration

options:
  -h, --help            show this help message and exit
  --recon_path RECON_PATH
                        path to reconstructed/retrieved outputs
  --all_images_path ALL_IMAGES_PATH
                        path to ground truth outputs
```

## Training MindEye (low-level pipeline)

Under construction (see train_autoencoder.py)

# Citation

If you make use of this work please cite both the Second Sight paper and the Natural Scenes Dataset paper.

```bibtex
@misc{kneeland2023second,
      title={Second Sight: Using brain-optimized encoding models to align image distributions with human brain activity}, 
      author={Reese Kneeland and Jordyn Ojeda and Ghislain St-Yves and Thomas Naselaris},
      year={2023},
      eprint={2306.00927}
}

@article{Allen_2021_NSD, 
      title={A massive 7T fmri dataset to Bridge Cognitive Neuroscience and artificial intelligence}, 
      volume={25}, 
      DOI={10.1038/s41593-021-00962-x}, 
      number={1}, 
      journal={Nature Neuroscience}, 
      author={Allen, Emily J. and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse L. and Prince, Jacob S. and Dowdle, Logan T. and Nau, Matthias and Caron, Brad and Pestilli, Franco and Charest, Ian and et al.}, 
      year={2021}, 
      pages={116–126}
} 
```


## Computational Overhead

# setup_data.py runtime:
  - Vector Processing: Approximately 3 hours 
  - Brain Data Processing: 