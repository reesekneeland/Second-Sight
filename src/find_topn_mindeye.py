import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from torchmetrics import PearsonCorrCoef
from torchvision import transforms
import torch
import clip
import numpy as np
from PIL import Image
import shutil
from utils import *
from tqdm import tqdm

class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False,
                 hidden_state=False, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"
        print(clip_variant, device)
        
        if clip_variant=="ViT-L/14" and hidden_state:
            from transformers import CLIPVisionModelWithProjection, CLIPModel,CLIPTokenizer,CLIPProcessor
            image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
            # text_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
            # image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",cache_dir="/fsx/proj-medarc/fmri/cache")
            #from transformers import CLIPVisionModelWithProjection
            #sd_cache_dir = '/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
            #image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_cache_dir, subfolder='image_encoder').eval()
            image_encoder = image_encoder.to(device)
            # tokenizer =tokenizer.to(device)
            # processor = processor.to(device)
            model = model.to(device)
            for param in image_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            for param in model.parameters():
                param.requires_grad = False # dont need to calculate gradients
            # for param in text_encoder.parameters():
            #     param.requires_grad = False # dont need to calculate gradients
            self.tokenizer = tokenizer
            self.processor = processor
            self.model = model
            self.image_encoder = image_encoder
        elif hidden_state:
            raise Exception("hidden_state embeddings only works with ViT-L/14 right now")
        
        clip_model, preprocess = clip.load(clip_variant, device=device)
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        self.clip = clip_model
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448,448)
        else:
            self.clip_size = (224,224)
            
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc
        self.hidden_state = hidden_state
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.device= device
        
        def versatile_normalize_embeddings(encoder_output):
            # original code:
            # ---------------------------------------------------------------
            # outputs = self.model.vision_model(pixel_values=pixels)
            # z = outputs.last_hidden_state
            # z = self.model.vision_model.post_layernorm(z)
            # z = self.model.visual_projection(z)
            # z_pooled = z[:, 0:1]
            # z = z / torch.norm(z_pooled, dim=-1, keepdim=True)
            # ---------------------------------------------------------------
            embeds = encoder_output.last_hidden_state
            embeds = image_encoder.vision_model.post_layernorm(embeds)
            embeds = image_encoder.visual_projection(embeds)
            return embeds
        self.versatile_normalize_embeddings = versatile_normalize_embeddings

        def versatile_normalize_text_embeddings(encoder_output):
            # original code:
            # ---------------------------------------------------------------
            # outputs = self.model.text_model(input_ids=tokens)
            # z = self.model.text_projection(outputs.last_hidden_state)
            # z_pooled = self.model.text_projection(outputs.pooler_output)
            # z = z / torch.norm(z_pooled.unsqueeze(1), dim=-1, keepdim=True)
            # ---------------------------------------------------------------
            txt_embeds = self.model.text_projection(encoder_output.last_hidden_state)
            txt_pooled = self.model.text_projection(encoder_output.pooler_output)
            return txt_embeds, txt_pooled
        self.versatile_normalize_text_embeddings = versatile_normalize_text_embeddings


    def resize_image(self, image):
        # note: antialias should be False if planning to use Pinkney's Image Variation SD model
        return transforms.Resize(self.clip_size)(image.to(self.device))

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        if self.hidden_state:
            # clip_emb = self.preprocess((image/1.5+.25).to(self.device)) # for some reason the /1.5+.25 prevents oversaturation
            clip_emb = self.preprocess((image).to(self.device))
            clip_emb = self.image_encoder(clip_emb)
            clip_emb = self.versatile_normalize_embeddings(clip_emb)
        else:
            clip_emb = self.preprocess(image.to(self.device))
            clip_emb = self.clip.encode_image(clip_emb)
        # input is now in CLIP space, but mind-reader preprint further processes embeddings:
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
        if self.norm_embs:
            if self.hidden_state:        
                # normalize all tokens by cls token's norm
                clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        return clip_emb

    def embed_text(self, text_samples):
        if self.hidden_state:
            batch_encoding = self.tokenizer(text_samples, truncation=True, max_length=77, return_length=True,
            return_overflowing_tokens=False, padding="max_length", return_tensors="pt").to(self.device)
            tokens = batch_encoding["input_ids"].to(self.device)
            clip_emb = self.model.text_model(input_ids=tokens)
            clip_emb, clip_pooled = self.versatile_normalize_text_embeddings(clip_emb)  
        else:          
            clip_text = clip.tokenize(text_samples).to(self.device)
            clip_text = self.clip.encode_text(clip_text)
        if self.clamp_embs:
            clip_text = torch.clamp(clip_text, -1.5, 1.5)
        if self.norm_embs:
            if self.hidden_state:        
                # normalize all tokens by cls token's norm
                clip_text = clip_emb / torch.norm(clip_pooled.unsqueeze(1), dim=-1).reshape(-1, 1, 1)
            else:
                clip_text = nn.functional.normalize(clip_text, dim=-1)
        return clip_text

    # def org_embed_text(self, text_samples):
    #     clip_text = clip.tokenize(text_samples).to(self.device)
    #     clip_text = self.clip.encode_text(clip_text)
    #     if self.clamp_embs:
    #         clip_text = torch.clamp(clip_text, -1.5, 1.5)
    #     if self.norm_embs:
    #         clip_text = nn.functional.normalize(clip_text, dim=-1)
    #     return clip_text

    def embed_curated_annotations(self, annots):
        for i,b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5,(1,1))[0][0]
                t = b[0,rand]
            if i==0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt,t))
        txt = txt.flatten()
        return self.embed_text(txt)
device="cuda"
clip_extractor = Clipper("ViT-L/14", device="cuda", hidden_state=True, norm_embs=True)
# subject_betas = []
# for s, subject in enumerate([1,2,5,7]):
#     _, _, x_test, _, _, _, _ = load_nsd(vector="images", subject=subject, loader=False, average=True, nest=False, include_heldout=True)
#     subject_betas.append(x_test.to("cpu"))
root = "output/boi_paper"
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

model = "mindeye1"
folder = "brain-optimized-inference-v2.1"

for n in range(1, 64):
    PeC = PearsonCorrCoef(num_outputs=n)
    for mode in ["clip"]:
        new_model_name = f"{model}_top{n}_{mode}"
        # using the rep dimension for subjects
        for s, subject in enumerate([1,2,5,7]):
            all_recons = torch.zeros((1000, 1, 3, 512, 512))
            print(f"Model: {model}, Folder: {folder}, N: {n}, Mode: {mode}, Subject: {subject}")
            # x_test = subject_betas[s]
            for sample in tqdm(range(1000)):
                source_dir = f"{root}/raw/{folder}/subject{subject}/{sample}/"
                dest_dir = f"{root}/raw/{new_model_name}/subject{subject}/{sample}/"
                os.makedirs(dest_dir, exist_ok=True)
                try:
                    shutil.copy(f"{source_dir}/clip_best.pt", f"{dest_dir}/clip.pt")
                except:
                    shutil.copy(f"{source_dir}/clip_vision.pt", f"{dest_dir}/clip.pt")
                try:
                    shutil.copy(f"{source_dir}/basemodel_recon.png", f"{dest_dir}/basemodel_recon.png")
                except:
                    shutil.copy(f"{source_dir}/MindEye.png", f"{dest_dir}/basemodel_recon.png")
                shutil.copy(f"{source_dir}/low_level.png", f"{dest_dir}/low_level.png")
                if mode == "clip":
                    try:
                        target_clip = torch.load(f"{source_dir}/clip_best.pt", map_location="cpu").reshape((1,197376))
                    except:
                        target_clip = torch.load(f"{source_dir}/clip_vision.pt", map_location="cpu").reshape((1,197376))
                    clip_path = f"{source_dir}/iter_0/clip/"
                    # Make the CLIP embeddings
                    os.makedirs(clip_path, exist_ok=True)
                    iter0_clip_path = f"{source_dir}/iter_0/iter0_clip.pt"
                    if not os.path.exists(iter0_clip_path):
                        iter0_images = torch.zeros((64, 3, 512, 512))
                        for i in range(64):
                            image_path = f"{source_dir}/iter_0/images/{i}.png"
                            image = Image.open(image_path).convert("RGB")
                            iter0_images[i] = transforms.ToTensor()(image)
                        iter0_clip = clip_extractor.embed_image(iter0_images.to(device)).reshape((64, target_clip.shape[-2] * target_clip.shape[-1])).to("cpu")
                        torch.save(iter0_clip, iter0_clip_path)
                    else:
                        iter0_clip = torch.load(iter0_clip_path, map_location="cpu")
                    # Calculate cosine similarity
                    similarities = cos(target_clip, iter0_clip[:n])
                    # print("similarity length", len(similarities), target_clip.shape, iter0_clip.shape)
                else:
                    beta = x_test[sample]
                    beta_prime_path = f"{source_dir}/iter_0/beta_primes"
                    beta_primes = torch.zeros((n, beta.shape[-1]))
                    for i in range(n):
                        beta_primes[i] = torch.load(f"{beta_prime_path}/{i}.pt", map_location="cpu")
                    similarities = PeC(beta.repeat(n, 1).moveaxis(0, 1), beta_primes.moveaxis(0, 1))
                    
                # Copy the best image
                best_image_idx = torch.argmax(similarities).item()
                best_image_path = f"{source_dir}/iter_0/images/{best_image_idx}.png"
                shutil.copy(best_image_path, f"{dest_dir}/0.png")
                all_recons[sample] = transforms.ToTensor()(Image.open(best_image_path).resize((512,512)).convert("RGB")).unsqueeze(0)
                
            torch.save(all_recons, f"{root}/evals/{new_model_name}_subj0{subject}_all_recons.pt")