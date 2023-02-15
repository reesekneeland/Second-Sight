def load_data_3D(vector, only_test=False):
        
        # 750 trails per session 
        # 40 sessions per subject
        # initialize some empty tensors to allocate memory
        
        
        if(vector == "c"):
            y_train = torch.empty((25500, 77, 1024))
            y_test  = torch.empty((2250, 77, 1024))
        elif(vector == "z"):
            y_train = torch.empty((25500, 4, 64, 64))
            y_test  = torch.empty((2250, 4, 64, 64))
        
        
        # 34 * 750 = 25500
        x_train = torch.empty((25500, 42, 22, 27))
        
        # 3 * 750 = 2250
        x_test  = torch.empty((2250, 42, 22, 27))
        
        #grabbing voxel mask for subject 1
        voxel_mask = voxel_data_dict['voxel_mask']["1"]
        voxel_mask_reshape = voxel_mask.reshape((81, 104, 83, 1))

        slices = get_slices(voxel_mask_reshape)

        # Checks if we are only loading the test data so we don't have to load all the training betas
        if(not only_test):
            
            # Loads the full collection of beta sessions for subject 1
            for i in tqdm(range(1,35), desc="Loading Training Voxels"):
                beta = nsda.read_betas(subject='subj01', 
                                    session_index=i, 
                                    trial_index=[], # Empty list as index means get all 750 scans for this session
                                    data_type='betas_fithrf_GLMdenoise_RR',
                                    data_format='func1pt8mm')
                roi_beta = np.where((voxel_mask_reshape), beta, 0)
                beta_trimmed = roi_beta[slices] 
                beta_trimmed = np.moveaxis(beta_trimmed, -1, 0)
                x_train[(i-1)*750:(i-1)*750+750] = torch.from_numpy(beta_trimmed)
        
        for i in tqdm(range(35,38), desc="Loading Test Voxels"):
            
            # Loads the test betas and puts it into a tensor
            test_betas = nsda.read_betas(subject='subj01', 
                                        session_index=i, 
                                        trial_index=[], # Empty list as index means get all for this session
                                        data_type='betas_fithrf_GLMdenoise_RR',
                                        data_format='func1pt8mm')
            roi_beta = np.where((voxel_mask_reshape), test_betas, 0)
            beta_trimmed = roi_beta[slices] 
            beta_trimmed = np.moveaxis(beta_trimmed, -1, 0)
            x_test[(i-35)*750:(i-35)*750+750] = torch.from_numpy(beta_trimmed)

        # Loading the description object for subejct1
        subj1y = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]

        for i in tqdm(range(0,25500), desc="Loading Training Vectors"):
            # Flexible to both Z and C tensors depending on class configuration
            index = int(subj1y.loc[(subj1y['subject1_rep0'] == i+1) | (subj1y['subject1_rep1'] == i+1) | (subj1y['subject1_rep2'] == i+1)].nsdId)
            y_train[i] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt")

        for i in tqdm(range(0,2250), desc="Loading Test Vectors"):
            index = int(subj1y.loc[(subj1y['subject1_rep0'] == 25501 + i) | (subj1y['subject1_rep1'] == 25501 + i) | (subj1y['subject1_rep2'] == 25501 + i)].nsdId)
            y_test[i] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt")

        if(vector == "c"):
            y_train = y_train.reshape((25500, 78848))
            y_test  = y_test.reshape((2250, 78848))
        elif(vector == "z"):
            y_train = y_train.reshape((25500, 16384))
            y_test  = y_test.reshape((2250, 16384))
            
        x_train = x_train.reshape((25500, 1, 42, 22, 27))
        x_test  = x_test.reshape((2250, 1, 42, 22, 27))

        print("3D STATS PRENORM", torch.max(x_train), torch.var(x_train))
        x_train_mean, x_train_std = x_train.mean(), x_train.std()
        x_test_mean, x_test_std = x_test.mean(), x_test.std()
        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_test_mean) / x_test_std

        print("3D STATS NORMALIZED", torch.max(x_train), torch.var(x_train))
        return x_train, x_test, y_train, y_test
    
# CNN Class, very rough still
class CNN(torch.nn.Module):
    def __init__(self, vector):
        super(CNN, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.flatten = nn.Flatten(start_dim=1)
        if(vector == "c"):
            self.fc1 = nn.Linear(64*9*4*5, 78848)
        elif(vector == "z"):
            self.fc1 = nn.Linear(64*9*4*5,  16384)
        # self.relu = nn.LeakyReLU()
        # self.batch=nn.BatchNorm1d(64*9*4*5)
        # self.drop=nn.Dropout(p=0.15)
        
            
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def forward(self, x):
        # print("size1: ", x.shape)
        out = self.conv_layer1(x)
        # print("size2 out size: ", out.shape)
        out = self.conv_layer2(out)
        # print("flattened out size: ", out.shape)
        out = self.flatten(out)
        # print("flattened out size: ", out.shape)
        # out = self.relu(out)
        # out = self.batch(out)
        out = self.fc1(out)
        return out
    
    
class Encoder():
    def __init__(self):
        self.config = OmegaConf.load("stablediffusion/configs/stable-diffusion/v2-inference.yaml")
        self.model = self.load_model_from_config(self.config, "stablediffusion/models/v2-1_512.ckpt")
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.scale = 9.0
        self.sampler = DDIMSampler(self.model)

        os.makedirs("reconstructions/samples", exist_ok=True)
        self.outpath = "reconstructions/samples"
        self.base_count = len(os.listdir(self.outpath))+3
        
    def load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        return model


    # Encode latent z (1x4x64x64) and condition c (1x77x1024) tensors into an image
    # Strength parameter controls the weighting between the two tensors
    def reconstruct(self, z, c, strength=0.8):
        # seed_everything(41)
        init_latent = z.reshape((1,4,64,64)).to(self.device)
        self.sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * 50)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    uc = None
                    if self.scale != 1.0:
                        uc = self.model.get_learned_conditioning(1 * [""])
                    c = c.reshape((1,77,1024)).to(self.device)
                    z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * 1).to(self.device))
                    samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=self.scale,
                                            unconditional_conditioning=uc, )

                    x_sample = self.model.decode_first_stage(samples)
                    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)

                    x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(self.outpath, f"{self.base_count:05}.png"))
                    self.base_count += 1
        return img
    


    if __name__ == "__main__":
        z = torch.load(sys.argv[1])
        c = torch.load(sys.argv[2])
        strength = float(sys.argv[3])    
        reconstruct(z, c, strength)

def run_fr_decoder():
    hashNum = update_hash()
    #hashNum = "179"
    D = RidgeDecoder(hashNum = hashNum,
                vector="c_combined", 
                 log=True, 
                 threshold=0.06397,
                 device="cuda",
                 n_alphas=20
                 )
    hashNum, outputs, target = D.train()
    return hashNum, outputs, target
