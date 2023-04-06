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


 def get_percentile(self, threshold):
        print(self.device)
        print("finding percentile for " + str(threshold))
        masked_threshold_x = self.mask_voxels(threshold, self.x_thresh.to(self.device))
        masked_threshold_encoded_x = self.mask_voxels(threshold, self.x_thresh_encoded.to(self.device))
        
        mask = torch.load(mask_path + str(threshold) + ".pt", map_location=self.device)
        PeC = PearsonCorrCoef(num_outputs=22735).to(self.device)
        PeC2 = PearsonCorrCoef().to(self.device)
        average_percentile = 0
        for i in tqdm(range(masked_threshold_x.shape[0]), desc="scanning library for threshold " + str(threshold)):
            
            xDup = masked_threshold_x[i].repeat(22735, 1).moveaxis(0, 1).to(self.device)
            scores = torch.zeros((2819141,))
            for batch in tqdm(range(124), desc="batching sample"):
                x_preds = torch.load(latent_path + "/cc3m_batches/" + str(batch) + ".pt", map_location=self.device)
                x_preds_m = x_preds[:, mask]
                x_preds_t = x_preds_m.moveaxis(0, 1)
                
                # Pearson correlation
                scores[22735*batch:22735*batch+22735] = PeC(xDup, x_preds_t).detach()
            scores[-1] = PeC2(masked_threshold_x[i], masked_threshold_encoded_x[i]).detach()
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            percentile = 1-float(sorted_indices[-1]/2819141)
            average_percentile += percentile
            tqdm.write(str(percentile))
        final_percentile = average_percentile/masked_threshold_x.shape[0]
        print("final percentile: ", str(final_percentile))
        file = open(mask_path + "results.txt", 'a+')
        file.write(str(threshold) + ": " + str(final_percentile))
        file.close()
        
        
        
        #Single trial search cross validation function (not working)
        if(self.cross_validate):
            #iterate over the 3 scans
            beta_mask = self.masks[0]
            for m in mask:
                beta_mask = torch.logical_or(beta_mask, self.masks[m])
            xDup = []
            for i in range(beta.shape[0]):
                #generating beta for current scan, and masking
                cur_beta = beta[i]
                cur_beta = cur_beta[beta_mask]
                xDup.append(cur_beta.repeat(n, 1).moveaxis(0, 1).to(self.device))
            for i in range(beta.shape[0]):
                images = []
                iter_scores = []
                best_vector_corrrelation = -1
                best_image_i = None
                z=None
                for cur_iter in tqdm(range(max_iter), desc="search iterations"): 
                    #configure strength parameter based on depth of search
                    strength = 1.0-0.7*(cur_iter/max_iter)
                    #setup previous best image as Z vector for new step of search
                    if(best_image_i):
                        im_tensor = self.R.im2tensor(best_image)
                        z = self.R.encode_latents(im_tensor)
                        #generate samples
                    samples = self.generateNSamples(n, clip, z, strength)
                    #evaluate samples
                    beta_primes = self.Alexnet.predict(samples)
                    beta_primes = beta_primes[:, beta_mask]
                    beta_primes = beta_primes.moveaxis(0, 1).to(self.device)
                    cv_scores = []
                    for j in range(beta.shape[0]):
                        if i != j:
                            score = PeC(xDup[j], beta_primes)
                            cv_scores.append(torch.max(score))
                    cv_score_mean = sum(cv_scores)/len(cv_scores)
                    single_trial_scores = PeC(xDup[i], beta_primes)
                    cur_vector_corrrelation = float(torch.max(single_trial_scores))
                    print("current scan score vs cross validation score average: ", cur_vector_corrrelation, cv_score_mean)
                    if(self.log):
                        wandb.log({'Current trial correlation': cur_vector_corrrelation, 'Cross validated trial correlation': cv_score_mean})
                    print(cur_vector_corrrelation)
                    images.append(samples[int(torch.argmax(single_trial_scores))])
                    iter_scores.append(cur_vector_corrrelation)
                    print("scores, cv_score, cv_best, st_score, st_best", cv_score_mean, best_cv_vector_correlation, cur_vector_corrrelation, best_vector_corrrelation)
                    if cv_score_mean > best_cv_vector_correlation or best_cv_vector_correlation == -1:
                        best_cv_vector_correlation = cv_score_mean
                        best_image = samples[int(torch.argmax(single_trial_scores))]
                    if cur_vector_corrrelation > best_vector_corrrelation or best_vector_corrrelation == -1:
                        best_vector_corrrelation = cur_vector_corrrelation
                        best_image_i = samples[int(torch.argmax(single_trial_scores))]
                    else:
                        loss_counter +=1
                        
                        
def load_nsd(vector, batch_size=375, num_workers=16, loader=True, split=True, ae=False, encoderModel=None, average=False, return_trial=False, old_norm=False, nest=False):
    if(old_norm):
        region_name = "whole_region_11838_old_norm.pt"
    else:
        region_name = "whole_region_11838.pt"
    if(ae):
        x = torch.load(prep_path + "x_encoded/" + encoderModel + "/" + "vector.pt").requires_grad_(False)
        y = torch.load(prep_path + "x/" + region_name).requires_grad_(False)
    else:
        x = torch.load(prep_path + "x/" + region_name).requires_grad_(False)
        y = torch.load(prep_path + vector + "/vector.pt").requires_grad_(False)
    
    if(not split): 
        return x, y
    
    else: 
        x_train, x_val, x_voxelSelection, x_thresholdSelection, x_test = [], [], [], [], []
        y_train, y_val, y_voxelSelection, y_thresholdSelection, y_test = [], [], [], [], []
        subj1_train = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == False)]
        subj1_test = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0) & (nsda.stim_descriptions['shared1000'] == True)]
        subj1_full = nsda.stim_descriptions[(nsda.stim_descriptions['subject1'] != 0)]
        alexnet_stimuli_order_list = np.where(subj1_full["shared1000"] == True)[0]
        
        # Loads the raw tensors into a Dataset object

        # TensorDataset takes in two tensors of equal size and then maps 
        # them to one dataset. 
        # x is the brain data 
        # y are the vectors
        # train_i, test_i, val_i, voxelSelection_i, thresholdSelection_i = 0,0,0,0,0
        alexnet_stimuli_ordering  = []
        test_trials = []
        for i in tqdm(range(7500), desc="loading training samples"):
            if(average==True):
                avx = []
                avy = []
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                
                    if(scanId < 27750):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                if(len(avx) > 0):
                    avx = torch.stack(avx)
                    x_train.append(torch.mean(avx, dim=0))
                    y_train.append(avy[0])
            else:
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                        x_train.append(x[scanId-1])
                        y_train.append(y[scanId-1])
                        
                        
        for i in tqdm(range(7500, 9000), desc="loading validation samples"):
            if(average==True):
                avx = []
                avy = []
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                
                    if(scanId < 27750):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                if(len(avx)>0):
                    avx = torch.stack(avx)
                    x_val.append(torch.mean(avx, dim=0))
                    y_val.append(avy[0])
            
            else:
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                            x_val.append(x[scanId-1])
                            y_val.append(y[scanId-1])
        
        for i in range(200):
            nsdId = subj1_train.iloc[i]['nsdId']
            if(average==True):
                avx = []
                avy = []
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                
                    if(scanId < 27750):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                if(len(avx)>0):
                    avx = torch.stack(avx)
                    x_voxelSelection.append(torch.mean(avx, dim=0))
                    y_voxelSelection.append(avy[0])
            else:
                for j in range(3):
                    scanId = subj1_test.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                        if(return_trial): 
                            x_test.append(x[scanId-1])
                        x_voxelSelection.append(x[scanId-1])
                        y_voxelSelection.append(y[scanId-1])
                        alexnet_stimuli_ordering.append(alexnet_stimuli_order_list[i])
                    
        for i in range(200, 400):
            nsdId = subj1_train.iloc[i]['nsdId']
            if(average==True): 
                avx = []
                avy = []
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                
                    if(scanId < 27750):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                if(len(avx)>0):
                    avx = torch.stack(avx)
                    x_thresholdSelection.append(torch.mean(avx, dim=0))
                    y_thresholdSelection.append(avy[0])
            else:
                for j in range(3):
                    scanId = subj1_test.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                        if(return_trial): 
                            x_test.append(x[scanId-1])
                        x_thresholdSelection.append(x[scanId-1])
                        y_thresholdSelection.append(y[scanId-1])
                        alexnet_stimuli_ordering.append(alexnet_stimuli_order_list[i])
                    
        for i in range(400, 1000):
            nsdId = subj1_train.iloc[i]['nsdId']
            if(average==True):
                avx = []
                avy = []
                for j in range(3):
                    scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                    if(scanId < 27750):
                        avx.append(x[scanId-1])
                        avy.append(y[scanId-1])
                if(len(avx)>0):
                    avx = torch.stack(avx)
                    x_test.append(torch.mean(avx, dim=0))
                    y_test.append(avy[0])
                    test_trials.append(nsdId)
            else:
                if nest:
                    x_row = torch.zeros((3, 11838))
                    y_row = []
                    valCount = 0
                    for j in range(3):
                        scanId = subj1_train.iloc[i]['subject1_rep' + str(j)]
                        if(scanId < 27750):
                            valCount +=1
                            x_row[j] = x[scanId-1]
                            y_row.append(y[scanId-1])
                    if(valCount > 0):
                        test_trials.append(nsdId)
                        x_test.append(x_row)
                        y_test.append(y_row[0])
                else:
                    for j in range(3):
                        scanId = subj1_test.iloc[i]['subject1_rep' + str(j)]
                        if(scanId < 27750):
                            x_test.append(x[scanId-1])
                            y_test.append(y[scanId-1])
                            test_trials.append(nsdId)
                            alexnet_stimuli_ordering.append(alexnet_stimuli_order_list[i])
        x_train = torch.stack(x_train).to("cpu")
        x_val = torch.stack(x_val).to("cpu")
        x_voxelSelection = torch.stack(x_voxelSelection).to("cpu")
        x_thresholdSelection = torch.stack(x_thresholdSelection).to("cpu")
        x_test = torch.stack(x_test).to("cpu")
        y_train = torch.stack(y_train)
        y_val = torch.stack(y_val)
        y_voxelSelection = torch.stack(y_voxelSelection)
        y_thresholdSelection = torch.stack(y_thresholdSelection)
        y_test = torch.stack(y_test)
        print("shapes: ", x_train.shape, x_val.shape, x_voxelSelection.shape, x_thresholdSelection.shape, x_test.shape, y_train.shape, y_val.shape, y_voxelSelection.shape, y_thresholdSelection.shape, y_test.shape)

        if(loader):
            trainset = torch.utils.data.TensorDataset(x_train, y_train)
            valset = torch.utils.data.TensorDataset(x_val, y_val)
            voxelset = torch.utils.data.TensorDataset(x_voxelSelection, y_voxelSelection)
            thresholdset = torch.utils.data.TensorDataset(x_thresholdSelection, y_thresholdSelection)
            testset = torch.utils.data.TensorDataset(x_test, y_test)
            # Loads the Dataset into a DataLoader
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            voxelloader = torch.utils.data.DataLoader(voxelset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            threshloader = torch.utils.data.DataLoader(thresholdset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            return trainloader, valloader, voxelloader, threshloader, testloader
        else:
            if(return_trial): 
                return x_train, x_val, x_voxelSelection, x_thresholdSelection, x_test, y_train, y_val, y_voxelSelection, y_thresholdSelection, y_test, alexnet_stimuli_ordering, test_trials
            else:
                return x_train, x_val, x_voxelSelection, x_thresholdSelection, x_test, y_train, y_val, y_voxelSelection, y_thresholdSelection, y_test, test_trials

def grab_samples(vector, threshold, hashNum):
    
    whole_region = torch.load(prep_path + "x/whole_region_11838_old_norm.pt") 
    mask = np.load("masks/" + hashNum + "_" + vector + "2voxels_pearson_thresh" + threshold + ".npy")
    new_len = np.count_nonzero(mask)
    target = torch.zeros((27750, new_len))
    for i in tqdm(range(27750), desc=(vector + " masking")): 
       
        # Indexing into the sample and then using y_mask to grab the correct samples. 
        target[i] = whole_region[i][torch.from_numpy(mask)]
    torch.save(target, prep_path + "x/" + vector + "_2voxels_pearson_thresh" + threshold + ".pt")

def compound_loss(pred, target):
        alpha = 0.9
        mse = nn.MSELoss()
        cs = nn.CosineSimilarity()
        loss = alpha * mse(pred, target) + (1 - alpha) * (1- torch.mean(cs(pred, target)))
        return loss
    
def format_clip(c):
    if(len(c.shape)<2):
        c = c.reshape((1,768))
    c_combined = []
    for i in range(c.shape[0]):
        c_combined.append(c[i].reshape((1,768)).to("cuda"))
    
    for j in range(5-c.shape[0]):
        c_combined.append(torch.zeros((1, 768), device="cuda"))
    
    c_combined = torch.cat(c_combined, dim=0).unsqueeze(0)
    c_combined = c_combined.tile(1, 1, 1)
    return c_combined

def extract_dim(vector, dim):
    
    if(vector == "z"):
        vec_target = torch.zeros((27750, 16384))
        datashape = (1, 16384)
    elif(vector == "c"):
        vec_target = torch.zeros((27750, 1536))
        datashape = (1, 1536)
    elif(vector == "c_prompt"):
        vec_target = torch.zeros((27750, 78848))
        datashape = (1, 78848)
    elif(vector == "c_combined" or vector == "c_img_mixer"):
        vec_target = torch.zeros((27750, 768))
        datashape = (1, 768)

    # Loading the description object for subejct1
    
    subj1x = nsda.stim_descriptions[nsda.stim_descriptions['subject1'] != 0]

    for i in tqdm(range(0,27750), desc="vector loader"):
        
        # Flexible to both Z and C tensors depending on class configuration
        
        # TODO: index the column of this table that is apart of the 1000 test set. 
        # Do a check here. Do this in get_data
        # If the sample is part of the held out 1000 put it in the test set otherwise put it in the training set. 
        index = int(subj1x.loc[(subj1x['subject1_rep0'] == i+1) | (subj1x['subject1_rep1'] == i+1) | (subj1x['subject1_rep2'] == i+1)].nsdId)
        full_vec = torch.load("/export/raid1/home/kneel027/nsd_local/nsddata_stimuli/tensors/" + vector + "/" + str(index) + ".pt")
        reduced_dim = full_vec[:,dim]
        vec_target[i] = torch.reshape(reduced_dim, datashape)

    torch.save(vec_target, prep_path + vector + "_" + str(dim) + "/vector.pt")
    
#Ghislain loader code
        def get_last_token(s, tokens={'@': list, '.': dict}):
            l,name,entry,t = 2**31,'','',None
            for tok,toktype in tokens.items():
                ss = s.split(tok)
                if len(ss)>1 and len(ss[-1])<l:
                    l = len(ss[-1])
                    entry = ss[-1]
                    name = tok.join(ss[:-1])
                    t = toktype
            return name, entry, t


        def has_token(s, tokens=['@', '.']):
            isin = False
            for tok in tokens:
                if tok in s:
                    isin = True
            return isin
            
        def extend_list(l, i, v):
            if len(l)<i+1:
                l += [None,]*(i+1-len(l))
            l[i] = v
            return l

        def flatten_dict(base, append=''):
            '''flatten nested dictionary and lists'''
            flat = {}
            for k,v in base.items():
                if type(v)==dict:
                    flat.update(flatten_dict(v, '%s%s.'%(append,k)))
                elif type(v)==list:
                    flat.update(flatten_dict({'%s%s@%d'%(append,k,i): vv for i,vv in enumerate(v)}))
                else:
                    flat['%s%s'%(append,k)] = v
            return flat

        def embed_dict(fd):
            d = {}
            for k,v in fd.items():
                name, entry, ty = get_last_token(k, {'@': list, '.': dict})
                if ty==list:
                    if name in d.keys():
                        d[name] = extend_list(d[name], int(entry), v)
                    else:
                        d[name] = extend_list([], int(entry), v)
                elif ty==dict:
                    if name in d.keys():
                        d[name].update({entry: v})
                    else:
                        d[name] = {entry: v}
                else:
                    if k in d.keys():
                        d[k].update(v)
                    else:
                        d[k] = v   
            return embed_dict(d) if has_token(''.join(d.keys()), tokens=['@', '.']) else d

def predictVector_cc3m(encModel, vector, x, mask=[], device="cuda:0"):
        
        if(vector == "c_img_0" or vector == "c_text_0"):
            datasize = 768
        elif(vector == "z_img_mixer"):
            datasize = 16384
        elif(vector == "images"):
            datasize = 541875
        # x = x.to(device)
        prep_path = "/export/raid1/home/kneel027/nsd_local/preprocessed_data/"
        latent_path = "latent_vectors/"
        
        PeC = PearsonCorrCoef(num_outputs=22735).to(device)
        
        out = torch.zeros((x.shape[0], 5, datasize))
        average_pearson = 0
        
        for i in tqdm(range(x.shape[0]), desc="scanning library for " + vector):
            xDup = x[i].repeat(22735, 1).moveaxis(0, 1).to(device)
            scores = torch.zeros((2819141,))
            # preds = torch.zeros((2819141,datasize))
            # batch_max_x = torch.zeros((620, x.shape[1]))
            # batch_max_y = torch.zeros((620, datasize))
            for batch in tqdm(range(124), desc="batching sample"):
                # y = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt")
                x_preds = torch.load(latent_path + encModel + "/cc3m_batches/" + str(batch) + ".pt")
                # print(x_preds.device)
                x_preds_t = x_preds.moveaxis(0, 1).to(device)
                # preds[22735*batch:22735*batch+22735] = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt")
                # Pearson correlation
                scores[22735*batch:22735*batch+22735] = PeC(xDup, x_preds_t).detach()
                # Calculating the Average Pearson Across Samples
            top5_pearson = torch.topk(scores, 5)
            average_pearson += torch.mean(top5_pearson.values.detach()) 
            print(top5_pearson.indices, top5_pearson.values, scores[0:5])
            for j, index in enumerate(top5_pearson.indices):
                batch = int(index // 22735)
                sample = int(index % 22735)
                batch_preds = torch.load(prep_path + vector + "/cc3m_batches/" + str(batch) + ".pt")
                out[i, j] = batch_preds[sample]
            
        torch.save(out, latent_path + encModel + "/" + vector + "_cc3m_library_preds.pt")
        print("Average Pearson Across Samples: ", (average_pearson / x.shape[0]) ) 
        return out