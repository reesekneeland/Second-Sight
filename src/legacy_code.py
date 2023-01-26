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