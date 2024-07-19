# %%
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2, _defaults_andi2
import numpy as np
from scipy.stats import loguniform

MODELS = np.arange(5)
NUM_FOVS = 300
PATH = '_Data/training14' # Chose your path!

dics = []
for m in MODELS:   
    dic = _get_dic_andi2(m+1)

    # Fix length and number of trajectories 
    dic['T'] = 500 
    dic['N'] = 300

    # # Add repeated fovs for the experiment
    for _ in range(NUM_FOVS):
        dic['T'] = 500 
        dic['N'] = 300
        if m == 0:  
            dic['Ds'] = np.array([loguniform.rvs(0.5, 2, size=1)[0], loguniform.rvs(10**-3, 1)])
            dic['alphas'] = np.array([np.random.uniform(1, 2, size=1)[0], loguniform.rvs(10**-3, 1)])
            
        if m == 1:
            if np.random.rand() < 0.5:
                # 2-state model with different alphas       
                D = loguniform.rvs(0.5, 2, size=1)[0] 
                A = np.random.uniform(1, 2, size=1)[0]        
                dic['Ds'] = np.array([[D, loguniform.rvs(10**-3, 10**1)],
                                      [D*np.random.uniform(0.05,0.9), loguniform.rvs(10**-3, 1)]])
                dic['alphas'] = np.array([[A, loguniform.rvs(10**-3, 10**1)],
                                         [A*np.random.uniform(0.05,0.9), loguniform.rvs(10**-3, 1)]])

            else:
                # 3-state model with different alphas  
                D = loguniform.rvs(0.5, 2, size=1)[0]    
                A = np.random.uniform(1, 2, size=1)[0]  
                
                dic['Ds'] = np.array([[D, loguniform.rvs(10**-3, 10**1)],
                                      [D*np.random.uniform(0.05,0.9)     , loguniform.rvs(10**-3, 1)],
                                      [D*np.random.uniform(0.05,0.9)     , loguniform.rvs(10**-3, 1)]])
                dic['alphas'] = np.array([[A, loguniform.rvs(10**-3, 10**1)],
                                         [A*np.random.uniform(0.05,0.9)     , loguniform.rvs(10**-3, 1)],
                                         [A*np.random.uniform(0.05,0.9)     , loguniform.rvs(10**-3, 1)]])
            
        #### IMMOBILE TRAPS ####
        if m == 2:
            dic['Ds'] = np.array([loguniform.rvs(10**-3, 10**1, size=1)[0], loguniform.rvs(10**-3, 1)])
            dic['alphas'] = np.array([np.random.uniform(1, 2, size=1)[0], loguniform.rvs(10**-3, 1)])
                
        
        #### DIMERIZATION ####
        if m == 3:
            D = loguniform.rvs(10**-3, 10**1, size=1)[0]
            dic['Ds'] = np.array([[D, loguniform.rvs(10**-3, 2*10**-1)],
                                  [D*loguniform.rvs(10**-2, 5*10**-1, size=1)[0], loguniform.rvs(10**-3, 1)]])
            
        #### CONFINEMENT ####
        if m == 4:
            A = np.random.uniform(1, 2, size=1)[0]
            D = loguniform.rvs(10**-3, 10**1, size=1)[0]
            dic['Ds'] = np.array([[D, loguniform.rvs(10**-3, 1)],
                                  [loguniform.rvs(10**-3, 10**-2, size=1)[0], loguniform.rvs(10**-3, 1)]])
            
            dic['alphas'] = np.array([[A, loguniform.rvs(10**-3, 1)],
                                  [A*loguniform.rvs(10**-2, 5*10**-1, size=1)[0], loguniform.rvs(10**-3, 1)]])

        dics.append(dic)

# %%

dfs_traj, labs_traj, labs_ens = challenge_phenom_dataset(save_data = True, # If to save the files
                                                         dics = dics, # Dictionaries with the info of each experiment (and FOV in this case)
                                                         path = PATH, # Parent folder where to save all data
                                                         return_timestep_labs = True, get_video = False, 
                                                         num_fovs = 1, # Number of FOVs                                                                
                                                         num_vip=10, # Number of VIP particles
                                                         files_reorg = True, # We reorganize the folders for challenge structure
                                                         path_reorg = 'ref/', # Folder inside PATH where to reorganized
                                                         save_labels_reorg = True, # The labels for the two tasks will also be saved in the reorganization                                                                 
                                                         delete_raw = True # If deleting the original raw dataset
                                                                )

# %%
labs_traj_flat = [item for sublist in labs_traj for item in sublist]
print(len(labs_traj_flat))

temporal_states_y = []
temporal_states_a = []
temporal_states_D = []
for info in labs_traj_flat:
    length_track = int(info[-1])
    temp_y = np.zeros(length_track)
    temp_a = np.zeros(length_track)
    temp_D = np.zeros(length_track)
    if len(info) == 5: # no changepoint
        # gør alpha også til temporal og bare gange den channel med 2
        temp_y[:] = info[3]
        temp_a[:] = info[2]
        temp_D[:] = info[1]
        temporal_states_y.append(temp_y)
        temporal_states_a.append(temp_a)
        temporal_states_D.append(temp_D)
    else:

        Ds = info[1::4]
        As = info[2::4]
        states = info[3::4]
        CPs = [0]+list(info[4::4])
    
        for i in range(len(CPs)-1):
            start = int(CPs[i])
            end = int(CPs[i+1])
            temp_y[start:end] = states[i]
            temp_a[start:end] = As[i]
            temp_D[start:end] = Ds[i]

        temporal_states_y.append(temp_y)
        temporal_states_a.append(temp_a)
        temporal_states_D.append(temp_D)

As = np.hstack(temporal_states_a)
Ds = np.hstack(temporal_states_D)
ys = np.hstack(temporal_states_y)

print(np.unique(ys, return_counts=True))
print(np.mean(As), np.std(As), np.min(As), np.max(As))
print(np.mean(Ds), np.std(Ds), np.min(Ds), np.max(Ds))
# %%
