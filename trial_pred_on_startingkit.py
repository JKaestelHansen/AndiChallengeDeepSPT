# %%
import os
import numpy as np
import pandas as pd
from global_config_AnDi import globals
import torch
from DeepSPT_AnDi import *

from sklearn.mixture import GaussianMixture
from andi_datasets.datasets_phenom import datasets_phenom


def fit_gaussians(data, n_states):
    """
    Fits Gaussian distributions to the data and returns the means and standard deviations of each state.
    
    Parameters:
    - data: array-like, shape (n_samples,)
      The data to be fitted.
    - n_states: int
      The number of Gaussian distributions (states) to fit.
    
    Returns:
    - means: array, shape (n_states,)
      The means of the fitted Gaussian distributions.
    - stds: array, shape (n_states,)
      The standard deviations of the fitted Gaussian distributions.
    """
    # Reshape data for the model
    data = np.array(data).reshape(-1, 1)
    
    # Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_states, random_state=0)
    gmm.fit(data)
    
    # Extract means and standard deviations
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()

    # relative weights
    rw = gmm.weights_
    
    return means, stds, rw

def merge_segments(changepoints):
    # Convert changepoints to integers
    changepoints = [int(point) for point in changepoints]
    
    segments = []
    current_start = 0
    
    # Calculate initial segments
    for point in changepoints:
        segments.append((current_start, point))
        current_start = point
    
    # Merge segments
    merged_segments = []
    i = 0
    while i < len(segments):
        start, end = segments[i]
        segment_length = end - start
        
        if segment_length < 20:
            # Try to merge with the left segment if it's the first segment
            if i > 0 and (merged_segments[-1][1] - merged_segments[-1][0]) >= 20:
                merged_segments[-1] = (merged_segments[-1][0], end)
            else:
                # Merge with right segments until the combined length is >= 20 or end of list
                while i < len(segments) - 1 and (end - start) < 20:
                    i += 1
                    end = segments[i][1]
                merged_segments.append((start, end))
        else:
            merged_segments.append((start, end))
        
        i += 1

    # Create the array with unique integers for each segment
    segment_array = []
    for idx, (start, end) in enumerate(merged_segments):
        segment_array.extend([idx + 1] * (end - start))

    return merged_segments, segment_array

def prep_csv_tracks(df, xname='x', yname='y', timename='A', identifiername='particle', center=False):
    df_by_particle = dict(tuple(df.groupby(identifiername)))
    X = [np.vstack(val[[xname, yname]].values).astype(float) for val in df_by_particle.values()]
    T = [np.vstack(val[timename].values).astype(float) for val in df_by_particle.values()]
    if center:
        X = [x-x[0] for x in X]
    return X, T


def create_tracks(df, identifiername='TRACK_ID', timename='FRAME', 
                  xname='POSITION_X', yname='POSITION_Y', center=False):
    X = df.sort_values(by=[identifiername, timename]).reset_index(drop=True)
    X, T = prep_csv_tracks(X, xname=xname, yname=yname, 
                    identifiername=identifiername,
                    timename=timename,
                    center=center)
    return X, T

# Define the path to the public data
public_data_path = '_Data/public_data_validation_v1-4/'
public_data_path = '_Data/starting_kit/'

# We only to track 2 in this example
track = 2

# The results go in the same folders generated above
path_results = 'results/'
path_track = path_results + f'track_{track}/'

# Load data 
trajs_fov_ = []
traj_labs_fov_ = []
for root, dirs, files in os.walk(public_data_path):
    for f in files:
        if 'trajs_fov_' in f:
            trajs_fov_.append(root+'/'+f)
            
        if 'traj_labs_fov_' in f:
            traj_labs_fov_.append(root+'/'+f)
import natsort
trajs_fov_ = natsort.natsorted(trajs_fov_)
traj_labs_fov_ = natsort.natsorted(traj_labs_fov_)

print(trajs_fov_)
print(traj_labs_fov_)

X = []
info_list = []
for tf, tl in zip(trajs_fov_, traj_labs_fov_):
    assert tf.split('traj')[0] == tl.split('traj')[0]
    assert tf.split('fov')[1].split('.')[0] == tl.split('fov')[1].split('.')[0]

    df = pd.read_csv(tf)
    tracks_centered, timepoints = create_tracks(df, identifiername='traj_idx', timename='frame', 
        xname='x', yname='y', center=True)
    for t in tracks_centered:
        X.append(t)

    with open(tl) as f:
        for line in f:
            # separate by commas and make list
            list_line = np.array(line.strip().split(',')).astype(float)
            info_list.append(list_line)
            
L = [len(x) for x in info_list]


print('make into dummies')
temporal_states_y = [np.zeros(len(x)) for x in X]
temporal_states_a = [np.zeros(len(x)) for x in X]
temporal_states_D = [np.zeros(len(x)) for x in X]
actual_CPs = [np.zeros(len(x)) for x in X]


print('add the correct path to trained models')
# define dataset and method that model was trained on to find the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
methods = ['XYZ_SL_DP']
# find the model
dir_name = ''
modelpath = 'Unet_results/mlruns/'
modeldir = '13'
modelname = '_UNETmodel' # '_DeepSPTAnDimodel'
modelname = '_DeepSPTAnDimodel'
use_mlflow = False
if use_mlflow:
    import mlflow
    mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("Unet_results", "mlruns")))
    best_models_sorted = find_models_for(datasets, methods)
else:
    # not sorted tho
    path = '/nfs/datasync4/jacobkh/SPT/AndiChallenge/mlruns/{}'.format(modeldir)
    print(path)
    best_models_sorted = find_models_for_from_path(path, modelname=modelname)
    print(best_models_sorted)

idx = []
valloss = []
for i, best_model_path in enumerate(best_models_sorted):
    if 'CV_' in best_model_path:
        continue
    try:
        bva = torch.load(best_model_path, map_location=torch.device(device))['best_val_acc']
        idx.append(i)
        valloss.append(bva)
    except:
        print('could not find best val acc')
        continue


min_idx_vallloss = np.argmin(valloss)
print(valloss, idx, min_idx_vallloss)

best_model_path = best_models_sorted[min_idx_vallloss]
print(best_model_path)

features = globals.features 
X = prep_tracks(X)
X = add_features(X, features)

print(len(X))
print(len(temporal_states_y))
print(len(temporal_states_a))
print(len(temporal_states_D))
print(len(actual_CPs))

print(len(X[0]))
print(len(temporal_states_y[0]))
print(len(temporal_states_a[0]))
print(len(temporal_states_D[0]))
print(len(actual_CPs[0]))

print(X[0][:5], X[1][:5])

# %%

# model params
n_features = 4
n_classes = 6 # 4 states + D + alpha

init_channels = globals.init_channels # number of initial channels in model - these will multiply with channel_multiplier during encoding
channel_multiplier = globals.channel_multiplier # channel multiplier size
dil_rate = globals.dil_rate # dilation rate of U-net encoder
depth = globals.depth # depth of U-net

kernelsize = globals.kernelsize # kernel size of encoder and decoder!
outconv_kernel = globals.outconv_kernel # kernel size in output block of model

pools = globals.pools # size of the pooling operation
pooling = globals.pooling # pooling type max / avg

enc_conv_nlayers = globals.enc_conv_nlayers # number of layers in encoder block of model
dec_conv_nlayers = globals.dec_conv_nlayers# number of layers in decoder block of model
bottom_conv_nlayers = globals.bottom_conv_nlayers # number of layers in bottom block of model
out_nlayers = globals.out_nlayers # number of layers in output block of model

batchnorm = globals.batchnorm # bool of batchnorm
batchnormfirst = globals.batchnormfirst # batchnorm before relu

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_padtoken = globals.X_padtoken # pad token for U-net
y_padtoken = globals.y_padtoken # padtoken for U-net
val_size, test_size = globals.val_size, globals.test_size # validation and test set sizes
dim = 2
batch_size = globals.batch_size
threshold = 0.2

model = hypoptUNet(n_features = n_features,
                    init_channels = init_channels,
                    n_classes = n_classes,
                    depth = depth,
                    enc_kernel = kernelsize,
                    dec_kernel = kernelsize,
                    outconv_kernel = outconv_kernel,
                    dil_rate = dil_rate,
                    pools = pools,
                    pooling = pooling,
                    enc_conv_nlayers = enc_conv_nlayers,
                    bottom_conv_nlayers = bottom_conv_nlayers,
                    dec_conv_nlayers = dec_conv_nlayers,
                    out_nlayers = out_nlayers,
                    X_padtoken = X_padtoken,
                    y_padtoken = y_padtoken,
                    channel_multiplier = channel_multiplier,
                    batchnorm = batchnorm,
                    batchnormfirst = batchnormfirst,
                    device = device,
                    threshold = threshold)

model_state_dict = torch.load(best_model_path, map_location=torch.device(device))['model_state_dict']
model.load_state_dict(model_state_dict)

min_max_length = np.max([len(x) for x in X])
print(min_max_length)
D_test = CustomDataset(X, 
                      temporal_states_y,
                      temporal_states_a,
                      temporal_states_D, 
                      actual_CPs,
                      X_padtoken=X_padtoken, y_padtoken=y_padtoken, 
                      device=device,
                      min_max_length = min_max_length)
test_loader = DataLoader(D_test, batch_size = batch_size, shuffle=False)
masked_argmaxs, masked_a_preds, masked_D_preds, masked_ys, andi_pred_formats, CPS_p_lists, CPS_a_lists, CPS_D_lists = model.predict(test_loader)

# %%
len(andi_pred_formats), andi_pred_formats[1]

# postprocess the predictions
def merge_segments(changepoints):
    # Convert changepoints to integers
    changepoints = [int(point) for point in changepoints]
    
    # Calculate initial segments
    segments = []
    current_start = 0
    for point in changepoints:
        segments.append((current_start, point))
        current_start = point
    
    # Merge segments
    merged_segments = []
    i = 0
    while i < len(segments):
        start, end = segments[i]
        segment_length = end - start
        
        # Merge with next segments if the current segment is too short
        while segment_length < 20 and i < len(segments) - 1:
            i += 1
            end = segments[i][1]
            segment_length = end - start
        
        merged_segments.append((start, end))
        i += 1
    
    # Ensure the last segment is long enough by merging it with the previous one if necessary
    if len(merged_segments) > 1 and (merged_segments[-1][1] - merged_segments[-1][0]) < 20:
        merged_segments[-2] = (merged_segments[-2][0], merged_segments[-1][1])
        merged_segments.pop()
    
    # Create the array with unique integers for each segment
    segment_array = []
    for idx, (start, end) in enumerate(merged_segments):
        segment_array.extend([idx + 1] * (end - start))

    # Return only the endpoints for merged segments
    merged_endpoints = [end for start, end in merged_segments]

    return merged_segments, segment_array, merged_endpoints


CPs_ensemble_exp = []
andi_pred_formats_postprocessed_all = []
for i in range(len(andi_pred_formats)):
    pre_CPs = andi_pred_formats[i][3::4]
    merged_segments, segment_array, postprocess_CP = merge_segments(pre_CPs)
    postprocess_CP = [0] + postprocess_CP
    assert pre_CPs[-1] == postprocess_CP[-1]

    masked_a_preds_i = masked_a_preds[i].copy()
    masked_D_preds_i = masked_D_preds[i].copy()
    masked_argmaxs_i = masked_argmaxs[i].copy()

    andi_pred_formats_postprocessed = []
    for cpi in range(len(postprocess_CP)-1):

        start = postprocess_CP[cpi]
        end = postprocess_CP[cpi+1]

        D_segment = masked_D_preds_i[0][start:end]
        alpha_segment = masked_a_preds_i[0][start:end]
        state_segment = masked_argmaxs_i[start:end]

        andi_pred_formats_postprocessed.append(D_segment.mean())
        andi_pred_formats_postprocessed.append(alpha_segment.mean())
        # mode of state_segment
        mode_numpy = np.bincount(state_segment).argmax()
        andi_pred_formats_postprocessed.append(mode_numpy)
        
        if cpi <= len(postprocess_CP)-2:
            andi_pred_formats_postprocessed.append(postprocess_CP[1:][cpi])
        else:
            andi_pred_formats_postprocessed.append(postprocess_CP[1:][-1])
    andi_pred_formats_postprocessed_all.append(andi_pred_formats_postprocessed)


# %%
# create submission file
N_EXP = 5
N_FOVS = 20
count = 0
alpha_ensemble_avg = []
K_ensemble_avg = []
alpha_ensemble_std = []
K_ensemble_std = []
states_ensemble = []
CPs_ensemble = []

andi_format_to_report = andi_pred_formats
andi_format_to_report = andi_pred_formats_postprocessed_all

for exp in range(N_EXP):
    
    path_exp = public_data_path + 'res/' + f'track_2/exp_{exp}/'
    # remove the folder if it exists
    if os.path.exists(path_exp):
        os.system(f'rm -r {path_exp}')
        os.makedirs(path_exp)
    
    alphas = []
    Ks = []
    states = []
    CPs = []
    for fov in range(N_FOVS):
        
        # We read the corresponding csv file from the public data and extract the indices of the trajectories:
        df = pd.read_csv(public_data_path+'ref/'+f'track_2/exp_{exp}/trajs_fov_{fov}.csv')
        traj_idx = df.traj_idx.unique()
        
        submission_file = path_exp + f'fov_{fov}.txt'
        with open(submission_file, 'a') as f:
            for idx in traj_idx:
                
                # Get the lenght of the trajectory
                length_traj = df[df.traj_idx == traj_idx[0]].shape[0]

                prediction_traj = [idx.astype(int)] + andi_format_to_report[count]
                formatted_numbers = ','.join(map(str, prediction_traj))
                f.write(formatted_numbers + '\n')

                Ks.append(andi_format_to_report[count][0::4])
                alphas.append(andi_format_to_report[count][1::4])
                states.append(andi_format_to_report[count][2::4])
                CPs.append(andi_format_to_report[count][3::4])

                count += 1

    alpha_ensemble_avg.append(alphas)
    K_ensemble_avg.append(Ks)
    states_ensemble.append(states)
    CPs_ensemble.append(CPs)

# %%

for exp in range(N_EXP):
    path_exp = public_data_path + 'res/' + f'track_2/exp_{exp}/'
    if not os.path.exists(path_exp):
        os.makedirs(path_exp)

    file_name = path_exp + 'ensemble_labels.txt'
    print(file_name)

    alpha_ensemble_avg_exp = alpha_ensemble_avg[exp]
    K_ensemble_avg_exp = K_ensemble_avg[exp]
    states_ensemble_exp = states_ensemble[exp]
    CPs_ensemble_exp = CPs_ensemble[exp]

    num_states = np.max([len(c) for c in CPs_ensemble_exp])-1
    num_states = 1 if num_states == 0 else num_states
    a_means, a_stds, a_rw = fit_gaussians(np.hstack(alpha_ensemble_avg_exp), num_states)
    D_means, D_stds, D_rw = fit_gaussians(np.hstack(K_ensemble_avg_exp), num_states)
    
    data = np.row_stack((a_means, a_stds, D_means, D_stds, D_rw))

    with open(file_name, 'a') as f:
        # wipe any existing content
        f.seek(0)
        # make sure f is empty
        f.truncate()
  
        # Save the model (random) and the number of states (2 in this case)
        model_name = np.random.choice(datasets_phenom().avail_models_name, size = 1)[0]
        f.write(f'model: {model_name}; num_state: {num_states} \n')

        # Create some dummy data for 2 states. This means 2 columns
        # and 5 rows
        data = np.row_stack((a_means, a_stds, D_means, D_stds, D_rw))

        # Save the data in the corresponding ensemble file
        np.savetxt(f, data, delimiter = ';')
        f.close()

# load file of ensemble labels to check if it worked
file_name = path_exp + 'ensemble_labels.txt'
with open(file_name, 'r') as f:
    lines = f.readlines()
    print(lines)



# %%
i = np.random.randint(0, len(andi_pred_formats))
print(i)
print(info_list[i][1::4], andi_pred_formats[i][0::4], andi_pred_formats_postprocessed_all[i][0::4])
print(info_list[i][2::4], andi_pred_formats[i][1::4], andi_pred_formats_postprocessed_all[i][1::4])
print(info_list[i][3::4], andi_pred_formats[i][2::4], andi_pred_formats_postprocessed_all[i][2::4])
print(info_list[i][4::4], andi_pred_formats[i][3::4], andi_pred_formats_postprocessed_all[i][3::4])

# %%


# %%
i = np.random.randint(0, len(andi_pred_formats))
print(i)
CPS_p_lists, CPS_a_lists, CPS_D_lists
print(info_list[i][4::4], CPS_p_lists[i], CPS_a_lists[i], CPS_D_lists[i], andi_pred_formats_postprocessed_all[i][3::4])

# %%

print(public_data_path)
print(os.listdir(public_data_path))

from andi_datasets.utils_challenge import codalab_scoring

codalab_scoring(INPUT_DIR = public_data_path,
                OUTPUT_DIR = '.')

# print scores.txt in a nice way
with open('scores.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
# %%
