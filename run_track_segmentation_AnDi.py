# %%
from Unet_mlflow_utils.convenience_functions import find_experiments_by_tags,\
                                               make_experiment_name_from_tags
from global_config_AnDi import globals
from torch.utils.data import DataLoader
from DeepSPT_AnDi import *
import numpy as np
import pandas as pd
import datetime
import pickle
import mlflow
import torch
import random
import os

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

#**********************Initiate variables**********************

# global config variables
globals._parse({})

# get consistent result
seed = globals.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# saving
features = globals.features 
method = "_".join(features,)
datapath = globals.datapath
data_naming = 'AnDi_model_'+datapath.replace('/','_')
best_model_save_name = timestamper() + '_DeepSPTAnDimodel.torch'
print(data_naming)

# %%
# training
lr = globals.lr
epochs = globals.epochs
batch_size = globals.batch_size
shuffle = globals.shuffle

# model variables
modelname = 'DeepSPTAnDimodel'

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

device = globals.device # device of model
X_padtoken = globals.X_padtoken # pad token for U-net
y_padtoken = globals.y_padtoken # padtoken for U-net
val_size, test_size = globals.val_size, globals.test_size # validation and test set sizes
dim = 2
print(device)

#**********************Model and Data**********************
# Load data 
trajs_fov_ = []
traj_labs_fov_ = []
for root, dirs, files in os.walk(datapath):
    for f in files:
        if 'trajs_fov_' in f:
            trajs_fov_.append(root+'/'+f)
            
        if 'traj_labs_fov_' in f:
            traj_labs_fov_.append(root+'/'+f)
trajs_fov_ = np.sort(trajs_fov_)
traj_labs_fov_ = np.sort(traj_labs_fov_)

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

temporal_states_y = []
temporal_states_a = []
temporal_states_D = []
actual_CPs = []
for info in info_list:
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
    
    actual_CPs.append(info[4::4])

Ly = [len(y) for y in temporal_states_y]
La = [len(a) for a in temporal_states_a]
Lt = [len(t) for t in X]

for i in range(len(Ly)):
    assert Ly[i] == Lt[i], f'Ly: {Ly[i]}, Lt: {Lt[i]}'

# %%

print('Data info')
print(len(X), len(temporal_states_y), len(temporal_states_a), len(temporal_states_D), len(actual_CPs))
print(np.unique(np.hstack(temporal_states_y), return_counts=True))

min_max_length = np.max([len(x) for x in X])
print('min_max_length', min_max_length)

# %%
# Add features
X = prep_tracks(X)
X = add_features(X, features)
n_features = X[0].shape[1] # number of input features
n_classes = len(np.unique(np.hstack(temporal_states_y))) + 2 # +1 for alpha channel, +1 for D channel

print('n_features, n_classes', n_features, n_classes)
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
                    device = device)

optim_choice = globals.optim_choice
optimizer = optim_choice(model.parameters(), lr=lr)

#**********************Run Experiment**********************
# Auto-saving
tags = {'DATASET': data_naming, 'METHOD': method, 'MODEL': modelname}
exp = find_experiments_by_tags(tags)
if len(exp) == 0:
    experiment_name = make_experiment_name_from_tags(tags)
    e_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    for t in tags.keys():
        mlflow.tracking.MlflowClient().set_experiment_tag(e_id, t, tags[t])
elif len(exp) == 1:
    mlflow.set_experiment(exp[0].name)
    experiment_id = exp[0].experiment_id
else:
    raise RuntimeError("There should be at most one experiment for a given tag combination!")

mlflow.start_run()
for i, seed in enumerate(globals.seeds):
    # Pytorch prep and Split
    D = CustomDataset(X, 
                      temporal_states_y,
                      temporal_states_a,
                      temporal_states_D, 
                      actual_CPs,
                      X_padtoken=X_padtoken, y_padtoken=y_padtoken, 
                      device=device)
    
    D_train, D_val, D_test = datasplitter(D, val_size, test_size, seed)
    
    # Dataloaders
    train_loader = DataLoader(D_train, batch_size = batch_size, shuffle = shuffle)
    val_loader = DataLoader(D_val, batch_size = batch_size)

    path ='mlruns/'+str(experiment_id)+'/'+str(mlflow.active_run().info.run_id)+'/'
    cv_indices_path = os.path.join(path, 'CV_indices')
    if not os.path.exists(cv_indices_path): 
        os.makedirs(cv_indices_path)
    torch.save(D_train.indices, cv_indices_path+'/CVfold'+str(i)+'_D_train_idx.pt')
    torch.save(D_val.indices, cv_indices_path+'/CVfold'+str(i)+'_D_val_idx.pt')
    torch.save(D_test.indices, cv_indices_path+'/CVfold'+str(i)+'_D_test_idx.pt')

    best_val_acc = np.inf
    for epoch in range(1, epochs + 1):
        starttime = datetime.datetime.now()
        train_loss, train_acc, train_F1, train_loss_states, train_loss_alpha, train_loss_D, train_loss_jaccard, train_loss_TP_CP = train_epoch(model, optimizer, train_loader, device, dim)

        val_loss, val_acc, val_F1, val_loss_states, val_loss_alpha, val_loss_D, val_loss_jaccard, val_loss_TP_CP = validate(model, optimizer, val_loader, device, dim)

        metric_to_optimize = val_loss

        improved = best_val_acc > metric_to_optimize
        if improved:
            print(metric_to_optimize, best_val_acc)
            best_model = model
            best_val_acc = metric_to_optimize

            print('Saving best model')
            torch.save({'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss': train_loss,
                'best_val_acc': best_val_acc
                }, path+best_model_save_name)

            with open(path+'best_model_results.txt', 'a') as f:
                print(path+'best_model_results.txt')
                f.write('CVfold: '+str(i+1)+'/'+str(len(globals.seeds))+
                        ' Epoch: '+str(epoch)+
                        ' Train loss: '+str(np.round(train_loss,3))+
                        ' Train Acc: '+str(np.round(train_acc,3))+
                        ' Train F1: '+str(np.round(train_F1,3))+
                        ' Train loss states: '+str(np.round(train_loss_states,3))+
                        ' Train loss alpha: '+str(np.round(train_loss_alpha,3))+
                        ' Train loss D: '+str(np.round(train_loss_D,3))+
                        ' Train loss jaccard: '+str(np.round(train_loss_jaccard,3))+
                        ' Train loss TP_CP: '+str(np.round(train_loss_TP_CP,3))+
                        ' Val loss: '+str(np.round(val_loss,3))+
                        ' Val Acc: '+str(np.round(val_acc,3))+
                        ' Val F1: '+str(np.round(val_F1,3))+
                        ' Val loss states: '+str(np.round(val_loss_states,3))+
                        ' Val loss alpha: '+str(np.round(val_loss_alpha,3))+
                        ' Val loss D: '+str(np.round(val_loss_D,3))+
                        ' Val loss jaccard: '+str(np.round(val_loss_jaccard,3))+
                        ' Val loss TP_CP: '+str(np.round(val_loss_TP_CP,3))+
                        ' Best Val Acc: '+str(np.round(best_val_acc,3))+
                        ' time/epoch: '+str(datetime.datetime.now()-starttime)+'\n')
            f.close()
        mlflow.log_metric('CVfold'+str(i)+'_TRAIN_LOSS', train_loss)
        mlflow.log_metric('CVfold'+str(i)+'_TRAIN_ACC', train_acc)
        mlflow.log_metric('CVfold'+str(i)+'_VAL_LOSS', val_loss)
        mlflow.log_metric('CVfold'+str(i)+'_VAL_ACC', val_acc)

        print()
        print('CVfold:', str(i+1)+'/'+str(len(globals.seeds)),
            'Epoch:', epoch, 
            'Train loss:', np.round(train_loss,3), 
            'Train Acc:', np.round(train_acc,3), 
            'Train F1', np.round(train_F1,3),
            'Train loss states', np.round(train_loss_states,3),
            'Train loss alpha', np.round(train_loss_alpha,3),
            'Train loss D', np.round(train_loss_D,3),
            'Train loss jaccard', np.round(train_loss_jaccard,3),
            'Train loss TP_CP', np.round(train_loss_TP_CP,3),
            'Val loss:', np.round(val_loss,3), 
            'Val Acc:', np.round(val_acc,3), 
            'Val F1', np.round(val_F1,3),
            'Val loss states', np.round(val_loss_states,3),
            'Val loss alpha', np.round(val_loss_alpha,3),
            'Val loss D', np.round(val_loss_D,3),
            'Val loss jaccard', np.round(val_loss_jaccard,3),
            'Val loss TP_CP', np.round(val_loss_TP_CP,3),
            'Best Val Acc:', np.round(best_val_acc,3), 
            'time/epoch:', datetime.datetime.now()-starttime)
        # save to text file

    best_model.eval()
    test_loader = DataLoader(D_test, batch_size = batch_size)
    masked_argmaxs, masked_a_preds, masked_D_preds, masked_ys, andi_pred_formats, CPS = best_model.predict(test_loader)
    testpred_indices_path = os.path.join(path, 'TestPredictions')
    print('Saving best model')
    torch.save({'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss': train_loss,
                'best_val_acc': best_val_acc
                }, path+best_model_save_name)
    if not os.path.exists(testpred_indices_path): 
        os.makedirs(testpred_indices_path)

mlflow.log_param('Data set', data_naming)
mlflow.log_param('Data features', method)
mlflow.log_param('Data path', datapath)

mlflow.log_metric('Learning rate', lr)
mlflow.log_param('Trained for Epochs', epochs)

mlflow.log_metric('Number of input features', n_features)
mlflow.log_metric('Number of classes', n_classes)

mlflow.log_metric('Initial Number of Channels', init_channels)
mlflow.log_metric('Channel multiplier', channel_multiplier)
mlflow.log_metric('Depth of model', depth)
mlflow.log_metric('Dilation rate', dil_rate)

mlflow.log_metric('Number encoder layers', enc_conv_nlayers)
mlflow.log_metric('Number bottom layers', bottom_conv_nlayers)
mlflow.log_metric('Number decoder layers', dec_conv_nlayers)
mlflow.log_metric('Number output layers', out_nlayers)

mlflow.log_param('Batchnorm', batchnorm)
mlflow.log_param('BN before ReLU', batchnormfirst)

mlflow.log_metric('Unet kernel size', kernelsize)
mlflow.log_metric('Output block kernel size', outconv_kernel)

mlflow.log_metric('Val size', val_size)
mlflow.log_metric('Test size', test_size)

mlflow.log_param('Pooling type', pooling)
for i in range(len(pools)):
    mlflow.log_metric('pools'+str(i), pools[i])

mlflow.log_metric('Cross Validation Folds', len(globals.seeds))
for i, seed in enumerate(globals.seeds):
    mlflow.log_metric('Seed'+str(i), seed)
mlflow.log_metric('Validation size', val_size)
mlflow.log_metric('Test size', test_size)
mlflow.log_param('Dataloader shuffle', shuffle)
mlflow.log_metric('Batch size', batch_size)
mlflow.log_param('X padtoken', X_padtoken)
mlflow.log_param('y padtoken', y_padtoken)

mlflow.end_run()

# %%