# %%
from Unet_mlflow_utils.convenience_functions import find_experiments_by_tags,\
                                               make_experiment_name_from_tags
from global_config import globals
from torch.utils.data import DataLoader
from Unet import *
import numpy as np
import datetime
import pickle
import mlflow
import torch
import random

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
filename_X = globals.filename_X
filename_y = globals.filename_y
data_naming = filename_X[filename_X.find('SimDiff_'):filename_X.find('_X.pkl')]
best_model_save_name = timestamper() + '_UNETmodel.torch'
print(filename_X)
# %%
# training
lr = globals.lr
epochs = globals.epochs
batch_size = globals.batch_size
shuffle = globals.shuffle

# model variables
modelname = 'Unet'

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

#**********************Model and Data**********************
# Load data 
X = np.array(pickle.load(open(datapath+filename_X, 'rb')), dtype=object)
y = np.array(pickle.load(open(datapath+filename_y, 'rb')), dtype=object)

# Add features
X = prep_tracks(X)
X = add_features(X, features)
n_features = X[0].shape[1] # number of input features
n_classes = len(np.unique(np.hstack(y)))

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
    D = CustomDataset(X, y, X_padtoken=X_padtoken, y_padtoken=y_padtoken, device=device)
    D_train, D_val, D_test = datasplitter(D, val_size, test_size, seed)
    
    # Dataloaders
    train_loader = DataLoader(D_train, batch_size = batch_size, shuffle = shuffle)
    val_loader = DataLoader(D_val, batch_size = batch_size)

    path ='Unet_results/mlruns/'+str(experiment_id)+'/'+str(mlflow.active_run().info.run_id)+'/'
    cv_indices_path = os.path.join(path, 'CV_indices')
    if not os.path.exists(cv_indices_path): 
        os.makedirs(cv_indices_path)
    torch.save(D_train.indices, cv_indices_path+'/CVfold'+str(i)+'_D_train_idx.pt')
    torch.save(D_val.indices, cv_indices_path+'/CVfold'+str(i)+'_D_val_idx.pt')
    torch.save(D_test.indices, cv_indices_path+'/CVfold'+str(i)+'_D_test_idx.pt')

    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        starttime = datetime.datetime.now()
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device)

        val_loss, val_acc = validate(model, optimizer, val_loader, device)

        improved = best_val_acc < val_acc
        if improved:
            best_model = model
            best_val_acc = val_acc
                    
        mlflow.log_metric('CVfold'+str(i)+'_TRAIN_LOSS', train_loss)
        mlflow.log_metric('CVfold'+str(i)+'_TRAIN_ACC', train_acc)
        mlflow.log_metric('CVfold'+str(i)+'_VAL_LOSS', val_loss)
        mlflow.log_metric('CVfold'+str(i)+'_VAL_ACC', val_acc)

        print('CVfold:', str(i+1)+'/'+str(len(globals.seeds)),
            'Epoch:', epoch, 
            'Train loss:', train_loss, 
            'Train Acc:', train_acc, 
            'Val loss:', val_loss, 
            'Val Acc:', val_acc, 
            'Best Val Acc:', best_val_acc, 
            'time/epoch:', datetime.datetime.now()-starttime)

    best_model.eval()
    test_loader = DataLoader(D_test, batch_size = batch_size)
    masked_pred, masked_score, masked_y = best_model.predict(test_loader)
    testpred_indices_path = os.path.join(path, 'TestPredictions')
    torch.save({'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss': train_loss,
                }, path+'CV_fold'+str(i)+'_'+best_model_save_name)
    if not os.path.exists(testpred_indices_path): 
        os.makedirs(testpred_indices_path)
    torch.save({'Scores': masked_score,
                'Preds': masked_pred,
                'Labels': masked_y
                }, testpred_indices_path+'/CVfold'+str(i)+'_TestPredictionsDict.pt')

mlflow.log_param('Data set', data_naming)
mlflow.log_param('Data features', method)
mlflow.log_param('Data path', datapath)
mlflow.log_param('Filename', filename_X)

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
