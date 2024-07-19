# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


data_path = '_Data/starting_kit/track_2' # make sure the folder has this name or change it

def load_and_transform_Andi_data(data_path):
    trajs_fov_ = []
    traj_labs_fov_ = []
    for root, dirs, files in os.walk(data_path):
        print(root)
        for f in files:
            if 'trajs_fov_' in f:
                trajs_fov_.append(root+'/'+f)
                
            if 'traj_labs_fov_' in f:
                traj_labs_fov_.append(root+'/'+f)
    trajs_fov_ = np.sort(trajs_fov_)
    traj_labs_fov_ = np.sort(traj_labs_fov_)

    tracks_all = []
    info_list = []
    for tf, tl in zip(trajs_fov_, traj_labs_fov_):
        assert tf.split('traj')[0] == tl.split('traj')[0]
        assert tf.split('fov')[1].split('.')[0] == tl.split('fov')[1].split('.')[0]

        df = pd.read_csv(tf)
        tracks_centered, timepoints = create_tracks(df, identifiername='traj_idx', timename='frame', 
            xname='x', yname='y', center=True)
        for t in tracks_centered:
            tracks_all.append(t)

        with open(tl) as f:
            for line in f:
                # separate by commas and make list
                list_line = np.array(line.strip().split(',')).astype(float)
                info_list.append(list_line)

    L = [len(x) for x in info_list]

    temporal_states_y = []
    temporal_states_a = []
    temporal_states_D = []
    for info in info_list:
        if np.sum(1.==info[1:]):
            print(info)
        length_track = int(info[-1])
        temp_y = np.zeros(length_track)
        temp_a = np.zeros(length_track)
        if len(info) == 5: # no changepoint
            # gør alpha også til temporal og bare gange den channel med 2
            temp_y[:] = info[3]
            temp_a[:] = info[2]
            temporal_states_y.append(temp_y)
            temporal_states_a.append(temp_a)
            temporal_states_D.append([info[1]])
        else:

            Ds = info[1::4]
            As = info[2::4]
            states = info[3::4]
            CPs = info[4::4]
            

            for i in range(len(CPs)-1):
                start = int(CPs[i])
                end = int(CPs[i+1])
                temp_y[start:end] = states[i]
                temp_a[start:end] = As[i]

            print(states, CPs)

            temporal_states_y.append(temp_y)
            temporal_states_a.append(temp_a)
            temporal_states_D.append(Ds)

    Ly = [len(y) for y in temporal_states_y]
    La = [len(a) for a in temporal_states_a]
    Lt = [len(t) for t in tracks_all]

    for i in range(len(Ly)):
        assert Ly[i] == Lt[i], f'Ly: {Ly[i]}, Lt: {Lt[i]}'

    return tracks_all, temporal_states_y, temporal_states_a, temporal_states_D


tracks_all, temporal_states_y, temporal_states_a, temporal_states_D = load_and_transform_Andi_data(data_path)

np.unique(np.hstack(temporal_states_y))

# %%

for info in 