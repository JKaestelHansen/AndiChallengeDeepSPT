import warnings
import torch
import numpy as np
import torch.optim as optim

class GlobalConfig(object):

    #*************Data Config*****************
    datapath = '_Data/training5/ref/track_2'
    datapath = '_Data/training/ref/track_2'
    datapath = '_Data/training6/ref/track_2'
    datapath = '_Data/training7/ref/track_2'
    datapath = '_Data/starting_kit/track_2'
    datapath = '_Data/training2/ref/track_2'
    datapath = '_Data/training8/ref/track_2'
    datapath = '_Data/training9/ref/track_2'
    datapath = '_Data/training11/ref/track_2'
    datapath = '_Data/training13/ref/track_2'
    datapath = '_Data/training14/ref/track_2'

    the_data_is = '2D'
    if the_data_is=='2D':
        #*************Standard Config*****************
        val_size = 0.2
        test_size = 0.2
        seed = 42
        seeds = [42] #[42, 99, 191, 12345, 0]
        X_padtoken = 0
        y_padtoken = -1
        shuffle = True

        #*************Training Config*****************
        lr = 0.0001
        epochs = 500
        batch_size = 128
        optim_choice = optim.Adam

        #*************Model Config*****************
        features = ['XYZ', 'SL', 'DP']
        n_classes = 4 # number of classes to predict

        # small model
        init_channels = 60
        channel_multiplier = 2

        pooling = 'max'
        pools = [2, 2, 2, 2, 2, 2, 2]

        depth = 2
        dil_rate = 2

        enc_conv_nlayers = 2
        dec_conv_nlayers = 2
        bottom_conv_nlayers = 1
        out_nlayers = 2

        kernelsize = 7
        outconv_kernel = 3

        batchnorm = True
        batchnormfirst = True

        # big model
        init_channels = 130
        channel_multiplier = 2

        pooling = 'max'
        pools = [2, 2, 2, 2, 2, 2, 2]

        depth = 4
        dil_rate = 2

        enc_conv_nlayers = 2
        dec_conv_nlayers = 1
        bottom_conv_nlayers = 3
        out_nlayers = 4

        kernelsize = 7
        outconv_kernel = 3

        batchnorm = True
        batchnormfirst = True

        # bigger model
        init_channels = 150
        channel_multiplier = 2

        pooling = 'max'
        pools = [2, 2, 2, 2, 2, 2, 2]

        depth = 4
        dil_rate = 2

        enc_conv_nlayers = 2
        dec_conv_nlayers = 2
        bottom_conv_nlayers = 3
        out_nlayers = 4

        kernelsize = 7
        outconv_kernel = 3

        batchnorm = True
        batchnormfirst = True
    
    if the_data_is=='3D':
        #*************Standard Config*****************
        val_size = 0.2
        test_size = 0.2
        seed = 42
        seeds = [42] #[42, 99, 191, 12345, 0]
        X_padtoken = 0
        y_padtoken = 10
        shuffle = True

        #*************Training Config*****************
        lr = 0.00020933097456506567
        epochs = 100
        batch_size = 256
        optim_choice = optim.RMSprop

        #*************Model Config*****************
        features = ['XYZ', 'SL', 'DP']
        n_classes = 4 # number of classes to predict

        init_channels = 48
        channel_multiplier = 2

        pooling = 'max'
        pools = [2, 2, 2, 2, 2, 2, 2]

        depth = 3
        dil_rate = 2

        enc_conv_nlayers = 3
        dec_conv_nlayers = 4
        bottom_conv_nlayers = 4
        out_nlayers = 2

        kernelsize = 5
        outconv_kernel = 3

        batchnorm = True
        batchnormfirst = True

    def _parse(self, kwargs):
        """
        update config based on kwargs dictionary
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        globals.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

globals = GlobalConfig()
