import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import datetime
import numpy as np

def jaccard_index(pred_cp, true_cp, threshold=10):
    # Convert to tensors if not already
    if not isinstance(pred_cp, torch.Tensor):
        pred_cp = torch.tensor(pred_cp)
    if not isinstance(true_cp, torch.Tensor):
        true_cp = torch.tensor(true_cp)

    # Sort the change points
    pred_cp = torch.sort(pred_cp)[0]
    true_cp = torch.sort(true_cp)[0]

    # Initialize TP, FP, FN
    TP = 0
    matched_pred = torch.zeros_like(pred_cp, dtype=torch.bool)
    matched_true = torch.zeros_like(true_cp, dtype=torch.bool)

    # Match true change points to predicted change points
    for i, true_point in enumerate(true_cp):
        # Find the predicted points within the threshold
        within_threshold = torch.abs(pred_cp - true_point) <= threshold
        if within_threshold.any():
            # Find the closest predicted point within the threshold
            distances = torch.abs(pred_cp[within_threshold] - true_point)
            closest_idx = torch.argmin(distances)
            pred_idx = torch.where(within_threshold)[0][closest_idx]
            if not matched_pred[pred_idx]:
                matched_pred[pred_idx] = True
                matched_true[i] = True
                TP += 1

    # Calculate FP and FN
    FP = (~matched_pred).sum().item()
    FN = (~matched_true).sum().item()

    # Calculate Jaccard index
    jaccard = TP / (TP + FP + FN)
    
    return jaccard, TP, FP, FN, matched_pred, matched_true


def mode_of_tensor(tensor):
    """
    Compute the mode of a PyTorch tensor.
    
    Args:
    tensor: A PyTorch tensor.
    
    Returns:
    The mode of the tensor.
    """
    # Ensure the tensor is 1-dimensional for simplicity
    if tensor.dim() != 1:
        tensor = tensor.flatten()
    
    # Get unique elements and their counts
    unique_elements, counts = torch.unique(tensor, return_counts=True)
    
    # Find the index of the maximum count
    max_count_index = torch.argmax(counts)
    
    # Retrieve the element with the highest count (mode)
    mode = unique_elements[max_count_index]
    
    return mode

def union_change_points(cp1, cp2, cp3):
    """
    Combine all unique change points from three PyTorch tensors.
    
    Args:
    cp1, cp2, cp3: PyTorch tensors of change points.
    
    Returns:
    A PyTorch tensor of unique change points sorted in ascending order.
    """
    
    concatenated_tensor = torch.cat([cp1, cp2, cp3])
    
    # Get unique elements while maintaining gradient tracking
    unique_tensor, _ = torch.unique(concatenated_tensor, sorted=True, return_inverse=True)
    unique_tensor = differentiable_union(concatenated_tensor)
    sorted_union = unique_tensor[torch.argsort(unique_tensor)]
    
    return sorted_union

def differentiable_union(tensor):
    unique_tensor, indices = torch.unique(tensor, return_inverse=True)
    one_hot = torch.nn.functional.one_hot(indices)
    one_hot_float = one_hot.type(tensor.dtype)
    summed_tensor = torch.mm(one_hot_float.T, tensor.unsqueeze(1)).squeeze()
    counts = one_hot_float.sum(0)
    union_tensor = summed_tensor / counts
    return union_tensor

def find_segments(inarray):
    """ 
    input: predicted labels, diffusion labels shape = (n,)
    output: segment run lengths, start positions of segments, difftypes of segments
    """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]             # pairwise unequal (string safe)
        i = np.append(np.where(y), n-1)   # must include last element posi
        z = np.diff(np.append(-1, i))     # run lengths
        p = np.cumsum(np.append(0, z)) # positions
        return(z, p, ia[i])

def find_segments_torch(inarray):
    """
    input: predicted labels, diffusion labels shape = (n,)
    output: segment run lengths, start positions of segments, difftypes of segments
    """
    ia = inarray.clone()
    device = ia.device
    n = ia.size(0)
    if n == 0:
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]             # pairwise unequal (string safe)
        i = torch.cat((torch.nonzero(y).flatten(), torch.tensor([n-1], device=device)))  # must include last element position
        z = torch.diff(torch.cat((torch.tensor([-1], device=device), i)))     # run lengths
        p = torch.cumsum(torch.cat((torch.tensor([0], device=device), z[:-1])), dim=0)  # positions
        return(z, p, ia[i])  # convert back to numpy if needed

def find_change_points(tensor):
    """
    Find the change points in a tensor while retaining gradients.
    
    Args:
    tensor (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: Tensor of the same shape as input with 1s at change points and 0s elsewhere.
    """
    # Compute the difference between consecutive elements
    # add a max value of tensor plus 1 to end of tensor
    tensor = torch.cat((tensor, tensor[-1].unsqueeze(0)+1))
    diff = tensor[1:] - tensor[:-1]
    
    # Use the ReLU function to get change points (differentiable)
    change_points = torch.abs(torch.sign(diff))
    
    # Pad the result to maintain the same shape as the input tensor
    change_points = torch.cat((torch.tensor([0.0], device=tensor.device), change_points))

    change_points = torch.arange(tensor.size(0), device=tensor.device).float() * change_points
    mask = change_points != 0
    change_points = change_points[mask]
    return change_points

def create_state_array(a, threshold):
    a = a[0]
    device = a.device

    b = torch.ones_like(a, device=device)
    c = a.clone()

    cluster = 1
    for i in range(1, a.size(0)):
        if (a[i] - a[i - 1]).abs() > threshold:
            cluster += 1
        b[i] = cluster

    # iterate thru unique in b
    for i, val in enumerate(b.unique()):
        indices = (b == val)
        c = torch.where(indices, torch.tensor(i, device=device), c)

    return c

class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Ensure y_pred and y_true are positive by adding a small value (epsilon) to avoid log(0)
        epsilon = 1e-6
        y_pred_log = torch.log(y_pred + 1 + epsilon)
        y_true_log = torch.log(y_true + 1 + epsilon)
        
        return self.mse_loss(y_pred_log, y_true_log)

def soft_jaccard_loss_with_threshold(preds, targets, threshold=10):
    """
    Compute a soft Jaccard loss where predictions and true values are paired if they are within a specified threshold.
    
    Args:
    preds (torch.Tensor): Predicted values.
    targets (torch.Tensor): True values.
    threshold (float): Maximum allowed distance for pairing.
    
    Returns:
    torch.Tensor: Soft Jaccard loss.
    """
    preds = preds.float()
    targets = targets.float()
    
    # Create an expanded distance matrix
    preds_expanded = preds.unsqueeze(1).expand(-1, targets.size(0))
    targets_expanded = targets.unsqueeze(0).expand(preds.size(0), -1)
    
    # Calculate the absolute differences
    abs_diff = torch.abs(preds_expanded - targets_expanded)
    
    # Create a mask for valid pairs (within threshold)
    valid_pairs = (abs_diff <= threshold).float()
    
    # Pairing logic: greedily match predictions to targets
    intersection = 0
    for i in range(valid_pairs.size(0)):
        for j in range(valid_pairs.size(1)):
            if valid_pairs[i, j] == 1:
                intersection += 1
                valid_pairs[:, j] = 0  # Mark the target as used
                valid_pairs[i, :] = 0  # Mark the prediction as used
                break
    
    union = preds.size(0) + targets.size(0) - intersection
    soft_jaccard = intersection / (union + 1e-6)
    
    return 1 - soft_jaccard

def mse_loss_on_paired_values(preds, targets, threshold=10):
    """
    Compute MSE loss on paired predicted and true values within a specified threshold,
    ensuring the closest pairs are chosen.
    
    Args:
    preds (torch.Tensor): Predicted values.
    targets (torch.Tensor): True values.
    threshold (float): Maximum allowed distance for pairing.
    
    Returns:
    torch.Tensor: MSE loss on paired values.
    """
    preds = preds.float()
    targets = targets.float()
    
    # Create an expanded distance matrix
    preds_expanded = preds.unsqueeze(1).expand(-1, targets.size(0))
    targets_expanded = targets.unsqueeze(0).expand(preds.size(0), -1)
    
    # Calculate the absolute differences
    abs_diff = torch.abs(preds_expanded - targets_expanded)
    
    # Create a mask for valid pairs (within threshold)
    valid_pairs = (abs_diff <= threshold).float()
    
    # Initialize lists to store paired predictions and targets
    paired_preds = []
    paired_targets = []
    
    # Iterate to find the closest pairs
    while valid_pairs.sum() > 0:
        # Find the closest pair
        min_val, min_idx = abs_diff[valid_pairs.bool()].min(dim=0)
        pair_idx = valid_pairs.nonzero(as_tuple=False)[min_idx]
        i, j = pair_idx[0].item(), pair_idx[1].item()
        
        # Add the pair to the lists
        paired_preds.append(preds[i])
        paired_targets.append(targets[j])
        
        # Mark the prediction and target as used
        valid_pairs[i, :] = 0
        valid_pairs[:, j] = 0
        abs_diff[i, :] = float('inf')
        abs_diff[:, j] = float('inf')
    
    # Convert paired lists to tensors
    if paired_preds and paired_targets:
        paired_preds = torch.stack(paired_preds)
        paired_targets = torch.stack(paired_targets)
        
        # Compute the MSE loss on paired values
        mse_loss = torch.mean((paired_preds - paired_targets) ** 2)
    else:
        mse_loss = torch.tensor(0.0, device=preds.device)
    
    return mse_loss


def combined_cp_loss(a, b, alpha=0.5, threshold=10):
    """
    Combine soft Jaccard loss and MSE loss.
    
    Args:
    a (torch.Tensor): Predicted values.
    b (torch.Tensor): True values.
    alpha (float): Weight for the soft Jaccard loss.
    
    Returns:
    torch.Tensor: Combined loss.
    """
    jaccard = soft_jaccard_loss_with_threshold(a, b, threshold=threshold)
    mse = mse_loss_on_paired_values(a, b)
    return alpha * jaccard + (1 - alpha) * mse, jaccard, mse


""" Parts of the U-Net model """
class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, 
                 dilation=2, nlayers=2, batchnorm=True, 
                 batchnormfirst=True, padding=1):
        super().__init__()
        layers = []
        for i in range(nlayers):
            channels = in_channels if i == 0 else out_channels
            layers.append(nn.Conv1d(channels, out_channels, kernel_size=kernel_size, 
                        padding=int(dilation*(kernel_size-1)/2), dilation=dilation))
            if batchnorm and batchnormfirst:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if batchnorm and not batchnormfirst:
                layers.append(nn.BatchNorm1d(out_channels))
        self.multi_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.multi_conv(x)


class Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 batchnorm=True, batchnormfirst=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=int((kernel_size-1)/2), dilation=1))
        if batchnorm and batchnormfirst:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if batchnorm and not batchnormfirst:
            layers.append(nn.BatchNorm1d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, outconv_kernel=3,
                 nlayers=2, batchnorm=True, batchnormfirst=False):
        super(OutConv, self).__init__()
        self.conv_softmax = nn.Sequential(MultiConv(in_channels, out_channels, 
                                                     kernel_size=outconv_kernel, dilation=1,
                                                     padding=int((outconv_kernel-1)/2), nlayers=nlayers, batchnorm=batchnorm, 
                                                     batchnormfirst=batchnormfirst),
                                          nn.Conv1d(in_channels, n_classes, kernel_size=1, padding=0))
    def forward(self, x):
        return self.conv_softmax(x)


""" Full assembly of the parts to form the complete network """
class hypoptUNet(nn.Module):
    def __init__(self, n_features: int = 4, init_channels: int = 16, 
                 n_classes: int = 4, depth: int = 4, enc_kernel: int = 5,
                 dec_kernel: int = 5, outconv_kernel: int = 5, 
                 dil_rate = 2, pools: list = [2, 2, 2, 2], 
                 pooling: str = 'max', enc_conv_nlayers: int = 2,
                 dec_conv_nlayers: int = 2, bottom_conv_nlayers: int = 2,
                 out_nlayers: int = 2, X_padtoken: int = 0, 
                 y_padtoken: int = 10, batchnorm: bool = True, 
                 batchnormfirst: bool = False, channel_multiplier: int = 2,
                 device: str = 'cpu',
                 threshold: float = 0.05):
        super(hypoptUNet, self).__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.depth = depth
        self.pools = pools
        self.decoder_scale_factors = pools[::-1]
        self.enc_kernel = enc_kernel
        self.dec_kernel = dec_kernel
        self.outconv_kernel = outconv_kernel
        self.dil_rate = dil_rate
        self.enc_conv_nlayers = enc_conv_nlayers
        self.dec_conv_nlayers = dec_conv_nlayers
        self.bottom_conv_nlayers = bottom_conv_nlayers
        self.out_nlayers = out_nlayers
        self.batchnorm = batchnorm
        self.batchnormfirst = batchnormfirst
        self.channel_multiplier = channel_multiplier
        self.pooling = nn.MaxPool1d if pooling=='max' else nn.AvgPool1d
        self.X_padtoken = X_padtoken
        self.y_padtoken = y_padtoken
        self.device = device
        self.threshold = threshold 

        self.module_list = nn.ModuleList()
        in_channels = n_features
        out_channels = init_channels
        res_channels = []
        for i in range(depth):
            self.module_list.append(MultiConv(int(in_channels), 
                                            int(out_channels), 
                                            kernel_size=self.enc_kernel, 
                                            dilation=self.dil_rate,
                                            nlayers=enc_conv_nlayers, 
                                            batchnorm=batchnorm, 
                                            batchnormfirst=batchnormfirst))
            in_channels = out_channels
            res_channels.append(out_channels)
            out_channels *= channel_multiplier
        self.module_list.append(MultiConv(int(in_channels), 
                                          int(out_channels), 
                                          kernel_size=self.enc_kernel, 
                                          dilation=self.dil_rate,
                                          nlayers=bottom_conv_nlayers, 
                                          batchnorm=batchnorm, 
                                          batchnormfirst=batchnormfirst))
        in_channels = out_channels
        out_channels /= channel_multiplier
        for i in range(depth):
            self.module_list.append(Conv_layer(int(in_channels), 
                                                int(out_channels), 
                                                kernel_size=self.dec_kernel,
                                                batchnorm=batchnorm, 
                                                batchnormfirst=batchnormfirst))
    
            merge_channels = out_channels + res_channels[::-1][i]

            self.module_list.append(MultiConv(int(merge_channels), 
                                              int(out_channels), 
                                              kernel_size=self.dec_kernel, 
                                              dilation=1,
                                              nlayers=dec_conv_nlayers, 
                                              batchnorm=batchnorm, 
                                              batchnormfirst=batchnormfirst))
            in_channels = out_channels
            if i != self.depth-1:
                out_channels /= channel_multiplier

        self.module_list.append(OutConv(int(out_channels), 
                                int(out_channels), 
                                self.n_classes, 
                                outconv_kernel=self.outconv_kernel,
                                nlayers=out_nlayers, 
                                batchnorm=batchnorm, 
                                batchnormfirst=batchnormfirst))
        self.decoder_list = self.module_list[(depth+1):(depth+2*depth+1)]

        self.to(device)

    def concat(self, x1, x2):
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffX, 0])
        x = torch.cat([x2, x1], dim=1)
        return x

    def match_x1_to_x2(self, x1=None, x2=None, value=0):
        diffX = x2.size()[2] - x1.size()[2]
        x = F.pad(x1, [diffX, 0], value=value)
        return x

    def predict(self, test_loader, temperature=1):
        with torch.no_grad():
            masked_ys = []
            masked_argmaxs = []
            masked_a_preds = []
            masked_D_preds = []
            andi_pred_formats = []
            CPS_p_lists = []
            CPS_a_lists = []
            CPS_D_lists = []
            for xb in tqdm(test_loader):
                out = self.forward(xb, inference=True, temperature=temperature)
                _, _, _, masked_y, masked_argmax, masked_a_pred, masked_D_pred, andi_pred_format, CPS_p_list, CPS_a_list, CPS_d_list  = out
                masked_ys.append(masked_y)
                masked_argmaxs.append(masked_argmax)
                masked_a_preds.append(masked_a_pred)
                masked_D_preds.append(masked_D_pred)
                andi_pred_formats.append(andi_pred_format)
                CPS_p_lists.append(CPS_p_list)
                CPS_a_lists.append(CPS_a_list)
                CPS_D_lists.append(CPS_d_list)
        
        masked_a_preds = [i for s in masked_a_preds for i in s]
        masked_D_preds = [i for s in masked_D_preds for i in s]
        masked_argmaxs = [i for s in masked_argmaxs for i in s]
        masked_ys = [i for s in masked_ys for i in s]
        andi_pred_formats = [i for s in andi_pred_formats for i in s]
        CPS_p_lists = [i for s in CPS_p_lists for i in s]
        CPS_a_lists = [i for s in CPS_a_lists for i in s]
        CPS_D_lists = [i for s in CPS_D_lists for i in s]

        return masked_argmaxs, masked_a_preds, masked_D_preds, masked_ys, andi_pred_formats, CPS_p_lists, CPS_a_lists, CPS_D_lists

    def simple_predict(self, xb, temperature=1):
        with torch.no_grad():
            out = self.forward(xb, inference=True, temperature=temperature)
            _, _, _, masked_y, masked_pred, masked_argmax, andi_pred_formats  = out
        return masked_argmax, masked_pred, masked_y, andi_pred_formats

    def forward(self, xb, inference=False, temperature=1, dim=2):
        x, y_states, y_alpha, y_D, y_CPs = xb
        traj_x, _, _, _, _ = xb

        input = xb[0]
        residuals_list = []
        for i in range(self.depth):
            pool = self.pools[i]
            res = self.module_list[i](x)
            x = self.pooling(pool)(res)
            residuals_list.append(res)
        x = self.module_list[self.depth](x)
        residual = residuals_list[::-1]
        for i in range(0, self.depth*2, 2):
            scale_factor = self.decoder_scale_factors[i//2]
            up = nn.Upsample(scale_factor=scale_factor, mode='nearest')(x)
            x = self.decoder_list[i](up)
            merged = self.concat(x, residual[i//2])
            x = self.decoder_list[i+1](merged)
        merged = self.match_x1_to_x2(x1=x, x2=input, value=0)
        pred = self.module_list[-1](merged)
        loss_states = 0
        loss_alpha = 0
        loss_D = 0
        loss_jacc = 0
        loss_combined_CP = 0
        acc = 0
        F1 = 0
        criterion = nn.CrossEntropyLoss()
        criterion_MAE = nn.L1Loss()
        criterion_MSLE = MSLELoss()
        masked_ys = []
        masked_preds = []
        masked_argmax = []
        masked_a_pred = []
        masked_D_pred = []
        andi_pred_formats = []
        CPS_p_list = []
        CPS_a_list = []
        CPS_d_list = []

        count = 0
        
        for i in range(len(y_states)):
            mask_idx = sum(y_states[i].le(self.y_padtoken))
            mask_idx_cp = sum(y_CPs[i]==self.y_padtoken)

            # nn.Softmax(dim=1) channels med states
            # nn.Sigmoid alpha channel of gang med 2
            masked_y = y_states[i][mask_idx:].unsqueeze(0)
            masked_a = y_alpha[i][mask_idx:].unsqueeze(0)
            masked_D = y_D[i][mask_idx:].unsqueeze(0)
            masked_CP = y_CPs[i][mask_idx_cp:].unsqueeze(0)
            masked_pred = pred[i][:,mask_idx:].unsqueeze(0)

            masked_pred_states = F.softmax(masked_pred[:,:-2], dim=1)
            masked_pred_alpha = torch.sigmoid(masked_pred[:,-2])*2
            masked_pred_D = nn.ReLU()(masked_pred[:,-1])

            pred_argmax_masked_pred = masked_pred_states.argmax(1)
            cps_pred = find_change_points(pred_argmax_masked_pred[0]+torch.tensor(1, device=masked_pred.device))

            threshold = self.threshold
            a_states = create_state_array(masked_pred_alpha, threshold)
            D_states = create_state_array(masked_pred_D, threshold)

            cps_pred_a = find_change_points(a_states)
            cps_pred_D = find_change_points(D_states)

            cp_pred_all = union_change_points(cps_pred, cps_pred_a, cps_pred_D)
            
            # augment with simple apparent D calculation
            sl_D_tensor = torch.zeros_like(masked_pred_D)
            cp_pred_all_w_zero = torch.cat((torch.tensor([0], device=cp_pred_all.device), cp_pred_all))
            for cpi in range(len(cp_pred_all_w_zero)-1):
                start = cp_pred_all_w_zero[cpi].int()
                end = cp_pred_all_w_zero[cpi+1].int()
                if end-start <= 1 and end+1 <= len(masked_pred_D[0]):
                    end += 1
                    
                elif end-start <= 1 and end+1 >= len(masked_pred_D[0]):
                    start -= 1
                
                x_segm = traj_x[i][:,mask_idx:][0,start:end]
                y_segm = traj_x[i][:,mask_idx:][1,start:end]

                MSD = ((x_segm[1:]-x_segm[:-1])**2 + (y_segm[1:]-y_segm[:-1])**2).mean().float()

                 # dim of track corresponds to the idx of SL. +1 to correct for SL starts in 0
                sl_D_tensor[0][start:end] = MSD / 4

            masked_pred_D_pre = masked_pred_D
            masked_pred_D =  masked_pred_D #+ sl_D_tensor


            # if nan in sl_D_tensor or masked_pred_D:
            #     print('nan in D')
            #     print(sl_D_tensor)


            # make andi format
            andi_pred_format = []
            for cpi in range(len(cp_pred_all_w_zero)-1):

                start = cp_pred_all_w_zero[cpi].int()
                end = cp_pred_all_w_zero[cpi+1].int()

                D_segment = masked_pred_D[0][start:end]
                alpha_segment = masked_pred_alpha[0][start:end]
                state_segment = pred_argmax_masked_pred[0][start:end]

                andi_pred_format.append(D_segment.mean().float().item())
                andi_pred_format.append(alpha_segment.mean().float().item())
                andi_pred_format.append(mode_of_tensor(state_segment).item())
                if cpi <= len(cp_pred_all_w_zero)-2:
                    andi_pred_format.append(cp_pred_all[cpi].detach().cpu().item())
                else:
                    andi_pred_format.append(cp_pred_all[-1].detach().cpu().item()+1)

            jaccard, TP, FP, FN, matched_pred, matched_true = jaccard_index(cp_pred_all, masked_CP[0], threshold=10)

            loss_states += criterion(masked_pred_states, masked_y.long()) # loss on channels for states
            loss_alpha += criterion_MAE(masked_pred_alpha, masked_a) # loss on extra channel for alpha
            loss_D += criterion_MSLE(masked_pred_D, masked_D)
            loss_jacc += 1-jaccard
            loss_combined_CP += combined_cp_loss(cp_pred_all, masked_CP[0])[0]

            #print('need a loss on CPs and also figure out how they compare if we have more/less cp than GT')

            # need to run find segments of states
            # calculate D in each segment
            # average the alpha in each segment
            # how to handle if I have too many segments or too few??
            
            acc += (masked_pred_states.argmax(1) == masked_y).float().mean()
            from sklearn.metrics import f1_score
            F1 += f1_score(masked_y.squeeze(0).detach().cpu().numpy(), 
                          masked_pred_states.argmax(1).squeeze(0).detach().cpu().numpy(),
                          average='micro')

            masked_ys.append(masked_y.cpu().squeeze(0).numpy())
            masked_preds.append(masked_pred.detach().cpu().squeeze(0).numpy())
            masked_argmax.append(masked_pred_states.argmax(1).squeeze(0).detach().cpu().numpy())
            masked_a_pred.append(masked_pred_alpha.detach().cpu().numpy())
            masked_D_pred.append(masked_pred_D.detach().cpu().numpy())
            andi_pred_formats.append(andi_pred_format)
            CPS_p_list.append(cp_pred_all.detach().cpu().numpy())
            CPS_a_list.append(cps_pred_a.detach().cpu().numpy())
            CPS_d_list.append(cps_pred_D.detach().cpu().numpy())

        loss_states /= y_states.shape[0]
        loss_alpha /= y_states.shape[0]
        loss_D /= y_states.shape[0]
        acc /= y_states.shape[0]
        F1 /= y_states.shape[0]
        loss_jacc /= y_states.shape[0]
        loss_combined_CP /= y_states.shape[0]

        loss = loss_states + loss_alpha + loss_D #+ loss_combined_CP

        if inference:
            return loss, acc, pred, masked_ys, masked_argmax, masked_a_pred, masked_D_pred, andi_pred_formats, CPS_p_list, CPS_a_list, CPS_d_list
        else:
            return loss, acc, pred, F1, loss_states, loss_alpha, loss_D, loss_jacc, loss_combined_CP


class encoderhypoptUNet(nn.Module):
    def __init__(self, n_features: int = 4, init_channels: int = 16, 
                 n_classes: int = 4, depth: int = 4, enc_kernel: int = 5,
                 dil_rate = 2, pools: list = [2, 2, 2, 2], pooling: str = 'max', 
                 enc_conv_nlayers: int = 2, bottom_conv_nlayers: int = 2,
                 X_padtoken: int = 0, y_padtoken: int = 10, batchnorm: bool = True, 
                 batchnormfirst: bool = False, channel_multiplier: int = 2,
                 device: str = 'cpu'):
        super(hypoptUNet, self).__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.depth = depth
        self.pools = pools
        self.enc_kernel = enc_kernel
        self.dil_rate = dil_rate
        self.enc_conv_nlayers = enc_conv_nlayers
        self.bottom_conv_nlayers = bottom_conv_nlayers
        self.batchnorm = batchnorm
        self.batchnormfirst = batchnormfirst
        self.channel_multiplier = channel_multiplier
        self.pooling = nn.MaxPool1d if pooling=='max' else nn.AvgPool1d
        self.X_padtoken = X_padtoken
        self.y_padtoken = y_padtoken
        self.device = device

        self.module_list = nn.ModuleList()
        in_channels = n_features
        out_channels = init_channels
        res_channels = []
        for i in range(depth):
            self.module_list.append(MultiConv(int(in_channels), 
                                            int(out_channels), 
                                            kernel_size=self.enc_kernel, 
                                            dilation=self.dil_rate,
                                            nlayers=enc_conv_nlayers, 
                                            batchnorm=batchnorm, 
                                            batchnormfirst=batchnormfirst))
            in_channels = out_channels
            res_channels.append(out_channels)
            out_channels *= channel_multiplier
        self.module_list.append(MultiConv(int(in_channels), 
                                          int(out_channels), 
                                          kernel_size=self.enc_kernel, 
                                          dilation=self.dil_rate,
                                          nlayers=bottom_conv_nlayers, 
                                          batchnorm=batchnorm, 
                                          batchnormfirst=batchnormfirst))
        
        self.to(device)

    def concat(self, x1, x2):
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffX, 0])
        x = torch.cat([x2, x1], dim=1)
        return x

    def match_x1_to_x2(self, x1=None, x2=None, value=0):
        diffX = x2.size()[2] - x1.size()[2]
        x = F.pad(x1, [diffX, 0], value=value)
        return x

    def forward(self, xb):
        x, _ = xb
        residuals_list = []
        for i in range(self.depth):
            pool = self.pools[i]
            res = self.module_list[i](x)
            x = self.pooling(pool)(res)
            residuals_list.append(res)
        x = self.module_list[self.depth](x)
        
        return x