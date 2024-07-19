# %%
import torch
import torch.nn.functional as F

def cluster_floats_with_gradients(a, distance):
    """
    Cluster a tensor of floats into unique groups while maintaining gradients.
    
    Args:
    a (torch.Tensor): Tensor of floats.
    distance (float): Maximum distance for clustering floats.
    
    Returns:
    torch.Tensor: Tensor of integers representing clusters.
    """
    a = a.to(dtype=torch.float32)
    a.requires_grad_(True)

    # Create a tensor to store the cluster assignments
    b = torch.zeros_like(a)
    # clone a to tensor c
    c = a.clone()
    d = a.clone()

    # Initialize the first cluster
    cluster = 0

    # Iterate through the tensor to assign clusters
    for i in range(1, a.size(0)):
        if (a[i] - a[i - 1]).abs() > distance:
            cluster += 1
        b[i] = cluster

    # iterate thru unique in b
    for i, val in enumerate(b.unique()):
        indices = (b == val)
        
        c = torch.where(indices, torch.tensor(i), c)

    # Make b a differentiable float tensor first and then convert to integer tensor
    b = b.float()
    b_int = b.clone().round().long()

    print(a)
    print(b)
    print(c)
    # Ensure gradient flow
    def custom_round(x):
        return (x - x) + x.round()

    b = custom_round(b)
    return b_int

# Example usage:
a = torch.tensor([0.1, 0.4, 1.2, 1.3, 2.5], requires_grad=True)
distance = 0.5
b = cluster_floats_with_gradients(a, distance)

# To verify gradients
b.sum().backward()
print(a.grad)


# %%

def find_change_points(tensor):
    """
    Find the change points in a tensor while retaining gradients.
    
    Args:
    tensor (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: Tensor of the same shape as input with 1s at change points and 0s elsewhere.
    """
    # Compute the difference between consecutive elements
    diff = tensor[1:] - tensor[:-1]
    
    # Use the ReLU function to get change points (differentiable)
    change_points = torch.relu(torch.sign(diff))
    
    # Pad the result to maintain the same shape as the input tensor
    change_points = torch.cat((torch.tensor([0.0], device=tensor.device), change_points))

    change_points = torch.arange(tensor.size(0), device=tensor.device).float() * change_points
    mask = change_points != 0
    change_points = change_points[mask]
    return change_points


tensor = torch.tensor([1., 4., 4., 5., 5], requires_grad=True)

find_change_points(tensor)