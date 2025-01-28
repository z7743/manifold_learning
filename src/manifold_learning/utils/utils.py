import torch
import numpy as np
from sklearn.decomposition import PCA

def get_td_embedding_torch(ts, dim, stride, return_pred=False, tp=0):
    tdemb = ts.unfold(0,(dim-1) * stride + 1,1)[...,::stride]
    tdemb = torch.swapaxes(tdemb,-1,-2)
    if return_pred:
        return tdemb[:tdemb.shape[0]-tp], ts[(dim-1) * stride + tp:]
    else:
        return tdemb
    
def get_td_embedding_np(time_series, dim, stride, return_pred=False, tp=0):
    num_points, num_dims = time_series.shape
    # Calculate the size of the unfolding window
    window_size = (dim - 1) * stride + 1
    
    # Ensure the time series is long enough for the unfolding
    if num_points < window_size:
        raise ValueError("Time series is too short for the given dimensions and stride.")
    
    # Create an array to hold the unfolded data
    # Calculate shape and strides for as_strided
    shape = (num_points - window_size + 1, dim, num_dims)
    strides = (time_series.strides[0], stride * time_series.strides[0], time_series.strides[1])
    tdemb = np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)
    
    if return_pred:
        # Return the embedded data excluding the last 'tp' points, and the prediction points starting from a specific index
        return tdemb[:tdemb.shape[0]-tp], time_series[(dim - 1) * stride + tp:]
    else:
        return tdemb
    

def get_td_embedding_specified(time_series, delays):
    """
    [ChatGPT written]
    Embeds a time series using specified time delays by truncating the beginning of the series.

    Parameters:
    time_series (array-like): The input time series data.
    delays (array-like): An array of time delays.

    Returns:
    np.ndarray: A matrix where each column is the time series delayed by the respective delay.
    """
    n = len(time_series)
    max_delay = max(delays)
    
    # Ensure the time series is long enough for the maximum delay
    if max_delay >= n:
        raise ValueError("Maximum delay exceeds the length of the time series.")

    # Create an embedding matrix with shape (n - max_delay, len(delays))
    embedded = np.empty((n - max_delay, len(delays)))
    
    for i, delay in enumerate(delays):
        embedded[:, i] = time_series[max_delay - delay:n - delay]

    return embedded


def calculate_correlation_dimension(embedded_data, radii=None, device="cpu",max_samples=1e6):
    """
    ChatGPT writen

    Calculate the correlation dimension for a given embedded dataset using the Grassberger-Procaccia algorithm,
    rewritten to utilize GPU with PyTorch.

    Parameters:
        embedded_data (torch.Tensor): The pre-embedded input dataset on the GPU.
        radii (torch.Tensor): Optional tensor of radius values to use for correlation sum calculation.

    Returns:
        float: Estimated correlation dimension.
    """
    # Ensure embedded data is on GPU
    embedded_data = torch.tensor(embedded_data,device=device)

    # Calculate pairwise distances using PyTorch
    diff = embedded_data.unsqueeze(1) - embedded_data.unsqueeze(0)
    distance_matrix = torch.sqrt((diff ** 2).sum(-1))

    # Flatten and filter the distance matrix to get relevant quantiles
    distances = distance_matrix.flatten()
    nonzero_distances = distances[distances != 0]
    
    # Randomly sample distances if there are too many
    num_distances = nonzero_distances.numel()
    if num_distances > max_samples:
        # Randomly choose a subset of indices from nonzero distances
        sampled_indices = torch.randperm(num_distances, device=device)[:int(max_samples)]
        nonzero_distances = nonzero_distances[sampled_indices]

    # Compute quantiles
    q1, q2 = torch.quantile(nonzero_distances, 0.001), torch.quantile(nonzero_distances, 0.999)
    a = torch.log2(q1) - 2
    b = torch.log2(q2) + 2


    # Calculate radii if not provided
    if radii is None:
        radii = torch.logspace(a.item(), b.item(), 100, base=2, device=device)

    # Calculate correlation sums
    correlation_sums = [(distance_matrix < r).float().mean() for r in radii]

    # Logarithm of radii and correlation sums
    log_r = torch.log(radii)
    log_C_r = torch.log(torch.tensor(correlation_sums, device=device))

    # Applying mask and linear regression as per the original method
    thr = (log_C_r.max() - log_C_r.min()) / 20
    mask = (log_C_r > (log_C_r.min() + 1 * thr)) & (log_C_r < (log_C_r.max() - 10 * thr))

    # Apply mask
    log_r_masked = log_r[mask]
    log_C_r_masked = log_C_r[mask]

    # Create the design matrix for linear regression
    X = torch.stack([log_r_masked, torch.ones_like(log_r_masked)], dim=1).to(device)
    Y = log_C_r_masked.unsqueeze(1)

    # Compute (X^T * X)^(-1) * X^T * Y
    XTX_inv_XT = torch.linalg.pinv(X.T @ X) @ X.T
    beta = XTX_inv_XT @ Y

    # The slope is the first component of beta
    slope = beta[0].item()
    return slope


def calculate_local_correlation_statistics(embedded_data, radii=None, device="cpu", max_samples=1e6):
    # Ensure embedded data is a torch tensor on the appropriate device
    embedded_data = torch.tensor(embedded_data, device=device)

    # Calculate pairwise distances using PyTorch
    diff = embedded_data.unsqueeze(1) - embedded_data.unsqueeze(0)  # Shape: [N, N, D]
    distance_matrix = torch.sqrt((diff ** 2).sum(-1))  # Shape: [N, N]

    # Flatten and filter the distance matrix to get relevant quantiles
    distances = distance_matrix.flatten()
    nonzero_distances = distances[distances > 0]

    # Randomly sample distances if there are too many
    num_distances = nonzero_distances.numel()
    if num_distances > max_samples:
        # Randomly choose a subset of indices from nonzero distances
        sampled_indices = torch.randperm(num_distances, device=device)[:int(max_samples)]
        nonzero_distances = nonzero_distances[sampled_indices]

    # Compute quantiles
    q1, q2 = torch.quantile(nonzero_distances, 0.001), torch.quantile(nonzero_distances, 0.999)
    a = torch.log2(q1) - 2
    b = torch.log2(q2) + 2

    # Calculate radii if not provided
    if radii is None:
        radii = torch.logspace(a.item(), b.item(), steps=100, base=2, device=device)

    log_r = torch.log(radii)  # Shape: [R]

    # Vectorized computation of masks for all radii
    # Shape of masks: [R, N, N]
    masks = (distance_matrix.unsqueeze(0) < radii.unsqueeze(1).unsqueeze(2))

    # Compute correlation sums per point
    # Shape after mean: [R, N], transpose to [N, R]
    correlation_sums = masks.float().mean(dim=2).T + 1e-10  # Avoid log(0)

    # Logarithm of correlation sums
    log_C_r = torch.log(correlation_sums)  # Shape: [N, R]

    return log_C_r

def calculate_rank_for_variance(data_matrix, variance_threshold=0.95):
    # Perform PCA
    pca = PCA()
    pca.fit(data_matrix)
    
    # Calculate the cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components that explain at least the desired variance
    rank = np.searchsorted(cumulative_variance_ratio, variance_threshold) + 1
    
    return rank

def calculate_moving_average_firing_rate(spike_times, window_size, output_sampling_rate):
    """
    Calculate the firing rate using a moving average from spike times.

    Parameters:
    spike_times (list of np.array): A list where each element is an array of spike times for a neuron.
    window_size (float): The size of the moving average window in seconds.
    output_sampling_rate (float): The sampling rate for the output firing rate in Hz.

    Returns:
    tuple: (firing_rates, time_points) where firing_rates is a list of np.array of firing rates
           for each neuron, and time_points is the common time axis.
    """
    # Determine the total duration
    max_duration = max(spikes[-1] for spikes in spike_times)
    
    # Create a common time axis
    dt = 1.0 / output_sampling_rate
    time_points = np.arange(0, max_duration + dt, dt)
    
    firing_rates = []
    for neuron_spikes in spike_times:
        # Initialize the firing rate array
        firing_rate = np.zeros(len(time_points))
        
        # Calculate cumulative spike count
        cumulative_spikes = np.zeros_like(time_points)
        for spike in neuron_spikes:
            cumulative_spikes[np.searchsorted(time_points, spike)] += 1
        cumulative_spikes = np.cumsum(cumulative_spikes)
        
        # Calculate the moving average using the cumulative spike count
        half_window_size = window_size / 2
        for i, t in enumerate(time_points):
            window_start = max(0, np.searchsorted(time_points, t - half_window_size))
            window_end = np.searchsorted(time_points, t + half_window_size)
            if window_start == 0:
                spike_count = cumulative_spikes[window_end - 1]
            else:
                spike_count = cumulative_spikes[window_end - 1] - cumulative_spikes[window_start - 1]
            firing_rate[i] = spike_count / window_size
        
        firing_rates.append(firing_rate)
    
    return firing_rates, time_points
