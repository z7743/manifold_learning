import torch
from torch.utils.data import Dataset


class RandomTpRangeSubsetDataset(Dataset):
    def __init__(self, X, sample_size, subset_size, num_batches=32, tp_range=torch.arange(1,2), device="cuda"):
        """
        Initializes the RandomSubsetDataset.
        
        Args:
            X (torch.Tensor): Multivariate time series to be sampled.
            y (torch.Tensor): Multivariate time series coupled with X.
            sample_len (int): Number of samples used for prediction.
            library_len (int): Number of samples used for kNN search.
            num_batches (int): Number of random batches to be produced.
            device (str, optional): Device to place the dataset on, default is "cuda".
        """
        self.device = device
        self.X = X
        self.sample_size = sample_size
        self.subset_size = subset_size
        self.tp_range = tp_range
        self.tp_max = tp_range.max()
        self.num_batches = num_batches
        self.num_datapoints = X.shape[0]

    def __len__(self):
        return self.tp_range.shape[0] #* self.num_batches #Temporary solution to sample number of samples
    
    def __getitem__(self, idx):
        sample_idx = torch.argsort(torch.rand(self.num_datapoints-self.tp_max-1,device=self.device))[0:self.sample_size]
        subset_idx = torch.argsort(torch.rand(self.num_datapoints-self.tp_max-1,device=self.device))[0:self.subset_size + self.sample_size]
        subset_idx = subset_idx[(subset_idx.view(1, -1) != sample_idx.view(-1, 1)).all(dim=0)][0:self.subset_size]
        
        return subset_idx, sample_idx, self.X[subset_idx],self.X[subset_idx+self.tp_range[idx%self.tp_range.shape[0]]],\
                                        self.X[sample_idx], self.X[sample_idx+self.tp_range[idx%self.tp_range.shape[0]]]


class RandomTimeDelaySubsetDataset(Dataset):
    def __init__(self, X, sample_size, subset_size, E, tau, num_batches=32, tp_range=torch.arange(1,2), device="cuda"):
        """
        Initializes the RandomSubsetDataset.
        
        Args:
            X (torch.Tensor): Multivariate time series to be sampled.
            y (torch.Tensor): Multivariate time series coupled with X.
            sample_len (int): Number of samples used for prediction.
            library_len (int): Number of samples used for kNN search.
            num_batches (int): Number of random batches to be produced.
            device (str, optional): Device to place the dataset on, default is "cuda".
        """
        self.device = device
        self.X = X
        self.sample_size = sample_size
        self.subset_size = subset_size
        self.tp_range = tp_range
        self.tp_max = tp_range.max()
        self.num_batches = num_batches
        self.num_datapoints = X.shape[0]
        self.E = E
        self.tau = tau
        

    def __len__(self):
        return self.tp_range.shape[0] #Temporary solution to sample number of samples
    
    def __getitem__(self, idx):
        sample_idx = (self.E-1)*self.tau + torch.argsort(torch.rand(self.num_datapoints-self.tp_max - 1 - (self.E-1)*self.tau, device=self.device))[0:self.sample_size]
        subset_idx = (self.E-1)*self.tau + torch.argsort(torch.rand(self.num_datapoints-self.tp_max - 1 - (self.E-1)*self.tau, device=self.device))[0:self.subset_size + self.sample_size]
        subset_idx = subset_idx[(subset_idx.view(1, -1) != sample_idx.view(-1, 1)).all(dim=0)][0:self.subset_size]

        sample_idx = sample_idx - (self.tau * torch.arange(1, self.E + 1, device=self.device).unsqueeze(1))
        subset_idx = subset_idx - (self.tau * torch.arange(1, self.E + 1, device=self.device).unsqueeze(1))
        
        return subset_idx, sample_idx, self.X[subset_idx],self.X[subset_idx+self.tp_range[idx%self.tp_range.shape[0]]],\
                                        self.X[sample_idx], self.X[sample_idx+self.tp_range[idx%self.tp_range.shape[0]]]

class RandomXYSubsetDataset(Dataset):
    def __init__(self, X, y, sample_len, library_len, num_batches=32, device="cuda"):
        """
        Initializes the RandomSubsetDataset.
        
        Args:
            X (torch.Tensor): Multivariate time series to be sampled.
            y (torch.Tensor): Multivariate time series coupled with X.
            sample_len (int): Number of samples used for prediction.
            library_len (int): Number of samples used for kNN search.
            num_batches (int): Number of random batches to be produced.
            device (str, optional): Device to place the dataset on, default is "cuda".
        """
        self.device = device
        self.X = X
        self.y = y
        self.sample_len = sample_len
        self.library_len = library_len
        self.num_batches = num_batches
        self.num_datapoints = X.shape[0]

    def __len__(self):
        return self.num_batches#Temporary solution to sample number of samples
    
    def __getitem__(self, idx):
        sample_idx = torch.argsort(torch.rand(self.num_datapoints,device=self.device))[0:self.sample_len]
        library_idx = torch.argsort(torch.rand(self.num_datapoints,device=self.device))[0:self.library_len + self.sample_len]
        library_idx = library_idx[(library_idx.view(1, -1) != sample_idx.view(-1, 1)).all(dim=0)][0:self.library_len]
        
        return library_idx, sample_idx, self.X[library_idx],self.y[library_idx],\
                                        self.X[sample_idx], self.y[sample_idx]

