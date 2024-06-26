import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class LinearProjectionNDim(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, device="cuda",random_state=None):
        """
        Initializes the linear projection module.
        
        Args:
            input_dim (int): The dimension of the input data.
            embed_dim (int): The dimension of the embedding.
            output_dim (int): The dimension of the output projection.
            device (str, optional): Device to place the model on, default is "cuda".
            random_state (int): Ignored if None.
        """
        super(LinearProjectionNDim, self).__init__()
        self.device = device

        self.embed_dim = embed_dim
        self.output_dim = output_dim

        if random_state != None:
            torch.manual_seed(random_state)
        self.model = nn.Linear(input_dim, output_dim*embed_dim, bias=False,device=self.device,)
        #self.model = nn.Sequential(nn.Linear(input_dim, input_dim, bias=True,device=self.device,),
        #                           nn.Sigmoid(),
        #                            nn.Linear(input_dim, input_dim, bias=True,device=self.device,),
        #                           nn.Sigmoid(),
        #                           nn.Linear(input_dim, output_dim*embed_dim, bias=False,device=self.device,)
        #                           )

    def forward(self, x):
        """
        Forward pass of the module.
        
        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, input_dim)`.
        
        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, embed_dim, output_dim)`.
        """
        x = self.model(x).reshape(-1,self.embed_dim,self.output_dim)

        return x
    
    def backward(self, y):
        """
        Backward pass using the pseudoinverse.
        
        Args:
            y (torch.Tensor): Output tensor of shape `(batch_size, embed_dim, output_dim)`.
        
        Returns:
            torch.Tensor: Input tensor of shape `(batch_size, input_dim)`.
        """
        batch_size = y.shape[0]
        y_flat = y.reshape(batch_size, -1)

        # Compute the pseudoinverse of the weight matrix
        weight = self.model.weight
        weight_pinv = torch.pinverse(weight)#.to(self.model.device)

        # Compute the input using the pseudoinverse
        x_reconstructed = y_flat @ weight_pinv.T

        return x_reconstructed

    
    def get_weights(self):
        """
        Retrieves the weights of the model.
        
        Returns:
            numpy.ndarray: Weights of the model reshaped and permuted.
        """
        return torch.permute(self.model.weight.reshape(-1,self.embed_dim,self.output_dim), dims=(0,2,1)).cpu().detach().numpy()



class IMD_nD:

    def __init__(self, input_dim, embed_dim, n_components, subtract_corr=True,device="cuda",random_state=None):
        """
        Initializes the IMD_nD model.
        
        Args:
            input_dim (int): Dimension of the input data.
            embed_dim (int): Dimension of the embedding of the component.
            n_components (int): Number of components.
            subtract_corr (bool, optional): Whether to subtract correlation, default is True.
            device (str, optional): Device to place the model on, default is "cuda".
            random_state (int): Ignored if None.
        """
        self.device = device

        self.model = LinearProjectionNDim(input_dim, embed_dim, n_components, device,random_state)
        self.optimizer_name = None
        self.learning_rate = None
        self.optimizer = None
        self.subtract_corr = subtract_corr
        self.loss_history = []

    def fit(self, X, sample_len, library_len, exclusion_rad, nbrs_num, tp=1, epochs=100, num_batches=32, optimizer="Adam", learning_rate=0.001, tp_policy="range"):
        """
        Fits the model to the data.

        Args:
            X (numpy.ndarray): Input data.
            sample_len (int): Length of samples used for prediction.
            library_len (int): Length of library used for kNN search.
            exclusion_rad (int): Exclusion radius for kNN search.
            nbrs_num (int): Number of neighbors for kNN search.
            tp (int): Time lag for prediction.
            epochs (int, optional): Number of training epochs, default is 100.
            num_batches (int, optional): Number of batches, default is 32.
            optimizer (str, optional): Optimizer to use, default is "Adam".
            learning_rate (float, optional): Learning rate for the optimizer, default is 0.001.
            tp_policy (str, optional): Batch sampling policy, default is "range". If "range" then the embedding will be optimized for the range of 1...tp and the total number of samplings calculated as num_batches*tp. If "fixed" then within one optimization cycle the embedding will be optimized only for one value of tp.
        """
        X = torch.tensor(X,requires_grad=True, device=self.device, dtype=torch.float32)

        if tp_policy == "fixed":
            dataset = RandomTpRangeSubsetDataset(X, sample_len, library_len, num_batches, torch.arange(tp,tp+1),device=self.device)
        elif tp_policy == "range":
            dataset = RandomTpRangeSubsetDataset(X, sample_len, library_len, num_batches, torch.arange(1,tp+1), device=self.device)
        else:
            pass #TODO: pass an exception

        dataloader = DataLoader(dataset, batch_size=1,pin_memory=False)

        # Reinitialize optimizer if parameters changed
        if (self.learning_rate != learning_rate) or (self.optimizer_name != optimizer):
            self.learning_rate = learning_rate
            self.optimizer_name = optimizer
            self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=learning_rate,)

        for epoch in range(epochs):
            total_loss = 0
            self.optimizer.zero_grad()

            for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
                subset_X_z = self.model(subset_X)
                subset_y_z = self.model(subset_y)
                sample_X_z = self.model(sample_X)
                sample_y_z = self.model(sample_y)

                loss = self.loss_fn(subset_idx, sample_idx,sample_X_z, sample_y_z, subset_X_z, subset_y_z, nbrs_num, exclusion_rad)
                
                if tp_policy == "range":
                    loss /= tp * num_batches
                elif tp_policy == "fixed":
                    loss /= num_batches
                loss.backward()
                total_loss += loss.item() 

            self.optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')
            self.loss_history += [total_loss]

    def loss_fn(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad):
        dim = sample_X.shape[-1]
        ccm = torch.abs(self._get_ccm_matrix_approx(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad))
        #ccm = -(self._get_ccm_matrix_approx(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad))
        mask = torch.eye(dim,dtype=bool,device=self.device)

        if self.subtract_corr:
            #corr = -(self._get_autoreg_matrix_approx(sample_y, sample_X))
            corr = torch.abs(self._get_autoreg_matrix_approx(sample_y, sample_X))
            if dim > 1:
                score = 1 + (torch.mean(ccm[:,~mask].reshape(-1,dim,dim-1),axis=2)/2 + \
                             torch.mean(ccm[:,~mask].reshape(-1,dim-1,dim),axis=1)/2).mean() +\
                           (-ccm[:,mask]**2 + corr[:,mask]**2).mean()
            else:
                score = 1 + (-ccm[:,0,0] + corr[:,0,0]).mean()
            return score
        else:
            if dim > 1:
                score = 1 + (torch.mean(ccm[:,~mask].reshape(-1,dim,dim-1),axis=2)/2 + \
                             torch.mean(ccm[:,~mask].reshape(-1,dim-1,dim),axis=1)/2).mean() + \
                           (-ccm[:,mask]**2).mean()
            else:
                score = 1 + (-ccm[:,0,0]).mean()
            return score
        
    
    def predict(self, X):
        """
        Calculates embeddings using the trained model.
        
        Args:
            X (numpy.ndarray): Input data.
        
        Returns:
            numpy.ndarray: Predicted outputs.
        """
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32,device=self.device)
            outputs = torch.permute(self.model(inputs),dims=(0,2,1)) #Easier to interpret
        return outputs.cpu().numpy()

    def generate(self, X, nbrs_num, exclusion_rad, tp=1, device="cpu"):
        """
        Generates the prediction using the model.
        
        Args:
            X (numpy.ndarray): Input data.
            nbrs_num (int): Number of neighbors for kNN search.
            exclusion_rad (int): Exclusion radius for kNN search.
            tp (int, optional): Time lag for prediction, default is 1.
        
        Returns:
            tuple: Generated results and the reconstructed input.
        """
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32,device=device)
            self.model.to(device)
            X_lib_z = self.model(X)
            lib = torch.permute(X_lib_z,dims=(2,0,1))

            dist = torch.cdist(lib,lib[:,:-tp])
            indices = torch.topk(dist, nbrs_num + 2 * exclusion_rad, largest=False)[1]

            mask = (indices >= (torch.arange(X_lib_z.shape[0],device=device) + exclusion_rad)[None,:,None]) | (indices <= (torch.arange(X_lib_z.shape[0],device=device) - exclusion_rad)[None,:,None])
            cumsum_mask = mask.cumsum(dim=2)
            selector = cumsum_mask <= nbrs_num
            selector = selector * mask
            indices = indices[selector].view(mask.shape[0],mask.shape[1],nbrs_num)

            indices += tp
            subset_pred_indexed = lib[torch.arange(X_lib_z.shape[-1],device=device)[:,None,None],indices]
            res = torch.permute(subset_pred_indexed.mean(axis=2), dims=(1,2,0))
            rec = self.model.backward(res)
            self.model.to(self.device)
        return res, rec

    def _get_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad):
        dim = sample_X.shape[-1]
        E = sample_X.shape[-2]
        
        indices = self._get_nbrs_indices(torch.permute(subset_X,(2,0,1)),torch.permute(sample_X,(2,0,1)), nbrs_num, subset_idx, sample_idx, exclusion_rad)
        I = indices.reshape(dim,-1).T 
        
        #subset_pred_indexed = subset_y[I[:, None,None, :],torch.arange(E,device=self.device)[:,None,None], torch.arange(dim,device=self.device)[None,:,None]]
        subset_pred_indexed = torch.permute(subset_y[I],(0,2,3,1))
        ## No gradient for the indexed variable ##
        #subset_pred_indexed = subset_pred_indexed.detach()

        A = subset_pred_indexed.reshape(-1, nbrs_num, E, dim, dim).mean(axis=1)
        B = sample_y[:,:,:,None].expand(sample_y.shape[0], E, dim, dim)
        
        r_AB = self._get_batch_corr(A,B)
        return r_AB
    
    def _get_autoreg_matrix_approx(self, A, B):
        dim = A.shape[-1]
        E = A.shape[-2]
        
        A = A[:,:,None,:].expand(-1, E, dim, dim)
        B = B[:,:,:,None].expand(-1, E, dim, dim)

        r_AB = self._get_batch_corr(A,B)
        return r_AB
    
    def _get_batch_corr(self,A, B):
        mean_A = torch.mean(A,axis=0)
        mean_B = torch.mean(B,axis=0)
        
        sum_AB = torch.sum((A - mean_A[None,:,:]) * (B - mean_B[None,:,:]),axis=0)
        sum_AA = torch.sum((A - mean_A[None,:,:]) ** 2,axis=0)
        sum_BB = torch.sum((B - mean_B[None,:,:]) ** 2,axis=0)
        
        r_AB = sum_AB / torch.sqrt(sum_AA * sum_BB)
        return r_AB
    
    def _get_batch_cosine_similarity(self,A, B):
        """
        Computes the batch-wise cosine similarity between two 4D tensors A and B.
        
        Args:
        A, B: Tensors of shape [num points, num dims, num components, num components].
        
        Returns:
        Tensor of cosine similarities with shape [num dims, num components, num components].
        """
        # Compute the norms of A and B along the num points axis
        norm_A = torch.norm(A, p=2, dim=0, keepdim=True)
        norm_B = torch.norm(B, p=2, dim=0, keepdim=True)
        
        # Avoid division by zero
        norm_A = torch.where(norm_A == 0, torch.ones_like(norm_A), norm_A)
        norm_B = torch.where(norm_B == 0, torch.ones_like(norm_B), norm_B)
        
        # Normalize A and B
        A_normalized = A / norm_A
        B_normalized = B / norm_B
        
        # Compute the dot product between normalized A and B
        dot_product = torch.sum(A_normalized * B_normalized, dim=0)
        
        return dot_product

    def _get_batch_rmse(self, A, B):
        """
        Computes the batch-wise Root Mean Square Error (RMSE) between two 4D tensors A and B.
        
        Args:
        A, B: Tensors of shape [num points, num dims, num components, num components].
        
        Returns:
        Tensor of RMSE values with shape [num dims, num components, num components].
        """
        # Compute the squared differences between A and B
        squared_diff = (A - B) ** 2
        
        # Compute the mean of the squared differences along the num points axis
        mean_squared_diff = torch.mean(squared_diff, dim=0)
        
        # Compute the square root of the mean squared differences
        rmse = torch.sqrt(mean_squared_diff)
        
        return rmse

    def _get_nbrs_indices(self, lib, sublib, n_nbrs, subset_idx, sample_idx, exclusion_rad):
        dist = torch.cdist(sublib,lib)
        indices = torch.topk(dist, n_nbrs + 2*exclusion_rad, largest=False)[1]
        if exclusion_rad > 0:
            
            mask = ~((subset_idx[0][indices] <= sample_idx[0][:,None]+exclusion_rad) & (subset_idx[0][indices] >= sample_idx[0][:,None]-exclusion_rad))
            cumsum_mask = mask.cumsum(dim=2)
            selector = cumsum_mask <= n_nbrs
            selector = selector * mask
            
            indices_exc = indices[selector].view(mask.shape[0],mask.shape[1],n_nbrs)
            return indices_exc
        else:
            return indices

    def get_loss_history(self):
        return self.loss_history
    

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
        return self.tp_range.shape[0] * self.num_batches #Temporary solution to sample number of samples
    
    def __getitem__(self, idx):
        sample_idx = torch.argsort(torch.rand(self.num_datapoints-self.tp_max-1,device=self.device))[0:self.sample_size]
        subset_idx = torch.argsort(torch.rand(self.num_datapoints-self.tp_max-1,device=self.device))[0:self.subset_size + self.sample_size]
        subset_idx = subset_idx[(subset_idx.view(1, -1) != sample_idx.view(-1, 1)).all(dim=0)][0:self.subset_size]
        
        return subset_idx, sample_idx, self.X[subset_idx],self.X[subset_idx+self.tp_range[idx%self.tp_range.shape[0]]],\
                                        self.X[sample_idx], self.X[sample_idx+self.tp_range[idx%self.tp_range.shape[0]]]


class MCM_reg:

    def __init__(self, input_dim, embed_dim, subtract_corr=True,device="cuda",random_state=None):
        """
        Initializes the IMD_nD model.
        
        Args:
            input_dim (int): Dimension of the input data.
            embed_dim (int): Dimension of the embedding of the component.
            subtract_corr (bool, optional): Whether to subtract correlation, default is True.
            device (str, optional): Device to place the model on, default is "cuda".
            random_state (int): Ignored if None.
        """
        self.device = device

        self.model = LinearProjectionNDim(input_dim, embed_dim, 1, device,random_state)
        self.optimizer_name = None
        self.learning_rate = None
        self.optimizer = None
        self.subtract_corr = subtract_corr
        self.loss_history = []

    def fit(self, X, y, sample_len, library_len, exclusion_rad, nbrs_num, epochs=100, num_batches=32, optimizer="Adam", learning_rate=0.001):
        """
        Fits the model to the data.

        Args:
            X (numpy.ndarray): Input data.
            sample_len (int): Length of samples used for prediction.
            library_len (int): Length of library used for kNN search.
            exclusion_rad (int): Exclusion radius for kNN search.
            nbrs_num (int): Number of neighbors for kNN search.
            epochs (int, optional): Number of training epochs, default is 100.
            num_batches (int, optional): Number of batches, default is 32.
            optimizer (str, optional): Optimizer to use, default is "Adam".
            learning_rate (float, optional): Learning rate for the optimizer, default is 0.001.
            tp_policy (str, optional): Batch sampling policy, default is "range". If "range" then the embedding will be optimized for the range of 1...tp and the total number of samplings calculated as num_batches*tp. If "fixed" then within one optimization cycle the embedding will be optimized only for one value of tp.
        """
        X = torch.tensor(X,requires_grad=True, device=self.device, dtype=torch.float32)
        y = torch.tensor(y,requires_grad=True, device=self.device, dtype=torch.float32)

        dataset = RandomXYSubsetDataset(X, y, sample_len, library_len, num_batches, device=self.device)
       
        dataloader = DataLoader(dataset, batch_size=1,pin_memory=False)

        if (self.learning_rate != learning_rate) or (self.optimizer_name != optimizer):
            self.learning_rate = learning_rate
            self.optimizer_name = optimizer
            self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=learning_rate,)

        for epoch in range(epochs):
            total_loss = 0
            self.optimizer.zero_grad()

            for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
                subset_X_z = self.model(subset_X)
                subset_y_z = subset_y[0][:,:,None]
                sample_X_z = self.model(sample_X)
                sample_y_z = sample_y[0][:,:,None]

                loss = self.loss_fn(subset_idx, sample_idx, sample_X_z, sample_y_z, subset_X_z, subset_y_z, nbrs_num, exclusion_rad)
                
                loss /= num_batches
                loss.backward()
                total_loss += loss.item() 

            self.optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')
            self.loss_history += [total_loss]

    def predict(self, X):
        """
        Calculates embeddings using the trained model.
        
        Args:
            X (numpy.ndarray): Input data.
        
        Returns:
            numpy.ndarray: Predicted outputs.
        """
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32,device=self.device)
            outputs = torch.permute(self.model(inputs),dims=(0,2,1)) #Easier to interpret
        return outputs.cpu().numpy()


    def loss_fn(self, subset_idx, sample_idx, sample_X_z, sample_y_z, subset_X_z, subset_y_z, nbrs_num, exclusion_rad):
        dim = sample_X_z.shape[-1]
        ccm = torch.abs(self.get_ccm_matrix_approx(subset_idx, sample_idx, sample_X_z, sample_y_z, subset_X_z, subset_y_z, nbrs_num, exclusion_rad))

        if self.subtract_corr:
            corr = torch.abs(self.get_autoreg_matrix_approx(sample_y_z, sample_X_z))
            score = 1 + (-ccm[:,0,0] + corr[:,0,0]).mean()
            return score
        else:
            score = 1 + (-ccm[:,0,0]).mean()
            return score

    def get_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X_z, sample_y_z, subset_X_z, subset_y_z, nbrs_num, exclusion_rad):
        dim_X = sample_X_z.shape[-1]
        E_X = sample_X_z.shape[-2]
        dim_y = sample_y_z.shape[-1]
        E_y = sample_y_z.shape[-2]
        
        indices = self.get_nbrs_indices(torch.permute(subset_y_z,(2,0,1)),torch.permute(sample_y_z,(2,0,1)), nbrs_num, subset_idx, sample_idx, exclusion_rad)
        I = indices.reshape(dim_y,-1).T 
        
        subset_pred_indexed = subset_X_z[I[:, None,None, :],torch.arange(E_X,device=self.device)[:,None,None], torch.arange(dim_X,device=self.device)[None,:,None]]
        
        A = subset_pred_indexed.reshape(-1, nbrs_num, E_X, dim_X, dim_y).mean(axis=1)
        B = sample_X_z[:,:,:,None].expand(sample_X_z.shape[0], E_X, dim_X, dim_y)
        
        r_AB = self.get_batch_corr(A,B)
        return r_AB
    
    def get_autoreg_matrix_approx(self, A, B):
        dim_A = A.shape[-1]
        E_A = A.shape[-2]
        dim_B = B.shape[-1]
        E_B = B.shape[-2]
        
        A = A[:,:,None,:].expand(-1, E_A, dim_B, dim_A)
        B = B[:,:,:,None].expand(-1, E_B, dim_B, dim_A)

        r_AB = self.get_batch_corr(A,B)
        return r_AB
    
    def get_batch_corr(self,A, B):
        mean_A = torch.mean(A,axis=0)
        mean_B = torch.mean(B,axis=0)
        
        sum_AB = torch.sum((A - mean_A[None,:,:]) * (B - mean_B[None,:,:]),axis=0)
        sum_AA = torch.sum((A - mean_A[None,:,:]) ** 2,axis=0)
        sum_BB = torch.sum((B - mean_B[None,:,:]) ** 2,axis=0)
        
        r_AB = sum_AB / torch.sqrt(sum_AA * sum_BB)
        return r_AB
    
    def get_batch_cosine_similarity(self,A, B):
        """
        Computes the batch-wise cosine similarity between two 4D tensors A and B.
        
        Args:
        A, B: Tensors of shape [num points, num dims, num components, num components].
        
        Returns:
        Tensor of cosine similarities with shape [num dims, num components, num components].
        """
        # Compute the norms of A and B along the num points axis
        norm_A = torch.norm(A, p=2, dim=0, keepdim=True)
        norm_B = torch.norm(B, p=2, dim=0, keepdim=True)
        
        # Avoid division by zero
        norm_A = torch.where(norm_A == 0, torch.ones_like(norm_A), norm_A)
        norm_B = torch.where(norm_B == 0, torch.ones_like(norm_B), norm_B)
        
        # Normalize A and B
        A_normalized = A / norm_A
        B_normalized = B / norm_B
        
        # Compute the dot product between normalized A and B
        dot_product = torch.sum(A_normalized * B_normalized, dim=0)
        
        return dot_product

    def get_nbrs_indices(self, lib, sublib, n_nbrs, subset_idx, sample_idx, exclusion_rad):
        dist = torch.cdist(sublib,lib)
        indices = torch.topk(dist, n_nbrs + 2*exclusion_rad, largest=False)[1]
        if exclusion_rad > 0:
            
            mask = ~((subset_idx[0][indices] <= sample_idx[0][:,None]+exclusion_rad) & (subset_idx[0][indices] >= sample_idx[0][:,None]-exclusion_rad))
            cumsum_mask = mask.cumsum(dim=2)
            selector = cumsum_mask <= n_nbrs
            selector = selector * mask
            
            indices_exc = indices[selector].view(mask.shape[0],mask.shape[1],n_nbrs)
            return indices_exc
        else:
            return indices

    def get_loss_history(self):
        return self.loss_history
    

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

