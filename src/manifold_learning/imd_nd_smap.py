import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .linear_projection import LinearProjectionNDim
from .data_samplers import RandomTpRangeSubsetDataset

class IMD_nD_smap:

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

    def fit(self, X, sample_len, library_len, exclusion_rad, omega=None, tp=1, epochs=100, num_batches=32, optimizer="Adam", learning_rate=0.001, tp_policy="range"):
        """
        Fits the model to the data.

        Args:
            X (numpy.ndarray): Input data.
            sample_len (int): Length of samples used for prediction.
            library_len (int): Length of library used for kNN search.
            exclusion_rad (int): Exclusion radius for kNN search.
            omega (float): S-map omega.
            tp (int): Time lag for prediction.
            epochs (int, optional): Number of training epochs, default is 100.
            num_batches (int, optional): Number of batches, default is 32.
            optimizer (str, optional): Optimizer to use, default is "Adam".
            learning_rate (float, optional): Learning rate for the optimizer, default is 0.001.
            tp_policy (str, optional): Batch sampling policy, default is "range". If "range" then the embedding will be optimized for the range of 1...tp and the total number of samplings calculated as num_batches*tp. If "fixed" then within one optimization cycle the embedding will be optimized only for one value of tp.
        """
        X = torch.tensor(X,requires_grad=True, device=self.device, dtype=torch.float32)

        if tp_policy == "fixed":
            dataset = RandomTpRangeSubsetDataset(X, sample_len, library_len, num_batches, torch.linspace(tp, tp+1 - 1e-5,num_batches,device=self.device).to(torch.int),device=self.device)
        elif tp_policy == "range":
            dataset = RandomTpRangeSubsetDataset(X, sample_len, library_len, num_batches, torch.linspace(1, tp+1 - 1e-5,num_batches,device=self.device).to(torch.int), device=self.device)
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

                loss = self.loss_fn(subset_idx, sample_idx,sample_X_z, sample_y_z, subset_X_z, subset_y_z, omega, exclusion_rad)

                loss /= num_batches
                loss.backward()
                total_loss += loss.item() 

            self.optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')
            self.loss_history += [total_loss]

    def loss_fn(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, omega, exclusion_rad):
        dim = sample_X.shape[-1]
        ccm = torch.abs(self._get_ccm_matrix_approx(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, omega, exclusion_rad))
        #ccm = -(self._get_ccm_matrix_approx(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad))
        mask = torch.eye(dim,dtype=bool,device=self.device)

        if self.subtract_corr:
            #corr = -(self._get_autoreg_matrix_approx(sample_y, sample_X))
            corr = torch.abs(self._get_autoreg_matrix_approx(sample_X, sample_y))
            if dim > 1:
                score = 1 + (torch.mean(ccm[:,~mask].reshape(-1,dim,dim-1),axis=2)/2 + \
                             torch.mean(ccm[:,~mask].reshape(-1,dim-1,dim),axis=1)/2 +\
                           -ccm[:,mask]**2 + corr[:,mask]**2).mean()
            else:
                score = 1 + (-ccm[:,0,0] + corr[:,0,0]).mean()
            return score
        else:
            if dim > 1:
                score = 1 + (torch.mean(ccm[:,~mask].reshape(-1,dim,dim-1),axis=2)/2 + \
                             torch.mean(ccm[:,~mask].reshape(-1,dim-1,dim),axis=1)/2 + \
                           -ccm[:,mask]**2).mean()
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

    def _get_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, omega, exclusion_rad):
        dim = sample_X.shape[-1]
        E_x = sample_X.shape[-2]
        E_y = sample_y.shape[-2]
        sample_size = sample_X.shape[0]
        subset_size = subset_X.shape[0]

        sample_X_t = sample_X.permute(2, 0, 1)
        subset_X_t = subset_X.permute(2, 0, 1)
        subset_y_t = subset_y.permute(2, 0, 1)
        
        weights = self._get_local_weights(subset_X_t,sample_X_t,subset_idx, sample_idx, exclusion_rad, omega)
        W = weights.unsqueeze(1).expand(dim, dim, sample_size, subset_size).reshape(dim * dim * sample_size, subset_size, 1)

        X = subset_X_t.unsqueeze(1).unsqueeze(1).expand(dim, dim, sample_size, subset_size, E_x)
        X = X.reshape(dim * dim * sample_size, subset_size, E_x)

        Y = subset_y_t.unsqueeze(1).unsqueeze(0).expand(dim, dim, sample_size, subset_size, E_y)
        Y = Y.reshape(dim * dim * sample_size, subset_size, E_y)

        X_intercept = torch.cat([torch.ones((dim * dim * sample_size, subset_size, 1),device=self.device), X], dim=2)
        
        X_intercept_weighted = X_intercept * W
        Y_weighted = Y * W

        XTWX = torch.bmm(X_intercept_weighted.transpose(1, 2), X_intercept_weighted)
        XTWy = torch.bmm(X_intercept_weighted.transpose(1, 2), Y_weighted)
        beta = torch.bmm(torch.inverse(XTWX), XTWy)
        #beta_ = beta.reshape(dim,dim,sample_size,*beta.shape[1:])

        X_ = sample_X_t.unsqueeze(1).expand(dim, dim, sample_size, E_x)
        X_ = X_.reshape(dim * dim * sample_size, E_x)
        X_ = torch.cat([torch.ones((dim * dim * sample_size, 1),device=self.device), X_], dim=1)
        X_ = X_.reshape(dim * dim * sample_size, 1, E_x+1)
        
        #A = torch.einsum('abpij,bcpi->abcpj', beta, X_)
        #A = torch.permute(A[:,0],(2,3,1,0))

        A = torch.bmm(X_, beta).reshape(dim, dim, sample_size, E_y)
        A = torch.permute(A,(2,3,1,0))

        B = sample_y.unsqueeze(-1).expand(sample_size, E_y, dim, dim)
        #TODO: test whether B = sample_y.unsqueeze(-2).expand(sample_size, E_y, dim, dim)
        
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
        
    def _get_local_weights(self, lib, sublib, subset_idx, sample_idx, exclusion_rad, omega):
        dist = torch.cdist(sublib,lib)
        if omega == None:
            weights = torch.exp(-(dist))
        else:
            weights = torch.exp(-(omega*dist/dist.mean(axis=2)[:,:,None]))

        if exclusion_rad > 0:
            exclusion_matrix = (torch.abs(subset_idx - sample_idx.T) > exclusion_rad)
            weights = weights * exclusion_matrix
        
        return weights

    def get_loss_history(self):
        return self.loss_history
    
