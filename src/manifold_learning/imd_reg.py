import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .linear_projection import LinearProjectionNDim
from .data_samplers import RandomXYSubsetDataset

class IMD_reg:

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
    
