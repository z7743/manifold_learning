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

    def fit(self, X, 
            sample_len, 
            library_len, 
            exclusion_rad, 
            theta=None, tp=1, epochs=100, num_batches=32, optimizer="Adam", learning_rate=0.001, tp_policy="range", loss_mask_size=None):
        """
        Fits the model to the data.

        Args:
            X (numpy.ndarray): Input data.
            sample_len (int): Length of samples used for prediction.
            library_len (int): Length of library used for kNN search.
            exclusion_rad (int): Exclusion radius for kNN search.
            theta (float): S-map theta.
            tp (int): Time lag for prediction.
            epochs (int, optional): Number of training epochs, default is 100.
            num_batches (int, optional): Number of batches, default is 32.
            optimizer (str, optional): Optimizer to use, default is "Adam".
            learning_rate (float, optional): Learning rate for the optimizer, default is 0.001.
            tp_policy (str, optional): Batch sampling policy, default is "range". If "range" then the embedding will be optimized for the range of 1...tp and the total number of samplings calculated as num_batches*tp. If "fixed" then within one optimization cycle the embedding will be optimized only for one value of tp.
            loss_mask_size (int, optional): Number of compoments compared for one batch when the loss is calculated. Helps speed up convergence. If None all components used.
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
            self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0
            self.optimizer.zero_grad()

            for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
                subset_X_z = self.model(subset_X)
                subset_y_z = self.model(subset_y)
                sample_X_z = self.model(sample_X)
                sample_y_z = self.model(sample_y)

                loss = self.loss_fn(subset_idx, sample_idx,sample_X_z, sample_y_z, subset_X_z, subset_y_z, theta, exclusion_rad, loss_mask_size)

                loss /= num_batches
                loss.backward()
                total_loss += loss.item() 

            #model_weights = torch.cat([x for x in self.model.parameters()])
            #norm_weights = model_weights/torch.abs(model_weights).max(axis=1).values[:,None]

            #l2_norms = torch.norm(norm_weights, p=2)/20
            #if epoch > 20:
            #    l2_norms.backward()
            
            self.optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')#, L2: {l2_norms.item():.4f}')
            self.loss_history += [total_loss]

    def loss_fn(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad, loss_mask_size):
        dim = sample_X.shape[-1]

        if loss_mask_size is not None:
            rand_idx = torch.argsort(torch.rand(dim))[:loss_mask_size]
            sample_X = sample_X[:,:,rand_idx]
            sample_y = sample_y[:,:,rand_idx]
            subset_X = subset_X[:,:,rand_idx]
            subset_y = subset_y[:,:,rand_idx]

            dim = loss_mask_size


        ccm = (self._get_ccm_matrix_approx(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad))
        #ccm = -(self._get_ccm_matrix_approx(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad))
        mask = torch.eye(dim,dtype=bool,device=self.device)
        #mask_1 = torch.roll(torch.eye(dim,dtype=bool,device=self.device),1,0)
        #mask += mask_1
        ccm = ccm**2

        if self.subtract_corr:
            #corr = -(self._get_autoreg_matrix_approx(sample_y, sample_X))
            corr = torch.abs(self._get_autoreg_matrix_approx(sample_X,sample_y))**2
            if dim > 1:
                score = 1 + torch.abs(ccm[:,~mask]).mean() - (ccm[:,mask]).mean() + (corr[:,mask]).mean()
            else:
                score = 1 + (-ccm[:,0,0] + corr[:,0,0]).mean()
            return score
        else:
            if dim > 1:
                score = 1 + torch.abs(ccm[:,~mask]).mean() - (ccm[:,mask]).mean() 
            else:
                score = 1 + (-ccm[:,0,0]).mean()
            return score
        
    def loss_fn_(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad, loss_mask_size):
        dim = sample_X.shape[-1]

        if loss_mask_size is not None:
            rand_idx = torch.argsort(torch.rand(dim))[:loss_mask_size]
            sample_X = sample_X[:,:,rand_idx]
            sample_y = sample_y[:,:,rand_idx]
            subset_X = subset_X[:,:,rand_idx]
            subset_y = subset_y[:,:,rand_idx]

            dim = loss_mask_size


        ccm = torch.abs(self._get_ccm_matrix_approx(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad))
        #ccm = -(self._get_ccm_matrix_approx(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad))
        mask = torch.eye(dim,dtype=bool,device=self.device)
        #mask_1 = torch.roll(torch.eye(dim,dtype=bool,device=self.device),1,0)
        #mask += mask_1
        ccm = ccm.mean(axis=0)[None]
        if self.subtract_corr:
            #corr = -(self._get_autoreg_matrix_approx(sample_y, sample_X))
            corr = torch.abs(self._get_autoreg_matrix_approx(sample_X,sample_y))
            if dim > 1:
                score = -torch.log(
                    torch.exp((ccm[:,mask]))/
                    torch.exp((ccm[:,~mask]).reshape(-1,dim,dim-1)).sum(axis=2)/
                    torch.exp((corr[:,mask]))
                    ).mean()
                #score = 1 + torch.abs(ccm[:,~mask]).mean() - (ccm[:,mask]).mean() + (corr[:,mask]).mean()
            else:
                #score = 1 + (-ccm[:,0,0] + corr[:,0,0]).mean()
                score = 1 + (-ccm[0,0] + corr[0,0]).mean()
            return score
        else:
            if dim > 1:
                score = -torch.log(
                    torch.exp((ccm[mask]))/
                    torch.exp((ccm)).sum(axis=1)).mean()
                #score = 1 + torch.abs(ccm[:,~mask]).mean() - (ccm[:,mask]).mean() 
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

    def generate(self, X, p, exclusion_rad, theta=5, tp=1, device="cpu"):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32,device=device)
            p = torch.tensor(p, dtype=torch.float32,device=device)

            self.model.to(device)
            subset_X = self.model(X[:-tp])
            subset_y = self.model(X[tp:])
            sample_X = self.model(p)

            dim = subset_X.shape[-1]
            E_x = subset_X.shape[-2]
            E_y = subset_y.shape[-2]
            sample_size = sample_X.shape[0]
            subset_size = subset_X.shape[0]

            subset_X_t = torch.permute(subset_X,dims=(2,0,1))
            subset_y_t = torch.permute(subset_y,dims=(2,0,1))
            sample_X_t = torch.permute(sample_X,dims=(2,0,1))

            subset_idx = torch.arange(subset_size,device=device)[None]
            sample_idx = torch.arange(sample_size,device=device)[None]

            weights = self._get_local_weights(subset_X_t,sample_X_t,subset_idx, sample_idx, exclusion_rad, theta)
            W = weights.reshape(dim * sample_size, subset_size, 1)

            X = subset_X_t.unsqueeze(1).expand(dim, sample_size, subset_size, E_x)
            X = X.reshape(dim * sample_size, subset_size, E_x)

            Y = subset_y_t.unsqueeze(1).expand(dim, sample_size, subset_size, E_y)
            Y = Y.reshape(dim * sample_size, subset_size, E_y)

            X_intercept = torch.cat([torch.ones((dim * sample_size, subset_size, 1),device=device), X], dim=2)
            
            X_intercept_weighted = X_intercept * W
            Y_weighted = Y * W

            XTWX = torch.bmm(X_intercept_weighted.transpose(1, 2), X_intercept_weighted)
            XTWy = torch.bmm(X_intercept_weighted.transpose(1, 2), Y_weighted)
            #XTWX = torch.bmm(X_intercept.transpose(1, 2), X_intercept_weighted)
            #XTWy = torch.bmm(X_intercept.transpose(1, 2), Y_weighted)
            beta = torch.bmm(torch.inverse(XTWX), XTWy)
            #beta_ = beta.reshape(dim,dim,sample_size,*beta.shape[1:])

            X_ = sample_X_t
            X_ = X_.reshape(dim * sample_size, E_x)
            X_ = torch.cat([torch.ones((dim * sample_size, 1),device=device), X_], dim=1)
            X_ = X_.reshape(dim * sample_size, 1, E_x+1)
            
            #A = torch.einsum('abpij,bcpi->abcpj', beta, X_)
            #A = torch.permute(A[:,0],(2,3,1,0))

            A = torch.bmm(X_, beta).reshape(dim, sample_size, E_y)
            res = self.model.backward(torch.permute(A,(1,2,0)),)
            #A = torch.permute(A,(2,3,1,0))
            #rec = self.model.backward(res)
            self.model.to(self.device)
        return A.numpy(), res.numpy()


    def _get_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad):
        dim = sample_X.shape[-1]
        E_x = sample_X.shape[-2]
        E_y = sample_y.shape[-2]
        sample_size = sample_X.shape[0]
        subset_size = subset_X.shape[0]

        sample_X_t = sample_X.permute(2, 0, 1)
        subset_X_t = subset_X.permute(2, 0, 1)
        subset_y_t = subset_y.permute(2, 0, 1)
        
        weights = self._get_local_weights(subset_X_t,sample_X_t,subset_idx, sample_idx, exclusion_rad, theta)
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
        #XTWX = torch.bmm(X_intercept.transpose(1, 2), X_intercept_weighted)
        #XTWy = torch.bmm(X_intercept.transpose(1, 2), Y_weighted)
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
        
        #r_AB = self._get_batch_corr(A,B)
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
        

    def _get_batch_rv(self,A, B):
        num_points, num_dim, n_comp, _ = A.shape
        
        # 1. Center A and B over num_points
        mean_A = A.mean(dim=0, keepdim=True)
        mean_B = B.mean(dim=0, keepdim=True)
        
        A_c = A - mean_A
        B_c = B - mean_B
        
        # 2. Flatten (n_comp, n_comp) dimension
        A_flat = A_c.reshape(num_points, num_dim, n_comp*n_comp)
        B_flat = B_c.reshape(num_points, num_dim, n_comp*n_comp)
        
        # 3. Permute to (batch, num_points, num_dim)
        A_batched = A_flat.permute(2, 0, 1)
        B_batched = B_flat.permute(2, 0, 1)
        
        # 4. Compute Gram matrices in batch
        A_mat = torch.bmm(A_batched, A_batched.transpose(1,2))
        B_mat = torch.bmm(B_batched, B_batched.transpose(1,2))
        
        # 5. Compute required Frobenius inner products
        trace_AB = (A_mat * B_mat).sum(dim=(1,2))
        trace_AA = (A_mat * A_mat).sum(dim=(1,2))
        trace_BB = (B_mat * B_mat).sum(dim=(1,2))
        
        # 6. Compute RV
        RV_values = trace_AB / torch.sqrt(trace_AA * trace_BB)
        
        # 7. Reshape to (n_comp, n_comp)
        RV_mat = RV_values.view(n_comp, n_comp)
        
        return RV_mat[None]



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
        
    def _get_local_weights(self, lib, sublib, subset_idx, sample_idx, exclusion_rad, theta):
        dist = torch.cdist(sublib,lib)
        if theta == None:
            weights = torch.exp(-(dist))
        else:
            weights = torch.exp(-(theta*dist/dist.mean(axis=2)[:,:,None]))

        if exclusion_rad > 0:
            exclusion_matrix = (torch.abs(subset_idx - sample_idx.T) > exclusion_rad)
            weights = weights * exclusion_matrix
        
        return weights

    def get_loss_history(self):
        return self.loss_history
    

    def find_iterative_solution(self, X, sample_len, library_len, exclusion_rad, theta=None, tp=1, epochs=100, num_batches=32, tp_policy="range"):

        data = torch.tensor(X, device=self.device, dtype=torch.float32)
        

        if tp_policy == "fixed":
            dataset = RandomTpRangeSubsetDataset(data, sample_len, library_len, num_batches, torch.linspace(tp, tp+1 - 1e-5,num_batches,device=self.device).to(torch.int),device=self.device)
        elif tp_policy == "range":
            dataset = RandomTpRangeSubsetDataset(data, sample_len, library_len, num_batches, torch.linspace(1, tp+1 - 1e-5,num_batches,device=self.device).to(torch.int), device=self.device)
        else:
            pass #TODO: pass an exception

        dataloader = DataLoader(dataset, batch_size=1,pin_memory=False)
        
        
        #E = 2
        WW = torch.normal(0,1,(data.shape[1],10),device=self.device)
        for epoch in range(epochs):
            WW_list = []
            for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
                subset_X_z = subset_X[0] @ WW
                subset_y_z = subset_y[0] @ WW
                sample_X_z = sample_X[0] @ WW
                sample_y_z = sample_y[0] @ WW

                sample_size = sample_X.shape[1]
                subset_size = subset_X.shape[1]

                dist = torch.cdist(sample_X_z,subset_X_z,)
                weights = torch.exp(-(theta*dist/dist.mean(axis=1)[:,None]))
                
                if exclusion_rad > 0:
                    exclusion_matrix = (torch.abs(subset_idx - sample_idx.T) > exclusion_rad)
                    weights = weights * exclusion_matrix

                W = torch.sqrt(weights.unsqueeze(2))

                X = subset_X_z.unsqueeze(0).expand(sample_size, subset_size, WW.shape[1])

                Y = subset_y_z.unsqueeze(0).expand(sample_size, subset_size, WW.shape[1])

                #X = torch.cat([torch.ones((sample_size, subset_size, 1),device=self.device), X], dim=2)
                
                X_weighted = X * W
                Y_weighted = Y * W

                # Compute the weighted cross-covariance matrix (XTWy)
                XTWX = torch.bmm(X_weighted.transpose(1, 2), X_weighted)  # (120, 256, 256)
                XTWy = torch.bmm(X_weighted.transpose(1, 2), Y_weighted)  # (120, 256, 256)

                # Compute the inverse of the weighted covariance matrix for X (XTWX)
                XTWX_inv = torch.inverse(XTWX)  # (120, 256, 256)

                # Generalized eigenvalue problem for maximum correlation
                left_matrix = torch.bmm(XTWy, torch.inverse(torch.bmm(Y_weighted.transpose(1, 2), Y_weighted)))  # (120, 256, 256)
                
                # Perform eigen decomposition (only keeping the needed components)
                eigvals, eigvecs = torch.linalg.eigh(torch.bmm(XTWX_inv, left_matrix))  # (120, 256, 256)

                # Sorting eigenvectors based on eigenvalues
                sorted_indices = torch.argsort(eigvals, descending=True)

                # Extract the first 10 sorted eigenvectors for each sample (batch)
                beta = torch.stack([eigvecs[i][:, sorted_indices[i][:10]] for i in range(eigvecs.shape[0])])


                X_ = sample_X_z.unsqueeze(1)
                #X_ = torch.cat([torch.ones((sample_size, 1, 1),device=self.device), X_], dim=2)
                
                A = torch.bmm(X_, beta).squeeze(1)
                #A = (A-A.mean(axis=0))/A.std(axis=0)

                B = sample_y[0]
                #B = (B-B.mean(axis=0))/B.std(axis=0)


                XtX = torch.matmul(B.T, B)
                XtX_inv = torch.inverse(XtX)
                Xty = torch.matmul(B.T, A)

                WW_ = torch.matmul(XtX_inv, Xty)

                WW_list += [WW_]
            WW__ = torch.stack(WW_list).mean(axis=0)
            #print(WW_.mean())
            WW__ = (WW__-WW__.mean())/WW__.std()
            #WW = WW__
            WW =  WW__
            #WW = (WW-WW.mean())/WW.std()

            print(self._get_batch_corr(A[:,:,None,None],sample_y_z[:,:,None,None]).mean())
        return WW.cpu().detach().numpy()


    def find_iterative_solution_(self, X, sample_len, library_len, exclusion_rad, theta=None, tp=1, epochs=100, num_batches=32, tp_policy="range"):
        data = torch.tensor(X, device=self.device, dtype=torch.float32)
        
        if tp_policy == "fixed":
            dataset = RandomTpRangeSubsetDataset(data, sample_len, library_len, num_batches, torch.linspace(tp, tp + 1 - 1e-5, num_batches, device=self.device).to(torch.int), device=self.device)
        elif tp_policy == "range":
            dataset = RandomTpRangeSubsetDataset(data, sample_len, library_len, num_batches, torch.linspace(1, tp + 1 - 1e-5, num_batches, device=self.device).to(torch.int), device=self.device)
        else:
            pass  # TODO: pass an exception

        dataloader = DataLoader(dataset, batch_size=1, pin_memory=False)
        
        WW = torch.normal(0, 1, (data.shape[1], 5), device=self.device)
        for epoch in range(epochs):
            WW_list = []
            for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
                subset_X_z = subset_X[0] @ WW
                subset_y_z = subset_y[0] @ WW
                sample_X_z = sample_X[0] @ WW
                sample_y_z = sample_y[0] @ WW

                sample_size = sample_X.shape[1]
                subset_size = subset_X.shape[1]

                dist = torch.cdist(sample_X_z, subset_X_z)
                weights = torch.exp(-(theta * dist / dist.mean(axis=1)[:, None]))

                if exclusion_rad > 0:
                    exclusion_matrix = (torch.abs(subset_idx - sample_idx.T) > exclusion_rad)
                    weights = weights * exclusion_matrix

                W = torch.sqrt(weights.unsqueeze(2))

                X = subset_X_z.unsqueeze(0).expand(sample_size, subset_size, WW.shape[1])
                Y = subset_y_z.unsqueeze(0).expand(sample_size, subset_size, WW.shape[1])

                Y_centered = Y - X.mean(dim=1, keepdim=True)
                X_centered = X - X.mean(dim=1, keepdim=True)

                _, _, V_x = torch.svd(X_centered, some=True)

                X_pca = torch.matmul(X_centered, V_x[:, :, :]) 
                Y_pca = torch.matmul(Y_centered, V_x[:, :, :]) 

            
                X_pca = torch.cat([torch.ones((sample_size, subset_size, 1), device=self.device), X_pca], dim=2)

                X_weighted = X_pca * W
                Y_weighted = Y_pca * W

                
                XTWX = torch.bmm(X_weighted.transpose(1, 2), X_weighted)
                XTWy = torch.bmm(X_weighted.transpose(1, 2), Y_weighted)
                beta = torch.bmm(torch.inverse(XTWX), XTWy)

                X_ = sample_X_z.unsqueeze(1)
                #X_ = X_ - X_.mean(dim=1, keepdim=True)
                X_ = torch.matmul(X_, V_x[:, :, :]) 

                X_ = torch.cat([torch.ones((sample_size, 1, 1), device=self.device), X_], dim=2)

                A = torch.bmm(X_, beta).squeeze(1)
                A = torch.matmul(A.unsqueeze(1), torch.linalg.pinv(V_x,)).squeeze(1)
                B = sample_y[0]

                XtX = torch.matmul(B.T, B)
                XtX_inv = torch.inverse(XtX)
                Xty = torch.matmul(B.T, A)

                WW_ = torch.matmul(XtX_inv, Xty)



                WW_list += [WW_]
            WW__ = torch.stack(WW_list).mean(axis=0)
            WW__ = (WW__ - WW__.mean()) / WW__.std()
            WW =WW__

            print(self._get_batch_rmse(A[:, :, None, None], sample_y_z[:, :, None, None]).mean())
        return WW.cpu().detach().numpy()
