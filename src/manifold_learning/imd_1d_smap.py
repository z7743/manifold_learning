import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .linear_projection import LinearProjectionNDim
from .data_samplers import RandomTimeDelaySubsetDataset
from .utils.utils import get_td_embedding_torch


class IMD_1D_smap:

    def __init__(self, input_dim, n_components, embed_dim, embed_lag, subtract_corr=True,device="cuda",random_state=None):
        self.device = device

        self.model = LinearProjectionNDim(input_dim, 1, n_components, device,random_state)
        self.optimizer_name = None
        self.learning_rate = None
        self.optimizer = None
        self.subtract_corr = subtract_corr
        self.loss_history = []
        self.n_components = n_components
        self.embed_lag = embed_lag
        self.embed_dim = embed_dim

    def fit(self, X, sample_len, library_len, exclusion_rad, theta=None, tp=1, epochs=100, num_batches=32, 
            optimizer="Adam", learning_rate=0.001, tp_policy="range",loss_mask_size=None):
        embed_lag = self.embed_lag
        embed_dim = self.embed_dim

        X = torch.tensor(X,requires_grad=True, device=self.device, dtype=torch.float32)

        if tp_policy == "fixed":
            dataset = RandomTimeDelaySubsetDataset(X, sample_len, library_len, embed_dim, embed_lag, num_batches, torch.linspace(tp, tp+1 - 1e-5,num_batches,device=self.device).to(torch.int),device=self.device)
        elif tp_policy == "range":
            dataset = RandomTimeDelaySubsetDataset(X, sample_len, library_len, embed_dim, embed_lag, num_batches, torch.linspace(1, tp+1 - 1e-5,num_batches,device=self.device).to(torch.int), device=self.device)
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
                subset_idx = subset_idx[:,0]
                sample_idx = sample_idx[:,0]

                subset_X_z = self.model(subset_X.reshape(1, embed_dim * library_len, -1)).reshape(embed_dim, library_len, self.n_components)
                subset_y_z = self.model(subset_y[:,0])
                sample_X_z = self.model(sample_X.reshape(1, embed_dim * sample_len, -1)).reshape(embed_dim, sample_len, self.n_components)
                sample_y_z = self.model(sample_y[:,0])

                subset_X_z = torch.permute(subset_X_z, (1, 0, 2))
                sample_X_z = torch.permute(sample_X_z, (1, 0, 2))

                loss = self.loss_fn(subset_idx, sample_idx,sample_X_z, sample_y_z, subset_X_z, subset_y_z, theta, exclusion_rad, loss_mask_size)
                
                loss /= num_batches
                loss.backward()
                total_loss += loss.item() 

            self.optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')
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

        if self.subtract_corr:
            #corr = -(self._get_autoreg_matrix_approx(sample_y, sample_X))
            #corr = torch.abs(self._get_autoreg_matrix_approx(sample_X[:,[0]], sample_y))
            corr = torch.abs(self._get_autoreg_matrix_approx( sample_y, sample_X[:,[0]]))
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
        
    
    def predict(self, X, return_embedding = True):
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32,device=self.device)
            outputs = torch.permute(self.model(inputs),dims=(0,2,1)) #Easier to interpret
            if return_embedding:
                outputs = get_td_embedding_torch(outputs,self.embed_dim,self.embed_lag).squeeze(-1)
        return outputs.cpu().numpy()

    def generate(self, X, nbrs_num, exclusion_rad, tp=1, device="cpu"):
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


    def _get_td_embedding(ts, dim, stride, return_pred=False, tp=0):
        tdemb = ts.unfold(0,(dim-1) * stride + 1,1)[...,::stride]
        tdemb = torch.swapaxes(tdemb,-1,-2)
        if return_pred:
            return tdemb[:tdemb.shape[0]-tp], ts[(dim-1) * stride + tp:]
        else:
            return tdemb

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
        beta = torch.bmm(torch.inverse(XTWX), XTWy)

        X_ = sample_X_t.unsqueeze(1).expand(dim, dim, sample_size, E_x)
        X_ = X_.reshape(dim * dim * sample_size, E_x)
        X_ = torch.cat([torch.ones((dim * dim * sample_size, 1),device=self.device), X_], dim=1)
        X_ = X_.reshape(dim * dim * sample_size, 1, E_x+1)
        
        A = torch.bmm(X_, beta).reshape(dim, dim, sample_size, E_y)
        A = torch.permute(A,(2,3,1,0))

        B = sample_y.unsqueeze(-1).expand(sample_size, E_y, dim, dim)
        
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
        embed_dim = 3
        embed_lag = 20

        if tp_policy == "fixed":
            dataset = RandomTimeDelaySubsetDataset(X, sample_len, library_len, embed_dim, embed_lag, num_batches, torch.linspace(tp, tp+1 - 1e-5,num_batches,device=self.device).to(torch.int),device=self.device)
        elif tp_policy == "range":
            dataset = RandomTimeDelaySubsetDataset(X, sample_len, library_len, embed_dim, embed_lag, num_batches, torch.linspace(1, tp+1 - 1e-5,num_batches,device=self.device).to(torch.int), device=self.device)
        else:
            pass #TODO: pass an exception

        dataloader = DataLoader(dataset, batch_size=1,pin_memory=False)
        
        
        WW = torch.normal(0,1,(data.shape[1],1),dtype=torch.float64, device=self.device)
        for epoch in range(epochs):
            WW_list = []
            for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
                subset_X_z = (subset_X[0] @ WW).squeeze(-1).T
                subset_y_z = (subset_y[0] @ WW).squeeze(-1).T
                sample_X_z = (sample_X[0] @ WW).squeeze(-1).T
                sample_y_z = (sample_y[0] @ WW).squeeze(-1).T
                subset_idx = subset_idx[:,-1]
                sample_idx = sample_idx[:,-1]

                sample_size = sample_X.shape[2]
                subset_size = subset_X.shape[2]

                dist = torch.cdist(sample_X_z,subset_X_z,)
                weights = torch.exp(-(theta*dist/dist.mean(axis=1)[:,None]))
                
                if exclusion_rad > 0:
                    exclusion_matrix = (torch.abs(subset_idx - sample_idx.T) > exclusion_rad)
                    weights = weights * exclusion_matrix

                W = torch.sqrt(weights.unsqueeze(2))

                X = subset_X_z.unsqueeze(0).expand(sample_size, subset_size, embed_dim)

                Y = subset_y_z.unsqueeze(0).expand(sample_size, subset_size, embed_dim)

                X = torch.cat([torch.ones((sample_size, subset_size, 1),device=self.device), X], dim=2)
                
                X_weighted = X * W
                Y_weighted = Y * W

                XTWX = torch.bmm(X_weighted.transpose(1, 2), X_weighted)
                XTWy = torch.bmm(X_weighted.transpose(1, 2), Y_weighted)
                beta = torch.bmm(torch.inverse(XTWX), XTWy)

                X_ = sample_X_z.unsqueeze(1)
                X_ = torch.cat([torch.ones((sample_size, 1, 1),device=self.device), X_], dim=2)
                
                A = torch.bmm(X_, beta).squeeze(1)[:,[-1]]
                #A = (A-A.mean(axis=0))/A.std(axis=0)

                B = sample_y[0,-1]
                #B = (B-B.mean(axis=0))/B.std(axis=0)


                XtX = torch.matmul(B.T, B)
                XtX_inv = torch.inverse(XtX)
                Xty = torch.matmul(B.T, A)

                WW_ = torch.matmul(XtX_inv, Xty)

                #X_pseudo_inverse = torch.linalg.pinv(B)
                #WW_ = X_pseudo_inverse @ A

                WW_list += [WW_]
            WW__ = torch.stack(WW_list).mean(axis=0)
            #print(WW_.mean())
            WW__ = (WW__-WW__.mean())/WW__.std()
            WW = WW__
            #WW = WW*0.8 + WW__*0.2

            #WW = (WW-WW.mean())/WW.std()

            print(self._get_batch_rmse(A[:,:,None,None],sample_y_z[:,:,None,None]).mean())
        return WW.cpu().detach().numpy()



