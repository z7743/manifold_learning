import torch
import torch.nn as nn

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
        #self.model = BitLinearNew(input_dim, output_dim*embed_dim, bias=False,device=self.device,)
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
