import torch 
from torch import nn, Tensor
from torch.nn import functional as fn

class CLIPLossCls(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) *  torch.log(torch.tensor(1/0.07)))
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, input:Tensor) -> Tensor:
        """
        input: Tensor of shape [batch_size, h_dim]
        """    

        input = input / input.norm(dim=-1, keepdim=True)
        sim_mat = input @ input.T * self.logit_scale.exp()
        labels = torch.arange(input.shape[0], device=input.device)
        return self.cross_entropy(sim_mat,labels)

def CLIPLoss(model:nn.Module, input:Tensor) -> Tensor:
    """
    input: Tensor of shape [batch_size, h_dim]
    """
    raise NotImplementedError
    logit_scale = getattr(model, "logit_scale", None)
    if logit_scale is None or not torch.is_tensor(logit_scale):
        raise ValueError("Model must have a tensor attribute 'logit_scale")

    input = input / input.norm(dim=-1, keepdim=True)
    sim_mat = input @ input.T * logit_scale.exp()
    labels = torch.arange(input.shape[0], device=input.device)
    return fn.cross_entropy(sim_mat, labels)