import torch 
from torch import nn, Tensor
from torch.nn import functional as fn

class CLIPLossCls(nn.Module):
    def __init__(self, temperature) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) *  torch.log(torch.tensor(1/0.07)))
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, input:Tensor) -> Tensor:
        """
        input: Tensor of shape [batch_size, h_dim]
        """    
        assert input.size(0) % 2 == 0, f"Batch size must be even. Got {input.size(0)}"
        half = input.size(0) // 2
        first, second = input[:half], input[half:]
        first = first / first.norm(dim=-1, keepdim=True)
        second = second / second.norm(dim=-1, keepdim=True)

        sim_mat = first @ second.T * self.logit_scale
        labels = torch.arange(half, device=input.device)
        first_loss = self.cross_entropy(sim_mat,labels)
        second_loss = self.cross_entropy(sim_mat.T, labels)
        return (first_loss + second_loss) / 2
    

def CLIPLoss(input:Tensor, temperature:float=0.07):
    assert input.size(0) % 2 == 0, f"Batch size must be even. Got {input.size(0)}"
    half = input.size(0) // 2
    first, second = input[:half], input[half:]
    first = fn.normalize(first)
    second = fn.normalize(second)

    sim_mat = first @ second.T / temperature
    labels = torch.arange(half, device=input.device)
    first_loss = fn.cross_entropy(sim_mat,labels)
    second_loss = fn.cross_entropy(sim_mat.T, labels)
    return (first_loss + second_loss) / 2