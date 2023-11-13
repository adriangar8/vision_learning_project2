import torch
import torch.nn.functional as F

def mse_loss(input, target):
    return F.mse_loss(input, target)

def mae_loss(input, target):
    return F.l1_loss(input, target)

class LinearEnsembleLoss(torch.nn.Module):
    def __init__(self, weights):
        super(LinearEnsembleLoss, self).__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(weights))
        
    def forward(self, input, target):
        loss1 = mse_loss(input, target)
        loss2 = mae_loss(input, target)
        ensemble_loss = torch.sum(self.weights * torch.tensor([loss1, loss2]))
        return ensemble_loss