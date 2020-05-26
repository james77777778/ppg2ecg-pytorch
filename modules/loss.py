from torch import nn
from torch.nn.functional import l1_loss


class QRSLoss(nn.Module):
    def __init__(self, beta=5):
        super(QRSLoss, self).__init__()
        self.beta = 5

    def forward(self, input, target, exp_rpeaks):
        return l1_loss(
            input*(1+self.beta*exp_rpeaks), target*(1+self.beta*exp_rpeaks))
