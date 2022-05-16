import torch
from torch import nn
from torch.nn import functional as F

class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp()))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=0)
        return kl.mean()

def label_smoothing_loss(pred, gold, trg_pad_idx, smoothing_eps=0.1):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - smoothing_eps) + (1 - one_hot) * smoothing_eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    non_pad_mask = gold.ne(trg_pad_idx)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).mean()
    return loss

def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum

class MaximumMeanDiscrepancyLoss(nn.Module):
    def __init__(self):
        super(MaximumMeanDiscrepancyLoss, self).__init__()

    def forward(self, z_tilde, z, z_var):
        r"""Calculate maximum mean discrepancy described in the WAE paper.
        Args:
            z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
                2D Tensor(batch_size x dimension).
            z (Tensor): samples from prior distributions. same shape with z_tilde.
            z_var (Number): scalar variance of isotropic gaussian prior P(Z).
        """
        batch_size = z_tilde.size(0)
        z_tilde = z_tilde.view(batch_size, -1)
        z = z.view(batch_size, -1)
        assert z_tilde.size() == z.size()
        assert z.ndimension() == 2

        n = z.size(0)
        out = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n*(n-1)) + \
            im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n*(n-1)) + \
            -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n*n).mul(2)

        return out