import torch
from torch import nn

def _gumbel_sigmoid(input, temperature=1, hard=False, eps = 1e-10):
    """
    A gumbel-sigmoid nonlinearity with gumbel(0,1) noize
    In short, it's a function that mimics #[a>0] indicator where a is the logit
    
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    
    Math:
    Sigmoid is a softmax of two logits: a and 0
    e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)
    
    Gumbel-sigmoid is a gumbel-softmax for same logits:
    gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
    where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
    gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
    gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
    
    For computation reasons:
    gumbel1-gumbel2 = -log(-log(uniform1(0,1)) +log(-log(uniform2(0,1)) = -log( log(uniform2(0,1)) / log(uniform1(0,1)) )
    gumbel_sigm(a) = sigm([a-log(log(uniform2(0,1))/log(uniform1(0,1))]/t)
    
    
    :param t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic.
    :param eps: a small number used for numerical stability
    :returns: a callable that can (and should) be used as a nonlinearity
    """
    # @staticmethod
    # def forward(ctx, input, temperature=1, hard=False, eps = 1e-10):
    with torch.no_grad():
        # generate a random sample from the uniform distribution
        uniform1 = torch.rand(input.size())
        uniform2 = torch.rand(input.size())
        gumbel_noise = -torch.log(torch.log(uniform1 + eps)/torch.log(uniform2 + eps) + eps).cuda()

    reparam = (input + gumbel_noise)/temperature
#         print(reparam)
    y_soft = torch.sigmoid(reparam)     
    if hard:
        # Straight through.
        index = (y_soft > 0.5).nonzero(as_tuple=True)[0] 
        y_hard = torch.zeros_like(input, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class PruneMask:
    def __init__(self):
        self.temperature = 1
        self._mask = torch.empty(0)
        self.important_score = torch.empty(0)
        self.mask_activation = _gumbel_sigmoid

    @property
    def get_prune_mask(self):
        return self.mask_activation(self._mask, self.temperature)

    @property
    def get_mask(self):
        return self.mask_activation(self._mask * self.important_score, self.temperature)

    @property
    def get_ste_mask(self):
        tmp = self._mask * self.important_score
        return ((torch.sigmoid(tmp) > 0.01).float()- torch.sigmoid(tmp)).detach() + torch.sigmoid(tmp)
    
    def set_mask(self, mask_size, opt):
        self._mask = nn.Parameter(torch.ones(mask_size[0], device="cuda"), requires_grad=True)
        l = [
            {'params': [self._mask], 'lr': 0.05, "name": "mask"},
        ]
        self.optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15)
        self.temperature = opt.gumbel_temp

