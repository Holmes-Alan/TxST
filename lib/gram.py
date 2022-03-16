import torch
from torch import nn
import torch.nn.functional as F


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G / (a *  b* c * d)


######################################################################
# Now the style loss module looks almost exactly like the content loss
# module. The style distance is also computed using the mean square
# error between :math:`G_{XL}` and :math:`G_{SL}`.
#

class Gram_StyleLoss(nn.Module):

    def __init__(self):
        super(Gram_StyleLoss, self).__init__()

    def forward(self, input, target):
        value = torch.tensor(0.).type_as(input[0])
        for in_m, in_n in zip(input, target):
            G = gram_matrix(in_m)
            T = gram_matrix(in_n)
            value += F.mse_loss(G, T)
        return value
