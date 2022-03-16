import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class ST_LossCriterion_v2(nn.Module):
    def  __init__(self):
        super(ST_LossCriterion_v2, self).__init__()
        self.L2_loss = nn.MSELoss()

    def forward(self, tF, sF, cF):
        # content loss
        totalContentLoss = (self.L2_loss(mean_variance_norm(tF.relu4_1), mean_variance_norm(cF.relu4_1)) +
                            self.L2_loss(mean_variance_norm(tF.relu5_1), mean_variance_norm(cF.relu5_1)))
        # style loss
        totalStyleLoss = 0
        for ft_x, ft_s in zip(tF, sF):
            mean_x, var_x = calc_mean_std(ft_x)
            mean_style, var_style = calc_mean_std(ft_s)
            totalStyleLoss = totalStyleLoss + self.L2_loss(mean_x, mean_style)
            totalStyleLoss = totalStyleLoss + self.L2_loss(var_x, var_style)

        return totalStyleLoss, totalContentLoss


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)

    def gaussian(x):
        return np.exp(
            (x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2

    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable conv we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    # conv img with a gaussian kernel that has been built with build_gauss_kernel
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        self.L1_loss = nn.L1Loss()

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(size=self.k_size, sigma=self.sigma,
                                                    n_channels=input.shape[1], cuda=input.is_cuda)

        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(self.L1_loss(a, b) for a, b in zip(pyr_input, pyr_target))