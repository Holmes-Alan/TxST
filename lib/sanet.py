import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as Func


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

class MHSA(nn.Module):
    def __init__(self, n_dims, width=16, height=16):
        super(MHSA, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # b, c, 16,16
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return out
        

class CrossAttN_v8(nn.Module):
    def __init__(self, in_planes, clip_dim):
        super(CrossAttN_v8, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        self.g = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        self.h = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        self.output = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        return

    def forward(self, F_c, F_s):
        b, c = F_c.shape[0], F_c.shape[1]

        # T_c = F_c
        F = self.f(F_c)
        F = F.view(b, F.shape[1], -1)
        G = self.g(F_s)
        G = G.view(b, G.shape[1], -1)
        # G = G / G.norm(dim=2, keepdim=True)
        H = self.h(F_s)
        H = H.view(b, H.shape[1], -1)
        # H = H / H.norm(dim=2, keepdim=True)
        S = torch.bmm(F.permute(0, 2, 1), G) # b, d_s, d_c
        S = Func.softmax(S, dim=-1)
        result = torch.bmm(H, S.permute(0, 2, 1)) # b, d_c, h*w
        result = result.view(b, result.shape[1], F_c.shape[2], F_c.shape[3])
        result = self.output(result)

        return result
        

    def __init__(self, in_planes):
        super(SelfAttN, self).__init__()
        scale = 0.01
        self.f = nn.Conv2d(in_planes, 64, 1, 1, 0)
        self.g = nn.Conv2d(in_planes, 64, 1, 1, 0)
        self.h = nn.Conv2d(in_planes, 64, 1, 1, 0)
        self.output = nn.Conv2d(64, in_planes, 1, 1, 0)
        return

    def forward(self, F_c):
        b, c = F_c.shape[0], F_c.shape[1]
        F = self.f(F_c)
        F = F.view(b, F.shape[1], -1)
        G = self.g(F_c)
        G = G.view(b, G.shape[1], -1)
        H = self.h(F_c)
        H = H.view(b, H.shape[1], -1)
        S = torch.bmm(F.permute(0, 2, 1), G) # b, d_s, d_c
        S = Func.softmax(S, dim=-1)
        result = torch.bmm(H, S.permute(0, 2, 1)) # b, d_c, h*w
        result = result.view(b, result.shape[1], F_c.shape[2], F_c.shape[3])
        result = self.output(result)
        result = result + F_c

        return result


class CA_SA_v4(nn.Module):
    def __init__(self, in_planes, out_planes, clip_dim=512, max_sample=256 * 256):
        super(CA_SA_v4, self).__init__()
        self.clip_dim = clip_dim
        self.fs = nn.Sequential(
            nn.Conv2d(clip_dim, in_planes, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        )
        self.prepare_out = nn.Conv2d(in_planes, out_planes, 1, 1, 0)
        self.attn1 = CrossAttN_v8(in_planes, clip_dim)  # 1st order
        self.attn2 = CrossAttN_v8(in_planes, clip_dim)  # 2ed order
        return

    def forward(self, F_clip_s, F_content):
        b, c, h, w = F_content.shape
        # F_clip_s: # b, 512, 16, 16
        F_clip_s = self.fs(F_clip_s)
        F_c = mean_variance_norm(F_content)
        mean_s, std_s = calc_mean_std(F_clip_s)
        F_s = (F_clip_s - mean_s.expand(F_clip_s.size())) / std_s.expand(F_clip_s.size())
        result = self.attn1(F_c, F_s) + self.attn2(F_c, torch.pow(F_s, 2)) + F_c
        result = result.view(b, -1, h, w).contiguous()
        result = mean_variance_norm(result) * std_s.expand(F_content.size()) + mean_s.expand(F_content.size())
        result = self.prepare_out(result)
        return result
        

class Transform_CA_SA_v4(nn.Module):
    def __init__(self, isDisable=False):
        super(Transform_CA_SA_v4, self).__init__()
        in_planes = [64, 128, 320, 512]
        self.isDisable = isDisable
        self.MHSA = MHSA(512)
        self.sanet0 = CA_SA_v4(in_planes=in_planes[0], out_planes=in_planes[0])
        self.sanet1 = CA_SA_v4(in_planes=in_planes[0] + in_planes[1], out_planes=in_planes[1])
        self.sanet2 = CA_SA_v4(in_planes=in_planes[0] + in_planes[1] + in_planes[2], out_planes=in_planes[2])

        return

    def get_key(self, feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return mean_variance_norm(feats[last_layer_idx])

    def edit_style(self, F_s, mean, std):
        mean = mean.expand_as(F_s)
        std = std.expand_as(F_s)
        return (1 + std) * (F_s) + mean

    def forward(self, F_clip_c, F_clip_s, F_c):
        if self.isDisable: return F_c  # if is disable style transfer
        F_clip_s = F_clip_s.permute(0, 2, 1).unsqueeze(-1)  # b, 512, 1, 1
        F_clip_s = F_clip_s.repeat(1, 1, 16, 16)  # b, 512, 16, 16
        F_clip_s = self.MHSA(F_clip_s) + F_clip_s# b, 512, 16, 16

        F0 = self.sanet0(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 0))
        F1 = self.sanet1(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 1))
        F2 = self.sanet2(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 2))
        F_out = [F0, F1, F2]
        return F_out  # feat # [3, 512, 64, 64]


