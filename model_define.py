import pathlib

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from lib.pl_utils import CheckpointEveryNSteps, UnNormalize, load_module_params_from_ckpt, fixParameter
import os
from lib.losses import mean_variance_norm
import torch.nn as nn
import torch.nn.functional as Func


Exp_name = "VGG_Pretrain"

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
seed_everything(123)


class StyleTransfer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # parameters
        self.isTrain_Decoder = True
        self.train_batch_size = 30

        # ==== VGG encoder =====
        from lib.VGG import VGG, loss_network
        vgg_net = VGG('VGG19')
        vgg_net.features.load_state_dict(torch.load("pretrained_models/vgg_normalised.pth"))
        vgg_net = torch.nn.Sequential(*list(vgg_net.features.children())[:44])
        self.vgg_feat = loss_network(vgg_net)


        # ==== encoder ====

        from lib.VGG import VGG_Feature
        self.encoder = VGG_Feature(vgg_net)  # VGG encoder, adjust channel same to SegFormer


        # ==== Editor ====
        from lib.CLIP import CLIP_Edit
        self.text_editor = CLIP_Edit()

        # ==== transform ====
        from lib.sanet import Transform_CA_SA_v4
        self.transform = Transform_CA_SA_v4(isDisable=False)

        # === decoder ===
        from lib.decoder import decoder_ViT4
        self.decoder = decoder_ViT4(isUseShallow=True)

        if not self.isTrain_Decoder: self.decoder = fixParameter(self.decoder)


        # for visualization
        self.save_pool = []
        self.unnormalize = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return

    def forward(self, in_data):

        I_c, I_s = in_data['c'], in_data['s']

        # encoding
        F_c = self.encoder(I_c)

        F_clip_c = self.text_editor.encode_img(I_c)
        F_clip_s = self.text_editor.encode_img(I_s)

        # style transfer
        styled_cs = self.transform(F_clip_c['raw_feat'], F_clip_s['raw_feat'], F_c)

        # decoding
        I_cs = self.decoder(styled_cs)

        meta = {
            'F_vit_c': F_c,
            'clip_F_style': F_clip_s,
            "F_clip_c": F_clip_c
        }

        return I_cs, meta
 
    def tensor_rgb2yuv(self, tensor_img):
        tensor_img_y = tensor_img[:, 0:1, :, :] * 0.299 + tensor_img[:, 1:2, :, :] * 0.587 + tensor_img[:, 2:3, :,
                                                                                                :] * 0.114
        tensor_img_u = tensor_img[:, 0:1, :, :] * -0.147 + tensor_img[:, 1:2, :, :] * -0.289 + tensor_img[:, 2:3, :,
                                                                                                :] * 0.436
        tensor_img_v = tensor_img[:, 0:1, :, :] * 0.615 + tensor_img[:, 1:2, :, :] * -0.515 + tensor_img[:, 2:3, :,
                                                                                                :] * -0.1
        tensor_img = torch.cat((tensor_img_y, tensor_img_u, tensor_img_v), dim=1)
        return tensor_img

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

        max_sample = 64 * 64
        from lib.sanet import calc_mean_std
        loss_local = torch.tensor(0., device=self.device)
        for i in range(4, 5):
            c_key = F_cs[i]
            s_key = F_c[i]
            b, c_c, h_c, w_c = c_key.size()
            c_key = mean_variance_norm(c_key)
            s_key = Func.interpolate(s_key, size=[64, 64], mode='bilinear')
            mean_s, std_s = calc_mean_std(s_key)
            s_key = (s_key - mean_s.expand(s_key.size())) / std_s.expand(s_key.size())
            c_key = c_key.view(b, c_c, -1).permute(0, 2, 1).contiguous()
            s_key = s_key.view(b, c_c, -1).contiguous()
            attn = torch.bmm(c_key, s_key)
            # S: b, n_c, n_s
            attn = torch.softmax(attn, dim=-1)
            # mean: b, n_c, c
            result = torch.bmm(s_key, attn.permute(0, 2, 1))
            # mean, std: b, c, h, w
            result = result.view(b, c_c, h_c, w_c).contiguous()
            result = result + c_key.view(b, c_c, h_c, w_c).contiguous()
            loss_local += self.l1_loss(F_c[i], std_s.expand(result.size()) * mean_variance_norm(result) + mean_s.expand(result.size()))
        return loss_local
        
    def tensor_rgb2gray(self, tensor_img):
        tensor_img_gray = tensor_img[:, 0:1, :, :] * 0.299 + tensor_img[:, 1:2, :, :] * 0.587 + tensor_img[:, 2:3, :,
                                                                                                :] * 0.114
        tensor_img_gray = tensor_img_gray.expand(tensor_img.size())
        return tensor_img_gray


if __name__ == '__main__':
    print('===> Call the trainer')
    m_model = StyleTransfer()

