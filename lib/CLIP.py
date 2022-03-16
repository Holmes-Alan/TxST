import tabnanny

import torch
import clip
from PIL import Image
import torch.nn as nn
import torch.nn.functional as Func
import einops

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(
        Image.open("datasets/wikiart/Abstract_Expressionism/aaron-siskind_acolman-1-1955.jpg")).unsqueeze(0).to(device)
    text = clip.tokenize(["Abstract Expressionism", "Art_Nouveau Modern", "Contemporary Realism"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

from lib.pl_utils import UnNormalize
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class CLIP_Edit(nn.Module):
    def __init__(self, used_model="ViT-B/32"):
        super(CLIP_Edit, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.CLIP_model, self.preprocess = clip.load(used_model, device=self.device)
        for param in self.CLIP_model.parameters():
            param.requires_grad = False
        #self.img_proj = nn.Linear(768, 512)
        #self.txt_proj = nn.Linear(512, 512)
        self.unify_size = 16
        self.unnormalize = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return

    def forward(self, style_token):
        raw_feat, feat_befact = self.CLIP_model.encode_text(style_token, isRetRaw=True)
        meta = {
            'raw_feat': raw_feat.float(),
            "feat_befact": feat_befact.float()
        }
        return meta

    def my_preprocessImg(self, img_tensor):
        # preprocess
        img_tensor = self.unnormalize(img_tensor)
        m_transform = transforms.Compose([
            #Resize((224,224)),
            transforms.RandomCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        img_tensor = m_transform(img_tensor)

        # encode
        # self.CLIP_model.encode_image(img_tensor)
        return img_tensor

    # def encode_img(self, img_tensor):
    #     # preprocess
    #     img_tensor = self.my_preprocessImg(img_tensor)
    #     # encode
    #     with torch.no_grad():
    #         # raw_feat = self.CLIP_model.encode_image(img_tensor).float()
    #         raw_feat, feat_befact = self.CLIP_model.encode_image(img_tensor, isRetRaw=True)
    #         raw_feat = raw_feat.float()
    #         feat_befact = feat_befact.float()
    #     m_feat = self.post_process(raw_feat.unsqueeze(-1).unsqueeze(-1))  # b,512,1,1
    #     b, c, h, w = m_feat.shape
    #     m_feat = m_feat.reshape((b, 2, self.unify_size, self.unify_size))
    #     [mean, std] = torch.split(m_feat, 1, dim=1)
    #     meta = {
    #         'raw_feat': raw_feat,
    #         "feat_befact": feat_befact
    #     }
    #     return [mean, std], meta

    def encode_img(self, img_tensor):
        # preprocess
        img_tensor = self.my_preprocessImg(img_tensor)

        # encode
        raw_feat, feat_befact = self.CLIP_model.encode_image(img_tensor, isRetRaw=True)
        #feat_befact = self.img_proj(feat_befact.float())
        meta = {
            'raw_feat': raw_feat.float().unsqueeze(1),
            "feat_befact": feat_befact.float()
        }
        return meta

    def encode_img_pyramid(self, img_tensor):
        img_tensor = self.unnormalize(img_tensor)
        m_normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        img_tensor = m_normalize(img_tensor)

        # image to patch
        b, c, h, w = img_tensor.shape
        hk, wk = h//3, w//3
        patches = Func.unfold(img_tensor, [hk, wk], stride=[hk, wk])
        B, c_hk_wk, N_pth = patches.shape
        patches = patches.permute(0,2,1)
        patches = patches.view(B, N_pth, c, hk, wk)
        patches = einops.rearrange(patches, "B N_pth c hk wk -> (B N_pth) c hk wk", B=B, N_pth=N_pth, c=c,hk=hk, wk=wk)

        # resize patch and image
        m_resizer = Resize((224,224))
        patches = m_resizer(patches)
        img_tensor = m_resizer(img_tensor)

        # encode
        meta = self.encode_img(patches)
        meta_global = self.encode_img(img_tensor) # B, 1, 512
        
        patch_feature = einops.rearrange(meta['raw_feat'].float().squeeze(1), "(B N_pth) dim -> B N_pth dim", B=B, N_pth=N_pth, dim=512) # N_pth = 9
        raw_feat = torch.cat([patch_feature, meta_global['raw_feat'].float()], dim=1) # B, 10, 512
        
        patch_feature = einops.rearrange(meta['feat_befact'].float().squeeze(1), "(B N_pth) dim -> B N_pth dim", B=B, N_pth=N_pth, dim=768) # N_pth = 9
        feat_befact = torch.cat([patch_feature, meta_global['feat_befact'].unsqueeze(1).float()], dim=1) # B, 10, 512
        
        
        meta = {
            'raw_feat': raw_feat,
            "feat_befact": feat_befact
        }
        
        return meta
        
    def getImgRawFeat(self, img_tensor):
        # preprocess
        img_tensor = self.my_preprocessImg(img_tensor)
        # encode
        with torch.no_grad():
            raw_feat = self.CLIP_model.encode_image(img_tensor)
        return raw_feat.float() # 1*1*512
        
    def getImgRawFeat_patch(self, img_tensor):
        # preprocess
        b = img_tensor.shape[0]
        img_tensor = self.my_preprocessImg(img_tensor)
        # encode
        with torch.no_grad():
            img = img_tensor.unfold(2, 112, 56).unfold(3, 112, 56)
            img = torch.cat(torch.split(img, 1, dim=2), dim=0)
            img = img.squeeze(dim=2)
            img = torch.cat(torch.split(img, 1, dim=2), dim=0)
            img = img.squeeze(dim=2)
            img = Func.interpolate(img, size=[224, 224])
            raw_feat = self.CLIP_model.encode_image(img)
            raw_feat = torch.cat(torch.split(raw_feat, b, dim=0), dim=1)
        return raw_feat.float() # 1*512

    def getTxtRawFeat(self, txt_token):
        with torch.no_grad():
            raw_feat = self.CLIP_model.encode_text(txt_token)
        return raw_feat.float()

    def getTxtRawFeat_pyramid(self, txt_token):
        with torch.no_grad():
            raw_feat = self.CLIP_model.encode_text(txt_token).float().unsqueeze(1)
        raw_feat_pyramid = torch.cat([raw_feat for i in range(10)], dim=1) # B, 10, 512
        return raw_feat_pyramid

# if __name__ == '__main__':
#     m_CLIP = CLIP_Edit().cuda()
#     text = "Abstract Expressionism"
#     print(m_CLIP.forward(text))
