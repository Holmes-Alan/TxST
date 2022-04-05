import os

import torch
import argparse
from lib.pl_utils import UnNormalize
from model_define import StyleTransfer
import torchvision.transforms as transforms
from PIL import Image
import clip
import pathlib
import torchvision.utils as vutils
from torchvision.transforms.functional import adjust_contrast

# Testing settings
parser = argparse.ArgumentParser(description='PyTorch TxST Example')
parser.add_argument('--content', type=str, default='data/content/church.jpeg', help="content images")
parser.add_argument('--style', type=str, default='chequered', help='text styles')

opt = parser.parse_args()

def read_content_img(img_path, img_siz=512):
    transform_list = [transforms.Resize((img_siz, img_siz)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    m_transform = transforms.Compose(transform_list)
    img = Image.open(img_path).convert('RGB')
    img_tensor = m_transform(img)
    return img_tensor.unsqueeze(0).cuda()


def read_style_img(img_path, img_siz=512):
    style_img = read_content_img(img_path, img_siz)
    style_name = os.path.basename(os.path.dirname(img_path))
    print(style_name)
    return style_img, clip.tokenize(style_name.replace("_", " "))[0].unsqueeze(0).cuda()


def custom_text(text):
    return clip.tokenize(text)[0].unsqueeze(0)


# text_dic = ['banded', 'bubbly', 'chequered', 'cracked', 'crosshatched', 'grid', 'honeycombed', 'lacelike', 'matted', 'meshed', 'srinkled', 'stained', 'zigzagged']

if __name__ == '__main__':
    m_model = StyleTransfer.load_from_checkpoint(
        "models/texture.ckpt").cuda()

    I_c = read_content_img(opt.content)

    # encoding
    # L = len(text_dic)
    # I_c = I_c.repeat(L, 1, 1, 1)
    F_c = m_model.encoder(I_c)
    F_clip_c = m_model.text_editor.encode_img(I_c)

    
    # === use text inference ===
    # text_input = torch.zeros(L, 77).cuda().long()
    # for i in range(L):
    #     text_token = custom_text(text_dic[i]).cuda()
    #     text_input[i:i+1, :] = text_token
    text_input = custom_text(opt.style).cuda()
    meta = m_model.text_editor.forward(text_input)
    F_clip_text = meta['raw_feat']
    F_clip_text = F_clip_text.unsqueeze(1)

    styled = m_model.transform(F_clip_c['raw_feat'], F_clip_text, F_c)

    # decoding
    I_cs = m_model.decoder(styled)


    # visualize
    m_unnormalize = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    content = m_unnormalize(I_c).clamp(0, 1).cpu().data
    style = m_unnormalize(I_s).clamp(0, 1).cpu().data
    transfer = m_unnormalize(I_cs).clamp(0, 1).cpu().data
    
    transfer = adjust_contrast(transfer,1.5)
    # for i in range(L):
    #     out = transfer[i:i+1, :, :, :].squeeze(0)
    #     ndarr = out.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #     im = Image.fromarray(ndarr)
    #     save_path = 'output/' + str(i).zfill(2) + '.png'
    #     im.save(save_path)
    out = transfer.squeeze(0)
    ndarr = out.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    save_path = 'output/' + pathlib.Path(opt.content).stem + '_' + pathlib.Path(opt.style).stem + '.png'
    im.save(save_path)





