import glob
import os
import pathlib
import argparse
import torch
from lib.pl_utils import UnNormalize
from model_define import StyleTransfer
import torchvision.transforms as transforms
from PIL import Image
import clip
import torchvision.utils as vutils


# Testing settings
parser = argparse.ArgumentParser(description='PyTorch TxST Example')
parser.add_argument('--content', type=str, default='data/content', help="content images")
parser.add_argument('--style', type=str, default='data/style', help='style images')

opt = parser.parse_args()

def tensor_rgb2gray(tensor_img):
        tensor_img_gray = tensor_img[:, 0:1, :, :] * 0.299 + tensor_img[:, 1:2, :, :] * 0.587 + tensor_img[:, 2:3, :,
                                                                                                :] * 0.114
        tensor_img_gray = tensor_img_gray.expand(tensor_img.size())
        return tensor_img_gray
        
def read_content_img(img_path, img_siz=None):
    if img_siz is None:
        transform_list = [#transforms.Resize((img_siz, img_siz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    else:
        transform_list = [transforms.Resize((img_siz, img_siz)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    m_transform = transforms.Compose(transform_list)
    img = Image.open(img_path).convert('RGB')
    img_tensor = m_transform(img)
    return img_tensor.unsqueeze(0).cuda()


def read_style_img(img_path, img_siz=256):
    style_img = read_content_img(img_path, img_siz)
    style_name = os.path.basename(os.path.dirname(img_path))
    # print(style_name)
    return style_img, clip.tokenize(style_name.replace("_", " "))[0].unsqueeze(0).cuda()


def custom_text(text):
    return clip.tokenize(text)[0].unsqueeze(0)


if __name__ == '__main__':
    m_model = StyleTransfer.load_from_checkpoint(
        "models/wikiart_subset.ckpt").cuda()

    cont_imgs = glob.glob(os.path.join(opt.content, "*.*"))
    style_imgs = glob.glob(os.path.join(opt.style, "*.*"))
    for cont_file in cont_imgs:
        for style_file in style_imgs:
            save_path = os.path.join("output/", "%s_stylized_%s.png"%(pathlib.Path(cont_file).stem,pathlib.Path(style_file).stem))
            I_c = read_content_img(cont_file, img_siz=512)
            I_s, style_token = read_style_img(style_file)
            # ==== original inference ====
            # encoding
            F_c = m_model.encoder(I_c)

            F_clip_c = m_model.text_editor.encode_img(I_c)
            F_clip_s = m_model.text_editor.encode_img(I_s)
            # style transfer
            styled = m_model.transform(F_clip_c['raw_feat'], F_clip_s['raw_feat'], F_c)

            # decoding
            I_cs = m_model.decoder(styled)

            # visualize
            m_unnormalize = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            transfer = m_unnormalize(I_cs).squeeze(0).clamp(0, 1).cpu().data
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            ndarr = transfer.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(save_path)
            # im.show()
