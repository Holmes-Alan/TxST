import json
import pathlib
from glob import glob

import torch

from data.image_folder import make_dataset
from PIL import Image
import random
from PIL import ImageFile
from data.template import imagenet_templates

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import torchvision.transforms as transforms

file_path = os.path.realpath(__file__)

import yaml


def read_yaml(yaml_path):
    with open(yaml_path, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(config)
        return config


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((90, 90)),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)

def style_transform():
    transform_list = [
        transforms.Resize(size=(320, 320)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((90, 90)),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


import torch.utils.data as data
from PIL import ImageFile
import clip

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

wiki_style_names = ["Abstract_Expressionism", "Art_Nouveau_Modern", "Contemporary_Realism", "Expressionism",
                    "Impressionism", "Naive_Art_Primitivism", "Pointillism", "Realism", "Symbolism",
                    "Action_painting", "Baroque", "Cubism", "Fauvism", "Mannerism_Late_Renaissance", "New_Realism",
                    "Pop_Art", "Rococo", "Synthetic_Cubism", "Analytical_Cubism", "Color_Field_Painting",
                    "Early_Renaissance", "High_Renaissance", "Minimalism", "Northern_Renaissance", "Post_Impressionism",
                    "Romanticism", "Ukiyo_e"]



class UnalignedDataset_S2(data.Dataset):
    def __init__(self, yaml_path, isTrain=True, isPreserveValidStyle=False):
        super(UnalignedDataset_S2, self).__init__()
        config = read_yaml(yaml_path)
        self.sytle_names = wiki_style_names
        self.isTrain = isTrain
        self.dir_A = config['content_path']
        self.dir_B = config['style_path']
        self.A_paths = sorted(make_dataset(self.dir_A, config['max_dataset_size']))

        # get style images
        if isTrain and isPreserveValidStyle:
            with open('data/style_train_test.json') as f:
                style_img_list_full = json.load(f)
            self.B_paths = style_img_list_full['trainset']
        else:
            self.B_paths = sorted(make_dataset(self.dir_B, config['max_dataset_size']))


        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print("content from %s : %d samples" % (self.dir_A, self.A_size))
        print("style from %s : %d samples" % (self.dir_B, self.B_size))
        if self.isTrain:
            self.transform_A = train_transform()
            self.transform_B = train_transform()
        else:
            def get_transform_valid():
                transform_list = []
                transform_list.append(transforms.Resize([config['crop_size'], config['crop_size']]))
                transform_list += [transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                return transforms.Compose(transform_list)

            self.transform_A = get_transform_valid()
            self.transform_B = get_transform_valid()

    def __getitem__(self, index):
        index_A = index
        index_B = random.randint(0, self.B_size - 1)

        A_path = self.A_paths[index_A]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform_A(A_img)
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert('RGB')
        B = self.transform_B(B_img)

        name_A = os.path.basename(A_path)
        name_B = os.path.basename(B_path)
        name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]


        style_name = os.path.basename(os.path.dirname(B_path))
        painting_name = pathlib.Path(B_path).stem
        painting_name = painting_name.split('_')[0]
        #caption = "%s %s"%(style_name, painting_name)
        caption = painting_name
        caption = caption.replace("-", ' ')
        caption = caption.replace("_", ' ')

        result = {'c': A, 's': B, 'name': name, 'style_token': clip.tokenize(caption)[0]}


        return result

    def __len__(self):
        return self.A_size


class UnalignedDataset_2style(data.Dataset):
    def __init__(self, yaml_path, isTrain=True, isPreserveValidStyle=False):
        super(UnalignedDataset_2style, self).__init__()
        config = read_yaml(yaml_path)
        self.sytle_names = wiki_style_names
        self.isTrain = isTrain
        self.dir_A = config['content_path']
        self.dir_B = config['style_path']
        self.A_paths = sorted(make_dataset(self.dir_A, config['max_dataset_size']))

        # get style images
        if isTrain and isPreserveValidStyle:
            with open('data/style_train_test.json') as f:
                style_img_list_full = json.load(f)
            self.B_paths = style_img_list_full['trainset']
        else:
            self.B_paths = sorted(make_dataset(self.dir_B, config['max_dataset_size']))


        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print("content from %s : %d samples" % (self.dir_A, self.A_size))
        print("style from %s : %d samples" % (self.dir_B, self.B_size))
        if self.isTrain:
            self.transform_A = train_transform()
            self.transform_B = style_transform()
        else:
            def get_transform_valid():
                transform_list = []
                transform_list.append(transforms.Resize([config['crop_size'], config['crop_size']]))
                transform_list += [transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                return transforms.Compose(transform_list)

            self.transform_A = get_transform_valid()
            self.transform_B = get_transform_valid()

    def __getitem__(self, index):
        index_A = index
        index_B = random.randint(0, self.B_size - 1)

        A_path = self.A_paths[index_A]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform_A(A_img)
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert('RGB')
        B = self.transform_B(B_img)

        name_A = os.path.basename(A_path)
        name_B = os.path.basename(B_path)
        name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]


        author_style = name_B.split("_")[0]
        C_path = random.choice(glob(os.path.join(pathlib.Path(B_path).parent.absolute(),"%s*"%author_style)))
        C_img = Image.open(C_path).convert('RGB')
        C = self.transform_B(C_img)

        if self.isTrain:
            style_name = os.path.basename(os.path.dirname(B_path))
            painting_name = pathlib.Path(B_path).stem
            painting_name = painting_name.split('_')[0]
            #caption = "%s %s"%(style_name, painting_name) # general or not
            caption = painting_name
            caption = caption.replace("-", ' ')
            caption = caption.replace("_", ' ')
            
            # style_index = torch.tensor(self.sytle_names.index(style_name), dtype=torch.long)
            
            template = imagenet_templates[torch.randint(0, 79, (1,))]
            caption1 = template.format(caption)
            template = imagenet_templates[torch.randint(0, 79, (1,))]
            caption2 = template.format(caption)
            template = imagenet_templates[torch.randint(0, 79, (1,))]
            source = template.format('a Photo')
            # result = {'c': A, 's': B, 'name': name, 'style_token': clip.tokenize(caption)[0], 's2': C}
            result = {'c': A, 's': B, 'name': name, 'style_token': clip.tokenize(caption1)[0], 'style_token2': clip.tokenize(caption2)[0], 'style_gt': clip.tokenize(source)[0], 's2': C}
        else:
            result = {'c': A, 's': B, 'name': name, 'style_token': clip.tokenize("none")[0]}

        return result

    def __len__(self):
        return self.A_size

class UnalignedDataset_wikisubset(data.Dataset):
    def __init__(self, yaml_path, isTrain=True, dir_A = "datasets/COCO_train2017" ):
        super(UnalignedDataset_wikisubset, self).__init__()
        config = read_yaml(yaml_path)
        self.sytle_names = wiki_style_names
        self.isTrain = isTrain
        self.dir_A = dir_A
        self.dir_B = "datasets/wikiart_subset_new"
        self.A_paths = sorted(make_dataset(self.dir_A, config['max_dataset_size']))

        # get style images

        self.B_paths = sorted(make_dataset(self.dir_B, config['max_dataset_size']))


        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print("content from %s : %d samples" % (self.dir_A, self.A_size))
        print("style from %s : %d samples" % (self.dir_B, self.B_size))
        if self.isTrain:
            self.transform_A = train_transform()
            self.transform_B = train_transform()
        else:
            def get_transform_valid():
                transform_list = []
                transform_list.append(transforms.Resize([config['crop_size'], config['crop_size']]))
                transform_list += [transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                return transforms.Compose(transform_list)

            self.transform_A = get_transform_valid()
            self.transform_B = get_transform_valid()

    def __getitem__(self, index):
        index_A = index
        index_B = random.randint(0, self.B_size - 1)

        A_path = self.A_paths[index_A]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform_A(A_img)
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert('RGB')
        B = self.transform_B(B_img)

        name_A = os.path.basename(A_path)
        name_B = os.path.basename(B_path)
        name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]


        #author_style = name_B.split("_")[0]
        C_path = random.choice(glob(os.path.join(pathlib.Path(B_path).parent.absolute(),"*")))
        C_img = Image.open(C_path).convert('RGB')
        C = self.transform_B(C_img)

        if self.isTrain:
            painting_name = str(pathlib.Path(B_path).parent)
            painting_name = painting_name.split('/')[-1]
            #caption = "%s %s"%(style_name, painting_name)
            caption = painting_name
            caption = caption.replace("-", ' ')
            caption = caption.replace("_", ' ')
            # style_index = torch.tensor(self.sytle_names.index(style_name), dtype=torch.long)


            template = imagenet_templates[torch.randint(0, 79, (1,))]
            caption1 = template.format(caption)
            template = imagenet_templates[torch.randint(0, 79, (1,))]
            caption2 = template.format(caption)
            template = imagenet_templates[torch.randint(0, 79, (1,))]
            source = template.format('a Photo')
            
            # result = {'c': A, 's': B, 'name': name, 'style_token': clip.tokenize(caption)[0], 's2': C}
            result = {'c': A, 's': B, 'name': name, 'style_token': clip.tokenize(caption1)[0], 'style_token2': clip.tokenize(caption2)[0], 'style_gt': clip.tokenize(source)[0], 's2': C}
        else:
            result = {'c': A, 's': B, 'name': name, 'style_token': clip.tokenize("none")[0]}

        return result

    def __len__(self):
        return self.A_size


class UnalignedDataset_dtd(data.Dataset):
    def __init__(self, yaml_path, isTrain=True, dir_A = "datasets/COCO_train2017" ):
        super(UnalignedDataset_dtd, self).__init__()
        config = read_yaml(yaml_path)
        self.sytle_names = wiki_style_names
        self.isTrain = isTrain
        self.dir_A = dir_A
        self.dir_B = "datasets/DTD"
        self.A_paths = sorted(make_dataset(self.dir_A, config['max_dataset_size']))

        # get style images

        self.B_paths = sorted(make_dataset(self.dir_B, config['max_dataset_size']))


        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print("content from %s : %d samples" % (self.dir_A, self.A_size))
        print("style from %s : %d samples" % (self.dir_B, self.B_size))
        if self.isTrain:
            self.transform_A = train_transform()
            self.transform_B = train_transform()
        else:
            def get_transform_valid():
                transform_list = []
                transform_list.append(transforms.Resize([config['crop_size'], config['crop_size']]))
                transform_list += [transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                return transforms.Compose(transform_list)

            self.transform_A = get_transform_valid()
            self.transform_B = get_transform_valid()

    def __getitem__(self, index):
        index_A = index
        index_B = random.randint(0, self.B_size - 1)

        A_path = self.A_paths[index_A]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform_A(A_img)
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert('RGB')
        B = self.transform_B(B_img)

        name_A = os.path.basename(A_path)
        name_B = os.path.basename(B_path)
        name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]


        #author_style = name_B.split("_")[0]
        C_path = random.choice(glob(os.path.join(pathlib.Path(B_path).parent.absolute(),"*")))
        C_img = Image.open(C_path).convert('RGB')
        C = self.transform_B(C_img)

        if self.isTrain:
            painting_name = str(pathlib.Path(B_path).parent)
            painting_name = painting_name.split('/')[-1]
            #caption = "%s %s"%(style_name, painting_name)
            caption = painting_name
            caption = caption.replace("-", ' ')
            caption = caption.replace("_", ' ')
            # style_index = torch.tensor(self.sytle_names.index(style_name), dtype=torch.long)


            template = imagenet_templates[torch.randint(0, 79, (1,))]
            caption1 = template.format(caption)
            template = imagenet_templates[torch.randint(0, 79, (1,))]
            caption2 = template.format(caption)
            template = imagenet_templates[torch.randint(0, 79, (1,))]
            source = template.format('a Photo')
            
            # result = {'c': A, 's': B, 'name': name, 'style_token': clip.tokenize(caption)[0], 's2': C}
            result = {'c': A, 's': B, 'name': name, 'style_token': clip.tokenize(caption1)[0], 'style_token2': clip.tokenize(caption2)[0], 'style_gt': clip.tokenize(source)[0], 's2': C}
        else:
            result = {'c': A, 's': B, 'name': name, 'style_token': clip.tokenize("none")[0]}

        return result

    def __len__(self):
        return self.A_size
        
        
def tryDataset():
    m_dataset = UnalignedDataset_wikisubset(yaml_path="data/train_conf.yaml")
    m_dataloader = data.DataLoader(m_dataset, batch_size=4, pin_memory=True, num_workers=0,
                                   shuffle=True,
                                   drop_last=False)
    rt_data = next(iter(m_dataloader))
    print(rt_data)


if __name__ == '__main__':
    tryDataset()
