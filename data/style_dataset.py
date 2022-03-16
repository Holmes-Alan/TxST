from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import os
import json
import torchvision.transforms as transforms

from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


file_path = os.path.realpath(__file__)

uni_size = 512
crop_size = 256

wiki_style_names = ["Abstract_Expressionism", "Art_Nouveau_Modern", "Contemporary_Realism", "Expressionism",
                    "Impressionism", "Naive_Art_Primitivism", "Pointillism", "Realism", "Symbolism",
                    "Action_painting", "Baroque", "Cubism", "Fauvism", "Mannerism_Late_Renaissance", "New_Realism",
                    "Pop_Art", "Rococo", "Synthetic_Cubism", "Analytical_Cubism", "Color_Field_Painting",
                    "Early_Renaissance", "High_Renaissance", "Minimalism", "Northern_Renaissance", "Post_Impressionism",
                    "Romanticism", "Ukiyo_e"]


def train_transform():
    transform_list = [
        transforms.Resize(size=(uni_size, uni_size)),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


import torch.utils.data as data
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


class StyleDataset(data.Dataset):
    def __init__(self, isTrain=True):
        super(StyleDataset, self).__init__()
        self.sytle_names = wiki_style_names
        with open('data/style_train_test.json') as f:
            img_list_full = json.load(f)

        if isTrain:
            img_list = img_list_full['trainset']
        else:
            img_list = img_list_full['testset']#[:1000]
        self.B_paths = img_list
        self.B_size = len(self.B_paths)
        print("style imgs: %d samples" % (self.B_size))
        if isTrain:
            self.transform_B = train_transform()
        else:
            def get_transform_valid():
                transform_list = []
                transform_list.append(transforms.Resize([crop_size, crop_size]))
                transform_list += [transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                return transforms.Compose(transform_list)

            self.transform_B = get_transform_valid()

    def __getitem__(self, index):
        index_B = index
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert('RGB')
        B = self.transform_B(B_img)

        style_name = os.path.basename(os.path.dirname(B_path))
        style_index = torch.tensor(self.sytle_names.index(style_name), dtype=torch.long)
        name_B = os.path.basename(B_path)
        name = B_path#name_B[:name_B.rfind('.')]

        result = {'img': B, 'style_index': style_index, 'name': name}

        return result

    def __len__(self):
        return self.B_size


def split_train_test_set():
    dir = "datasets/wikiart"
    img_files = sorted(make_dataset(dir))

    import random
    random.seed(10)
    print(random.random())
    shuffled = random.sample(img_files, len(img_files))

    train_list = shuffled[:int(len(shuffled) * 0.9)]
    test_list = shuffled[int(len(shuffled) * 0.9):]
    print(len(train_list), len(test_list))
    save_json = {
        'trainset': train_list,
        'testset': test_list
    }
    import json
    with open('./style_train_test.json', 'w') as outfile:
        json.dump(save_json, outfile)

    with open('./style_train_test.json') as f:
        data = json.load(f)

    print(len(data['trainset']), len(data['testset']))


def tryDataset():
    m_dataset = StyleDataset(isTrain=True)
    m_dataloader = data.DataLoader(m_dataset, batch_size=4, pin_memory=True, num_workers=0,
                                   shuffle=True,
                                   drop_last=False)
    rt_data = next(iter(m_dataloader))
    print(rt_data)


if __name__ == '__main__':
    None
    #split_train_test_set()
