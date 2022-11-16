from torch.utils import data
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import pandas as pd
# from torchvision.io import read_image
from PIL import Image
import torch

def get_transform(random_crop=True):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = []
    #transform.append(transforms.Resize(256))
    if random_crop:
        transform.append(transforms.CenterCrop(800))
    else:
        transform.append(transforms.CenterCrop(800))
    transform.append(transforms.ColorJitter(brightness=.5, hue=.3))
    transform.append(transforms.Resize(300))
    transform.append(transforms.ToTensor())
    transform.append(normalize)
    return transforms.Compose(transform)


class CustomDataset(data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = dict([l.replace('\n', '').split(' ') for l in open(annotations_file).readlines()])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_origin_path = list(self.img_labels.keys())[idx]

        img_path_S1 = os.path.join(self.img_dir, img_origin_path) + '-S01.jpg' # 각 이미지 주소
        img_path_M1 = os.path.join(self.img_dir, img_origin_path) + '-M01.jpg'
        img_path_E1 = os.path.join(self.img_dir, img_origin_path) + '-E01.jpg'

        image_id = img_path_S1.split('-')[0].split('/')[-1]

        image_S1 = Image.open(img_path_S1)
        image_M1 = Image.open(img_path_M1)
        image_E1 = Image.open(img_path_E1)
        label = self.img_labels[img_origin_path]

        if self.transform:
            image_S1 = self.transform(image_S1)
            image_M1 = self.transform(image_M1)
            image_E1 = self.transform(image_E1)
            image = torch.cat((image_S1, image_M1, image_E1))

        label = int(label)

        return image_id, image, label


class TestDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.image_dir = root.replace('label','data')
        self.image_list = sorted(list(set(map(lambda x: x.split('-')[0], os.listdir(root)))))
        self.transform = transform
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_id  = self.image_list[idx]
        img_path_S1 = os.path.join(self.image_dir, img_id) + '-S01.jpg'
        img_path_M1 = os.path.join(self.image_dir, img_id) + '-M01.jpg'
        img_path_E1 = os.path.join(self.image_dir, img_id) + '-E01.jpg'
        image_S1 = Image.open(img_path_S1)
        image_M1 = Image.open(img_path_M1)
        image_E1 = Image.open(img_path_E1)

        image_S1 = self.transform(image_S1)
        image_M1 = self.transform(image_M1)
        image_E1 = self.transform(image_E1)
        image = torch.cat((image_S1, image_M1, image_E1))

        return img_id, image


def test_data_loader(root, phase='train', batch_size=64):
    if phase == 'train':
        is_train = True
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError

    dataset = TestDataset(
        root.replace('label','data'),
        transform=get_transform(random_crop=is_train)
    )
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=4,
                           shuffle=is_train)


def data_loader_with_split(root, train_split=1.0, batch_size=64, val_label_file='./val_label'):
    if root[-1]=='/':
        mode = root.split('/')[-2]
    else:
        mode = root.split('/')[-1]
    dataset = CustomDataset(
        os.path.join(root, mode+'_label'),
        os.path.join(root, mode+'_data'),
        transform=get_transform(
        random_crop=True)
    )
    tr_loader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=4,
                                shuffle=True)


    return tr_loader