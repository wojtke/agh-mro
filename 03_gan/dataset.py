import torch

from torchvision import transforms
from tqdm import tqdm

import os
from tqdm import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CarrotDataset(torch.utils.data.Dataset):
    def __init__(self, data_root_dir, img_size=(32, 32), augment=False):
        if augment:
            self.transform = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        else:
            self.transform = [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        self.transform = transforms.Compose(self.transform)

        if type(data_root_dir) == list:
            self.image_paths = [os.path.join(d, f) for d in data_root_dir for f in os.listdir(d)]
        else:
            self.image_paths = [os.path.join(data_root_dir, f) for f in os.listdir(data_root_dir)]
        self.image_paths = [f for f in self.image_paths if f.endswith((".png", ".jpg", ".jpeg"))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        return self.transform(img)
