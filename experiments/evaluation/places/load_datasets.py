from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import torch
from place import cat_2_idx, img_transform
import os

class Places(Dataset):
    def __init__(self, img_transform):
        self.labels = []
        self.img_transform = img_transform
        with open('./val_256/places365_val.txt', 'r') as f:
            self.labels = [int(line.strip().split()[-1]) for line in f]
        with open('./val_256/places365_val.txt', 'r') as f:
            self.img_paths = [('./val_256/' + str(line.strip().split()[0])) for line in f]
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        img_tensor = self.img_transform(image)
        return img_tensor, torch.tensor(self.labels[index], dtype=torch.long)
    

custom_dataset = datasets.ImageFolder(
    root='./custom_dataset',
    transform=img_transform
)
new_map = {cls_name: cat_2_idx[cls_name] for cls_name in custom_dataset.classes}

custom_dataset.class_to_idx = new_map
custom_dataset.samples = [
    (path, new_map[class_name])
    for (path, _old_idx) in custom_dataset.samples
    for class_name in [path.split(os.sep)[-2]]
]

custom_dataset.targets = [label for _, label in custom_dataset.samples]
import matplotlib.pyplot as plt
from collections import Counter