import os

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# referred from "https://hoya012.github.io/blog/albumentation_tutorial/"
class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        # Read an image with OpenCV
        image = cv2.imread(file_path)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label


albumentations_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.OneOf([
        A.HorizontalFlip(p=1),
        A.RandomRotate90(p=1),
        A.VerticalFlip(p=1)
    ], p=1),
    A.OneOf([
        A.MotionBlur(p=1),
        A.OpticalDistortion(p=1),
        A.GaussNoise(p=1)
    ], p=1),
    ToTensorV2()
])

albumentations_transform_test = A.Compose([
    # A.CenterCrop(224, 224),
    ToTensorV2()
])
from typing import Dict, List, Tuple
from math import cos, pi


def proposed_lr(initial_lr, iteration, total_training_iteration):
    # proposed learning late function
    return initial_lr * (cos(pi * iteration / total_training_iteration) + 1) / 2


def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
) -> List[Tuple[str, int]]:
    # instances = []
    paths = []
    labels = []
    directory = os.path.expanduser(directory)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                # item = path, class_index
                # instances.append(item)
                paths.append(path)
                labels.append(class_index)

    return paths, labels


def split_train_val(paths, labels):
    num_train = len(paths)
    indices = np.arange(num_train)
    split = int(np.floor(0.2 * num_train))

    np.random.seed(7)
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    return np.array(paths)[train_idx].tolist(), np.array(labels)[train_idx].tolist(), np.array(paths)[
        test_idx].tolist(), np.array(labels)[test_idx].tolist()


import matplotlib.pyplot as plt
import numpy as np
from augmentation import rand_bbox


def show_sample(img):
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break


if __name__ == '__main__':
    data_dir = 'garbage/Garbage/'
    classes, class_to_idx = find_classes(data_dir)
    paths, labels = make_dataset(data_dir, class_to_idx)

    albumentations_dataset = AlbumentationsDataset(
        file_paths=paths,
        labels=labels,
        transform=albumentations_transform,
    )
    # no_transform_dataset = AlbumentationsDataset(
    #     file_paths=paths,
    #     labels=labels,
    #     transform=albumentations_transform_test
    # )

    ## augmentation 되는지 확인
    # num_samples = 5
    # fig, ax = plt.subplots(1, num_samples, figsize=(25, 5))
    # for i in range(num_samples):
    #     ax[i].imshow(transforms.ToPILImage()(albumentations_dataset[0][0]))
    #     ax[i].axis('off')
    # plt.show()

    # print(albumentations_dataset[0])

    # -- transform vis
    # for i in range(10):
    #     img, label = albumentations_dataset[i]
    #     show_sample(img)

    # -- no transform vis
    # for i in range(10):
    #     img, label = no_transform_dataset[i]
    #     show_sample(img)

    #  -- batch vis
    train_loader = DataLoader(albumentations_dataset, 16, num_workers=0, pin_memory=True,
                              shuffle=True)  # sampler=train_sampler,

    # mixup
    for inputs, labels in train_loader:
        indices = torch.randperm(inputs.size(0))
        shuffled_data = inputs[indices]
        shuffled_target = labels[indices]

        lam = np.clip(np.random.beta(1.0, 1.0), 0.3, 0.7)
        inputs = lam * inputs + (1 - lam) * shuffled_data

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(inputs, nrow=16).permute(1, 2, 0))
        plt.show()

        break