from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from utils.ws_augmentation import TransformFixMatchMedium
import os


class Animal10N_dataset(Dataset):

    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        self.mode = mode
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_fixmatch = TransformFixMatchMedium((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214), 3, 10)

        self.train_dir = root_dir + '/training/'
        self.test_dir = root_dir + '/testing/'
        train_imgs = os.listdir(self.train_dir)
        test_imgs = os.listdir(self.test_dir)

        # Initialize lists to store image paths, labels, and tensors
        self.train_imgs = []
        self.test_imgs = []
        self.train_targets = []
        self.test_targets = []

        # Process training images (without loading the images yet)
        for img in train_imgs:
            label = int(img[0])
            self.train_imgs.append([img, label])
            self.train_targets.append(label)

        # Process testing images (without loading the images yet)
        for img in test_imgs:
            label = int(img[0])
            self.test_imgs.append([img, label])
            self.test_targets.append(label)

        # Set targets and data attribute based on mode
        if mode == 'train':
            self.data = self.train_imgs  # Store train image paths and labels
            self.targets = self.train_targets
        elif mode == 'test':
            self.data = self.test_imgs  # Store test image paths and labels
            self.targets = self.test_targets

    def __getitem__(self, index):
        # Get the data corresponding to the index
        if self.mode == 'train':
            img_id, target = self.train_imgs[index]  # Get image filename and label
            img_path = self.train_dir + img_id  # Construct image path
            image = Image.open(img_path).convert('RGB')  # Load image
            img_w = self.transform_fixmatch.weak(image)  # Apply weak transform
            img_s = self.transform_fixmatch.strong(image)  # Apply strong transform
            return img_w, img_s, target, index

        elif self.mode == 'test':
            img_id, target = self.test_imgs[index]  # Get image filename and label
            img_path = self.test_dir + img_id  # Construct image path
            image = Image.open(img_path).convert('RGB')  # Load image
            img = self.transform_test(image)  # Apply test transform
            return img, target, index

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        elif self.mode == 'train':
            return len(self.train_imgs)



if __name__ == '__main__':
    animaldata = Animal10N_dataset(root_dir='./Animal10N/', mode='train')