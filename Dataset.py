import os
import sys

sys.path.append('InsightFace_Pytorch/')

from torch.utils.data import DataLoader

import torch
from mtcnn import MTCNN
import cv2
import numpy as np

import PIL.Image as Image
from InsightFace_Pytorch.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from torchvision import transforms as trans

from torch.utils.data import TensorDataset
import random
import torchvision.transforms as transforms


class FaceDataset(TensorDataset):
    def __init__(self, image_folder='dataset/celeb_cropped', same_prob=0.8):
        self.same_prob = same_prob

        self.filenames = []
        for root, dirs, files in os.walk(image_folder):
            for name in files:
                if name.endswith('jpg') or name.endswith('png'):
                    p = os.path.join(root, name)
                    self.filenames.append(p)

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        X_s = cv2.imread(self.filenames[item])[:, :, ::-1]
        X_s = Image.fromarray(X_s)
        if random.random() > self.same_prob:
            random_target_img_idx = random.randint(0, len(self.filenames)-1)
            image_path = self.filenames[random_target_img_idx]
            X_t = cv2.imread(image_path)[:, :, ::-1]
            X_t = Image.fromarray(X_t)
            same_person = 0

            return self.transforms(X_s), self.transforms(X_t), same_person

        else:
            X_t = X_s.copy()
            same_person = 1

            return self.transforms(X_s), self.transforms(X_t), same_person

image_folder = 'datset/CelebAMask-HQ/CelebA-HQ-img'
same_prob = 0.8
class FaceDatasetMtcnn(TensorDataset):
    def __init__(self, image_folder=image_folder, same_prob=same_prob):
        self.same_prob = same_prob

        self.filenames = []
        for root, dirs, files in os.walk(image_folder):
            for name in files:
                if name.endswith('jpg') or name.endswith('png'):
                    p = os.path.join(root, name)
                    self.filenames.append(p)

        self.mtcnn = MTCNN()


        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, item):
        img = cv2.imread(self.filenames[item])[:, :, ::-1]
        print('get')
        if random.random() > self.same_prob:

            random_target_img_idx = random.randint(0, len(self.filenames)-1)
            image_path = self.filenames[random_target_img_idx]
            Xt = cv2.imread(image_path)[:, :, ::-1]

            same_person = 0
            X_s = self.align_and_crop(img)
            X_t = self.align_and_crop(Xt)

            if not X_s:
                item += 1
                return self.__getitem__(item)
            if not X_t:
                return self.__getitem__(item)
            return self.transforms(X_s), self.transforms(X_t), same_person
        else:
            Xt = img.copy()
            same_person = 1

            X_s = self.align_and_crop(img)
            X_t = self.align_and_crop(Xt)
            if not X_s:
                item += 1
                return self.__getitem__(item)
            if not X_t:
                return self.__getitem__(item)
            return self.transforms(X_s), self.transforms(X_t), same_person


    def align_and_crop(self, img):
        faces = self.mtcnn.align_multi(Image.fromarray(img), min_face_size=64, crop_size=(256, 256))
        if not faces:
            return False

        return faces[0]
