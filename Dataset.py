import csv

import json

import io
import boto3

from PIL import Image

import sys
sys.path.append('InsightFace_Pytorch/')
from mtcnn import MTCNN

from itertools import chain, count
import numpy as np
import random

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms


class FaceDataset(IterableDataset):

    same_prob = 0.8

    def __init__(self, mtcnn, batch_size, keys_file="meta/test_keys.csv"):
        super(FaceDataset, self).__init__()
        self.mtcnn = mtcnn
        self.batch_size = batch_size
        self._get_s3_creds()
        self.get_keys(keys_file)
        self.random_keys_generator = self.get_random_key_generator()

        self.iterables = [self.get_image_generator() for _ in range(batch_size)]

        self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    def _get_s3_creds(self):
        with open("meta/aws_cred.json", "r") as file:
            self.creds = json.load(file)
        self.bucket_name = 'faceshifter'

    def get_s3_client(self):
        return boto3.client("s3", aws_access_key_id=self.creds["AWSAccessKeyId"],
                                   aws_secret_access_key=self.creds["AWSSecretKey"])

    def get_keys(self, keys_file):
        self.keys = []
        with open(keys_file, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                self.keys.extend(line)

        self._keys_len = len(self.keys)

    def align_and_crop(self, pil_img):
        faces = self.mtcnn.align_multi(pil_img, min_face_size=64,
                                    crop_size=(256, 256))
        if not faces or len(faces) > 1:
            return False
        return faces

    def get_random_key_generator(self):
        while True:
            idx = np.random.choice(self._keys_len)
            yield self.keys[idx]

    def process_image(self, _):
        random_key = next(self.random_keys_generator)#[0]
        print(random_key)

        image = Image.open(img_bytes)
        image = self.align_and_crop(image)

        if (not image) or len(image) > 1:
            return self.process_image(None)
        return image

    def get_image_generator(self):
        return chain.from_iterable(map(self.process_image, count()))

    def get_batch(self, img_generator):

        if random.random() > self.same_prob:
            return self.transforms(next(img_generator)), self.transforms(next(img_generator)), np.array([0])
        else:
            X_s = self.transforms(next(img_generator))
            return X_s, X_s, np.array([1])

    def __iter__(self):
        return self

    def __next__(self):
        batch = [*map(self.get_batch, self.iterables)]

        image_1, image_2, same_person = [[batch[i][j] for i \
                    in range(self.batch_size)] for j in range(3)]

        return torch.stack(image_1), torch.stack(image_2),\
                                        torch.tensor(same_person)
