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


class S3_operations:
    bucket_name = "faceshifter"
    def __init__(self):
        self.get_client()

    def get_client(self):
        self._get_s3_creds()
        self.client = boto3.client("s3", aws_access_key_id=self.creds["AWSAccessKeyId"],
                                   aws_secret_access_key=self.creds["AWSSecretKey"])

    def _get_s3_creds(self):
        with open("meta/aws_cred.json", "r") as file:
            self.creds = json.load(file)

    def retrieve_object(self, key):
        img_bytes = io.BytesIO()
        self.client.download_fileobj(self.bucket_name, key, img_bytes)
        return img_bytes

    def get_page_iterator(self, Prefix, MaxKeys=1000):
        paginator = self.client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=Prefix, MaxKeys=1000)
        return page_iterator

class FaceDataset(IterableDataset):

    same_prob = 0.8

    def __init__(self, s3_op, batch_size):
        super(FaceDataset, self).__init__()
        self.batch_size = batch_size
        self.s3_op = s3_op
        self.get_keys()
        self.random_keys_generator = self.get_random_key_generator()
        self.mtcnn = MTCNN()


        self.iterables = [self.get_image_generator() for _ in range(batch_size)]

        self.transforms = transforms.Compose([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    def align_and_crop(self, pil_img):
        faces = self.mtcnn.align_multi(pil_img, min_face_size=64, crop_size=(256, 256))
        if not faces or len(faces) > 1:
            return False

        return faces

    def get_keys(self):
        self.keys = []
        with open("meta/test_keys.csv", "r") as f:
            reader = csv.reader(f)
            for line in reader:
                self.keys.append(line)
                
        self._keys_len = len(self.keys)

    def get_random_key_generator(self):
        while True:
            idx = np.random.choice(self._keys_len)
            yield self.keys[idx]

    def process_image(self, _):
        random_key = next(self.random_keys_generator)[0]
        img_bytes = self.s3_op.retrieve_object(random_key)
        image = Image.open(img_bytes)
        result = self.align_and_crop(image)
        if (not result) or len(result) > 1:
            return self.process_image(None)

        return result

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
        
        image_1, image_2, same_person = [[batch[i][j] for i in range(self.batch_size)] for j in range(3)]

        return torch.stack(image_1), torch.stack(image_2), torch.from_numpy(np.array(same_person))
