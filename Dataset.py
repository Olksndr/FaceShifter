import json

import io
import boto3

import cv2
from PIL import Image

import sys
sys.path.append('InsightFace_Pytorch/')
from mtcnn import MTCNN

from itertools import chain, count
import numpy as np

from torch.utils.data import Dataset, IterableDataset, DataLoader

class S3_operations:
    bucket_name = "faceshifter"
    def __init__(self):
        self.get_client()

    def get_client(self):
        self._get_s3_creds()
        self.client = boto3.client("s3", aws_access_key_id=self.creds["AWSAccessKeyId"],
                                   aws_secret_access_key=self.creds["AWSSecretKey"])

    def _get_s3_creds(self):
        with open("aws_cred.json", "r") as file:
            self.creds = json.load(file)

    def retrieve_object(self, key):
        img_bytes = io.BytesIO()
        self.client.download_fileobj(self.bucket_name, key, img_bytes)
        return img_bytes

    def get_page_iterator(self, Prefix, MaxKeys=1000):
        paginator = client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=Prefix, MaxKeys=1000)
        return page_iterator

class FaceDataset(torch.utils.data.IterableDataset):
    num_ffhq = 51096
    num_celeb = 29540

    min_image_b_size = 12000

    same_prob = 0.5

    def __init__(self, s3_op):
        super(FaceDataset, self).__init__()

        self.s3_op = s3_op
        self.get_vgg_folders()
        self.mtcnn = MTCNN()

        self.iterables = [self.get_image_generator() for _ in range(batch_size)]
#         self.transforms = transforms.Compose([
#                 transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                     ])

    def get_vgg_folders(self):
        with open("vgg_folders_upd.txt", "r") as file:
            vgg_folders = file.read()
            self.vgg_folders = vgg_folders.split('\n')[:-1]

    @staticmethod
    def get_arr_obj(img_bytes):
        img_bytes.seek(0)
        file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    def get_random_ffhq_image(self):

        random_ffhq_number = np.random.choice(self.num_ffhq)
        ffhq_key = "".join([ffhq_name[:-len(str(random_ffhq_number))], str(random_ffhq_number), ".png"])

        img_bytes = self.s3_op.retrieve_object(self, ffhq_key)
        img_array = self.get_arr_obj(img_bytes)

        return img_array


    def get_random_celeb_image(self):

        random_celeb = np.random.choice(self.num_celeb)
        celeb_key = "".join([str(random_celeb), ".jpg"])

        img_bytes = self.s3_op.retrieve_object(self, celeb_key)
        img_array = self.get_arr_obj(img_bytes)

        return img_array

    def align_and_crop(self, pil_img):
        faces = self.mtcnn.align_multi(pil_img, min_face_size=64, crop_size=(256, 256))
        if not faces or len(faces) > 1:
            return False

        return faces[0]

    def get_random_vgg_image(self):
        random_folder = np.random.choice(self.vgg_folders)
        page_iterator = s3_op.get_page_iterator(Prefix="vgg_raw/{}".format(folder))

        ret_keys = []
        for page in page_iterator:
            for item in page["Contents"]:
                if item["Size"] >= self.min_image_b_size:
                    ret_keys.append(item["Key"])

        arr_img = self.retrieve_and_prep_vgg(ret_keys)
        return arr_img

    def retrieve_and_prep_vgg(self, ret_keys):
        ret_keys_ = ret_keys.copy()

        random_vgg_image = np.random.choice(ret_keys)
        vgg_img_key = random_vgg_image.split('/')[-1]

        img_bytes = self.s3_op.retrieve_object(self, vgg_img_key)

        vgg_img_pil = Image.open(img_bytes)

        result = self.align_and_crop(vgg_img_pil)

        if not result:
            ret_keys_.remove(random_vgg_image)
            self.retrieve_and_prep_vgg(ret_keys_)

        return np.asarray(result)

    def process_images(self, _):
        for method in [self.get_random_ffhq_image, self.get_random_celeb_image, self.get_random_vgg_image]:
            yield method()

    def get_image_generator(self):
        return chain.from_iterable(map(self.process_data_old, count()))

    def get_batch(self, img_generator):
        ## make outputs  torch tensors
        ## apply transform!!
        if random.random() > self.same_prob:
            return([next(img_generator), next(img_generator), 0])
        else:
            X_s = next(img_generator)
            return([X_s, X_s, 1])

    def __iter__(self):
        return self

    def __next__(self):
        return [*map(self.get_batch, self.iterables)]
