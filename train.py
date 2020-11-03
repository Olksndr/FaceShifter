import os
import time
import torch

from tqdm import tqdm
from model.Generator import AEINet
from model.Discriminator import Discriminator
from model.losses import get_generator_loss, get_discriminator_loss
from Dataset import FaceDataset
from InsightFace_Pytorch.model import Backbone

from model.auxilary_methods import *

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

class Trainer():
    opt = DummyOptions()

    def __init__(self, resume=False, batch_size=2):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.build_model()
        self.init_optimizers()
        self.build_face_aligner()
        self.build_face_recognizer()


        if resume == False:
            self.generator.apply(init_weights)
            self.discriminator.apply(init_weights)
            self.step = 0
            self.step_test = 0
        else:
            self.step, self.step_test, self.generator, self.discriminator,\
                self.opt_g, self.opt_d = load_checkpoint(self.generator,\
                                    self.discriminator, self.opt_g, self.opt_d)

        self.init_datasets(batch_size)
        self.writer = SummaryWriter("summary/")

    def build_face_aligner(self):
        self.mtcnn = MTCNN()

    def build_face_recognizer(self):
        self.arcface = Backbone(50, 0, 'ir_se').to(self.device)
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load(
                './InsightFace_Pytorch/model_ir_se50.pth', map_location='cpu'))

    def encode_identity(self, X):
        with torch.no_grad():
            Z_id = self.arcface(F.interpolate(X, [112, 112], mode='bilinear',
                                            align_corners=True))

        return Z_id.unsqueeze(-1).unsqueeze(-1)

    def init_datasets(self):
        test_keys, train_keys = [os.path.join("meta", file) for file in\
                                    os.listdir('meta/')[1:]]

        self.train_loader = FaceDataset(self.mtccn, batch_size=self.batch_size,
                                                keys_file=train_keys)
        self.test_loader = FaceDataset(batch_size=self.batch_size,
                                                keys_file=test_keys)
        self.test_loader.same_prob = 0
        print("datasets successfully init")

    def build_model(self):

        self.generator = AEINet(self.device)
        self.discriminator = Discriminator(self.opt).to(self.device)

        print("model successfully built")

    def init_optimizers(self):
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=4e-4,
                                                            betas=(0, 0.999))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=4e-4,
                                                            betas=(0, 0.999))

    def train_step(self, inputs, step):
        Z_id = encode_identity(X_S)
        Y, Z_att = generator([Z_id, X_T])

        fake_discr_scores = discriminator(Y.detach())
        true_discr_scores = discriminator(X_T)

        # discriminator step
        opt_d.zero_grad()
        loss_d, metrics = get_discriminator_loss(fake_discr_scores, true_discr_scores)
        loss_d.backward()
        opt_d.step()

        # generator step
        opt_g.zero_grad()

        fake_discr_scores = discriminator(Y)
        Z_Y_att = generator.att_enc(Y)
        Z_Y_id = encode_identity(Y)
        loss_g, metrics_g = get_generator_loss(fake_discr_scores, Z_att,
                                            Z_Y_att, Z_id, Z_Y_id, Y, X_T, same)
        loss_g.backward()
        opt_g.step()

        metrics.update(metrics_g)
        return Y, metrics

    def test_step(self, inputs):
        X_s, X_t, same = inputs

        X_s = X_s.to(self.device)
        X_t = X_t.to(self.device)
        same = same.to(self.device)

        with torch.no_grad():
            Z_id = self.encode_identity(X_s)
            Y, Z_att = self.generator([Z_id, X_t])
            fake_discr_scores = discriminator(Y.detach())
            true_discr_scores = discriminator(X_T)
            loss_d, metrics = get_discriminator_loss(fake_discr_scores, true_discr_scores)
            fake_discr_scores = discriminator(Y)
            Z_Y_att = generator.att_enc(Y)
            Z_Y_id = encode_identity(Y)
            loss_g, metrics_g = get_generator_loss(fake_discr_scores, Z_att,
                                            Z_Y_att, Z_id, Z_Y_id, Y, X_T, same)
            metrics.update(metrics_g)
        return Y, metrics

    def train(self):

        self.generator.train()
        self.discriminator.train()

        while True:
            start_time = time.time()
            train_batch = next(self.train_loader)
            Y, metrics = self.train_step(train_batch, self.step)
            log_metrics(self.writer, metrics, self.train_step, 'train')
            print("train_step: {}, elapsed_time: {:.5}".format(self.step,
                                                    time.time() - start_time))

            if self.step % 100 == 0 and self.step != 0:
                images = [train_batch[0].to(self.device),
                            train_batch[1].to(self.device), Y.detach()]
                perform_visualization(self.batch_size, images, self.step)


            if self.step % 1000 == 0 and self.step != 0:

                # del train_batch

                for i in range(100):
                    start_time = time.time()
                    test_batch = next(self.test_loader)
                    Y, metrics = self.test_step(test_batch, self.step_test)
                    log_metrics(self.writer, metrics, self.step_test, 'test')
                    print("test_step: {}, elapsed_time: {:.5}".format(\
                                self.step_test, time.time() - start_time))
                    if i % 10 == 0:
                        images = [test_batch[0].to(self.device),
                                        test_batch[1].to(self.device), Y]
                        perform_visualization(self.batch_size, images,
                                                    self.step, self.step_test)
                    self.step_test += 1
                save_checkpoint(self.step, self.step_test, self.generator,
                                    self.opt_g, self.discriminator, self.opt_d)
            self.step += 1
