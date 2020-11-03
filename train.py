import os
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.Generator import AEINet
from model.Discriminator import Discriminator

from model import losses as model_losses
from model.config import Config
from Dataset import FaceDataset

from model.IdentityEncoder import IdentityEncoder
from model.Discriminator import DummyOptions

from torch.utils.tensorboard import SummaryWriter
import torchvision
from Dataset import S3_operations, FaceDataset

from InsightFace_Pytorch.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch.nn.functional as F
import torch

import numpy as np

def init_weights(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, 0, 0.001)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)

class Trainer():
    opt = DummyOptions()

    def __init__(self, resume=False, batch_size=2):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resume = resume

        self.build_model()
        self.init_optimizers()
        self.build_face_recognizer()
        if self.resume == False:
            self.generator.apply(init_weights)
            self.discriminator.apply(init_weights)
            self.step = 0
        else:
            self.load_checkpoint()

        self.init_datasets(batch_size)
        self.writer = SummaryWriter("summary/")

    def build_face_recognizer(self):
        self.arcface = Backbone(50, 0, 'ir_se').to(self.device)
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load('./InsightFace_Pytorch/model_ir_se50.pth', map_location='cpu'), strict=False)

    def encode_identity(self, X):
        with torch.no_grad():
            Z_id = self.arcface(F.interpolate(X, [112, 112], mode='bilinear', align_corners=True))

        return Z_id.unsqueeze(-1).unsqueeze(-1)

    def init_datasets(self, batch_size):
        test_keys, train_keys = [os.path.join("meta", file) for file in os.listdir('meta/')[1:]]

        s3_op = S3_operations()

        self.train_loader = FaceDataset(s3_op, batch_size=batch_size, keys_file=train_keys)
        self.test_loader = FaceDataset(s3_op, batch_size=batch_size, keys_file=test_keys)
        print("datasets successfully init")

    def build_model(self):
#         if not self.resume:
        self.generator = AEINet(self.device)
        self.discriminator = Discriminator(self.opt).to(self.device)

        self.criterionGAN = model_losses.loss.GANLoss(self.opt.gan_mode,
                    self.discriminator.FloatTensor, opt=self.opt)
#     else:
#             device = "cpu"
#             self.generator = AEINet(device)
#             self.discriminator = Discriminator(self.opt)#.to(device)

#             self.criterionGAN = model_losses.loss.GANLoss(self.opt.gan_mode,
#                         self.discriminator.FloatTensor, opt=self.opt)
        print("model successfully built")
    def save_checkpoint(self, ckpt_to_keep=10):
        if len(os.listdir("weights")) > ckpt_to_keep:
            ckpt_files = sorted(os.listdir("weights"),
                key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
            files_to_remove = [os.path.join("weights", file) for file in\
                                            ckpt_files[:-ckpt_to_keep]]
            [*map(os.remove, files_to_remove)]
        print("saving checkpoint")
        ckpt_path = os.path.join("weights", "model_{}.ckpt".format(self.step))
        torch.save({'step': self.step,
                    'test_step': self.step_test,
                    'generator_state_dict': self.generator.state_dict(),
                    'generator_opt_state_dict': self.opt_G.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'discriminator_opt_state_dict': self.opt_D.state_dict()
                    }, ckpt_path)

    def load_checkpoint(self):
        latest_ckpt_path = sorted(os.listdir("weights"),
                key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))[-1]

        print("loading :{}".format(latest_ckpt_path))
        torch.cuda.empty_cache()
        ckpt = torch.load(os.path.join("weights", latest_ckpt_path))

#         self.generator = self.generator.to(self.device)
#         self.discriminator = self.discriminator.to(self.device)

        self.step = ckpt['step']
        self.step_test = ckpt['test_step']
        self.generator.load_state_dict(ckpt['generator_state_dict'])
        self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        self.opt_G.load_state_dict(ckpt['generator_opt_state_dict'])
        self.opt_D.load_state_dict(ckpt['discriminator_opt_state_dict'])


        self.generator.to(self.device)
        self.generator.identity_encoder.arcface.to(self.device)
        self.discriminator.to(self.device)


    def perform_visualization(self, images, test=False):
#         img = torchvision.utils.make_grid(images)
#         np_img = np.clip(img.permute(1,2,0).cpu().numpy(), 0, 1)

#         X_s, X_t, same = next(test_dataset)

#         batch_size = 2

        def smart_stack(X_s, X_t, Y):
            for i in range(self.batch_size):
                yield torch.stack([X_s[i], X_t[i], Y[i]])

        ss = smart_stack(images[0], images[1], images[2])

        tensors = torch.cat([*ss], axis=0)

        img = torchvision.utils.make_grid(tensors, nrow=3)
        np_img = np.clip(img.permute(1,2,0).cpu().numpy(), 0, 1)
        print(np_img.shape)
#         print(np_img)
#        import numpy as np
#        np.save("images.npy", images.cpu().numpy())
#	print(np_img.shape)
        if test:
            plt.imsave("test_images/{}.png".format(self.step_test), np_img)
            return
        plt.imsave("train_images/{}.png".format(self.step), np_img)
#         img_grid = torchvision.utils.make_grid(images, nrow=3, )
#         self.writer.add_image("visualization_{}".format(self.step), np_img, global_step=self.step)

    def log_metrics(self, losses, step, mode):
        self.writer.add_scalar("{}/adversarial_loss".format(mode), losses["adversarial_loss"], step)
        self.writer.add_scalar("{}/attribute_loss".format(mode), losses["attribute_loss"], step)
        self.writer.add_scalar("{}/identity_loss".format(mode), losses["identity_loss"], step)
        self.writer.add_scalar("{}/reconstruction_loss".format(mode), losses["reconstruction_loss"], step)
        self.writer.add_scalar("Loss/{}/loss_G".format(mode), losses["loss_G"], step)
        self.writer.add_scalar("Loss/{}/loss_D".format(mode), losses["loss_D"], step)
        self.writer.flush()


    def init_optimizers(self):
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=4e-4, betas=(0, 0.999))
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=4e-4, betas=(0, 0.999))

    def get_generator_loss(self, pred_fake, Z_att, Z_Y_att, Z_id, Z_Y_id, Y, X_t, same):

        adversarial_loss = self.criterionGAN(pred_fake, True,
                                                for_discriminator=False)
        attribute_loss = model_losses.attribute_loss(Z_att, Z_Y_att)
        identity_loss = model_losses.identity_loss(Z_id, Z_Y_id)
        reconstruction_loss = model_losses.reconstruction_loss(Y, X_t, same)

        loss_G = 1 * adversarial_loss + 10 * attribute_loss +\
                        5 * identity_loss + 10 * reconstruction_loss
        losses = {"adversarial_loss": adversarial_loss,
                    "attribute_loss": attribute_loss,
                    "identity_loss": identity_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "loss_G": loss_G}

        return loss_G, losses

    def get_discriminator_loss(self, pred_real, pred_fake):
        loss_D = 0.5 * (self.criterionGAN(pred_real, True,
                                        for_discriminator=True) +\
                                            self.criterionGAN(pred_fake, False,
                                                        for_discriminator=True))
        return loss_D


    def train_step(self, inputs, step):
        X_s, X_t, same = inputs

        X_s = X_s.to(self.device)
        X_t = X_t.to(self.device)
        same = same.to(self.device)
        # generator
        self.opt_G.zero_grad()

        Z_id = self.encode_identity(X_s)
        Y, Z_att, Z_id = self.generator([Z_id, X_t])

        pred_fake_g, pred_real_g = self.discriminator(torch.cat([Y, X_s], dim=0))

        Z_Y_att = self.generator.att_enc(Y)

        Z_Y_id = self.encode_identity(Y)

        loss_G, losses = self.get_generator_loss(pred_fake_g, Z_att, Z_Y_att,
                                        Z_id, Z_Y_id, Y, X_t, same)

        loss_G.backward()

        self.opt_G.step()

        # discriminator
        self.opt_D.zero_grad()

        pred_fake_d, pred_real_d = self.discriminator(torch.cat(
                                            [Y.detach(), X_s], dim=0))

        loss_D = self.get_discriminator_loss(pred_real_d, pred_fake_d)
        losses["loss_D"] = loss_D

        self.log_metrics(losses, step, mode="train")
        # discriminator loss

        loss_D.backward()
        self.opt_D.step()

        return Y

    def test_step(self, inputs, step):
        X_s, X_t, same = inputs

        X_s = X_s.to(self.device)
        X_t = X_t.to(self.device)
        same = same.to(self.device)

        with torch.no_grad():
            Z_id = self.encode_identity(X_s)
            Y, Z_att, Z_id = self.generator([Z_id, X_t])

            pred_fake_g, pred_real_g = self.discriminator(torch.cat([Y, X_s], dim=0))

            Z_Y_att = self.generator.att_enc(Y)
            Z_Y_id = self.generator.identity_encoder.encode(Y)

            loss_G, losses = self.get_generator_loss(pred_fake_g, Z_att, Z_Y_att,
                                            Z_id, Z_Y_id, Y, X_t, same)

            pred_fake_d, pred_real_d = self.discriminator(torch.cat([Y.detach(), X_s], dim=0))
            loss_D = self.get_discriminator_loss(pred_real_d, pred_fake_d)
            losses['loss_D'] = loss_D

            self.log_metrics(losses, step, mode="test")

        return Y

    def train(self):
        if not self.resume:
            self.step = 0
            self.step_test = 0
        self.generator.train()
        self.discriminator.train()
        while True:
            start_time = time.time()
            train_batch = next(self.train_loader)
            Y = self.train_step(train_batch, self.step)
            print("train_step: {}, elapsed_time: {:.5}".format(self.step, time.time() - start_time))
            images = [train_batch[0].to(self.device), train_batch[1].to(self.device), Y.detach()]
            if self.step % 100 == 0 and self.step != 0:
                self.perform_visualization(images)


            if self.step % 1000 == 0 and self.step != 0:

                del train_batch

                for i in range(100):
                    start_time = time.time()
                    test_batch = next(self.test_loader)
                    Y = self.test_step(test_batch, self.step_test)
                    print("test_step: {}, elapsed_time: {:.5}".format(self.step_test, time.time() - start_time))
                    if i % 10 == 0:
                        images = [test_batch[0].to(self.device), test_batch[1].to(self.device), Y]
                        self.perform_visualization(images, test=True)
                    self.step_test += 1
                self.save_checkpoint()
            self.step += 1
        """
            if self.step % 100 == 0 and self.step != 0:
                images = torch.cat([train_batch[0].to(self.device), train_batch[1].to(self.device), Y])
#                self.perform_visualization(images,)


#                 self.generator.eval()
#                 self.discriminator.eval()

                for i in tqdm(range(100)):
#                     print(i)
                    test_batch = next(self.test_loader)
                    self.step_test += i
                    Y = self.test_step(test_batch, self.step_test)

#                 self.generator.train()
#                 self.discriminator.train()

                self.save_checkpoint()
             """
#            self.step += 1
