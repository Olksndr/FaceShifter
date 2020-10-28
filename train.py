import torch

from model.Generator import AEINet
from model.Discriminator import Discriminator

from model import losses
from model.config import Config
from Dataset import FaceDataset

from model.IdentityEncoder import IdentityEncoder
from model.Discriminator import DummyOptions

from torch.utils.tensorboard import SummaryWriter

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

    def __init__(self, resume=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.init_optimizers()
        self.writer = SummaryWriter("summary/")
        if resume == False:
            model.generator.apply(init_weights)
            model.discriminator.apply(init_weights)
            self.step = 0
        else:
            self.load_model_from_checkpoint()


    def build_model(self):
        self.generator = AEINet(self.device)
        self.discriminator = Discriminator(self.opt).to(self.device)

        self.criterionGAN = losses.loss.GANLoss(self.opt.gan_mode,
                        self.discriminator.FloatTensor, opt=self.opt)

    def save_checkpoint(self, ckpt_to_keep=10):
        if len(os.listdir("weights")) > ckpt_to_keep:
            ckpt_files = sorted(os.listdir("weights"),
                key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
            files_to_remove = [os.path.join("weights", file) for file in\
                                            ckpt_files[:-ckpt_to_keep]]
            [*map(os.remove, files_to_remove)]

        ckpt_path = os.path.join("weights", "model_{}.ckpt".format(self.step))
        torch.save({'step': self.step,
                    'test_step': self.test_step
                    'generator_state_dict': self.generator.state_dict(),
                    'generator_opt_state_dict': self.opt_G.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'discriminator_opt_state_dict': self.opt_D.state_dict()
                    }, ckpt_path)

    def load_checkpoint(self):
        latest_ckpt_path = ckpt_files = sorted(os.listdir("weights"),
                key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))[-1]
        ckpt = torch.load(latest_ckpt_path)

        self.step = ckpt['step']
        self.test_step = ckpt['test_step']
        model.generator.load_state_dict(ckpt['generator_state_dict'])
        model.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        model.opt_G.load_state_dict(ckpt['generator_opt_state_dict'])
        model.opt_D.load_state_dict(ckpt['discriminator_opt_state_dict'])

    def perform_visualization(self, images):

        img_grid = torchvision.utils.make_grid(images, nrow=3, )
        self.writer.add_image("visualization_{}".format(step), img_grid, global_step=step)

    def log_metrics(self, losses, step, mode):
        self.writer.add_scalar("Loss/{}/adversarial_loss".format(mode), losses["adversarial_loss"], step)
        self.writer.add_scalar("Loss/{}/attribute_loss".format(mode), losses["attribute_loss"], step)
        self.writer.add_scalar("Loss/{}/identity_loss".format(mode), losses["identity_loss"], step)
        self.writer.add_scalar("Loss/{}/reconstruction_loss".format(mode), losses["reconstruction_loss"], step)
        self.writer.add_scalar("Loss/{}/loss_G".format(mode), losses["loss_G"], step)
        self.writer.add_scalar("Loss/{}/loss_D".format(mode), losses["loss_D"], step)
        self.writer.flush()


    def init_optimizers(self):
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=4e-4, betas=(0, 0.999))
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=4e-4, betas=(0, 0.999))

    def get_generator_loss(self, pred_fake, Z_att, Z_Y_att, Z_id, Z_Y_id, Y, X_t, same):

        adversarial_loss = self.criterionGAN(pred_fake, True,
                                                for_discriminator=False)
        attribute_loss = losses.attribute_loss(Z_att, Z_Y_att)
        identity_loss = losses.identity_loss(Z_id, Z_Y_id)
        reconstruction_loss = losses.reconstruction_loss(Y, X_t, same)

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
        Y, Z_att, Z_id = self.generator([X_s, X_t])

        pred_fake_g, pred_real_g = self.discriminator(torch.cat([Y, X_s], dim=0))

        Z_Y_att = self.generator.att_enc(Y)

        Z_Y_id = self.generator.identity_encoder.encode(Y, unsqueeze=False)

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
        Y, Z_att, Z_id = self.generator([X_s, X_t])

        pred_fake_g, pred_real_g = self.discriminator(torch.cat([Y, X_s], dim=0))

        Z_Y_att = self.generator.att_enc(Y)
        Z_Y_id = self.generator.identity_encoder.encode(Y, unsqueeze=False)

        loss_G, losses = self.get_generator_loss(pred_fake_g, Z_att, Z_Y_att,
                                            Z_id, Z_Y_id, Y, X_t, same)

        pred_fake_d, pred_real_d = D(torch.cat([Y.detach(), X_s], dim=0))
        loss_D = self.get_discriminator_loss()
        losses['loss_D'] = loss_D

        self.log_metrics(losses, step, mode="test")

        return Y

    def train(self):
        if not resume:
            self.step = 0
            self.test_step = 0

        while True:
            train_batch = next(self.train_loader)
            Y, loss_G, loss_D = train_step(train_batch)

            if self.step % 100 == 0:
                images = torch.cat([train_batch[:2], Y])
                self.perform_visualization()

            if self.step % 1000 == 0:
                self.save_checkpoint()
                self.generator.eval()
                self.discriminator.eval()

                for i in range(100):
                    test_batch = next(self.test_loader)
                    self.test_step += i
                    Y, loss_G, loss_D = test_batch(test_batch, self.test_step)

                self.generator.train()
                self.discriminator.train()
                
            self.step += 1
