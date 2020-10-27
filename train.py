import torch

from model.Generator import AEINet
from model.Discriminator import Discriminator

from model import losses
from model.config import Config
from Dataset import FaceDataset

from model.IdentityEncoder import IdentityEncoder

class Trainer():
    opt = DummyOptions()

    def __init__(self, device):
        self.device = device
        self.build_model()
        self.init_optimizers()

        if resume == False:
            self.init_model_weights()
        else:
            self.load_model_from_checkpoint()

    def load_model_from_checkpoint():
        "do smth about model step"
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        generator.load_state_dict(ckpt["generator"], strict=False)
        if discriminator is not None:
            discriminator.load_state_dict(ckpt["discriminator"], strict=False)
        if optimizer_g is not None:
            optimizer_g.load_state_dict(ckpt["optimizer_g"])
        if optimizer_d is not None:
            optimizer_d.load_state_dict(ckpt["optimizer_d"])

    def save_checkpoint(self, step):
    ckpt = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_g": self.opt_G.state_dict(),
            "optimizer_d": self.opt_D.state_dict(),
        }
        ckpt_path = os.path.join(self.model_dir, "model_{}.ckpt".format(step))
        torch.save(ckpt, ckpt_path)

    def build_model(self):
        self.generator = AEINet(self.device)
        self.discriminator = Discriminator(self.opt).to(self.device)

        self.criterionGAN = losses.loss.GANLoss(self.opt.gan_mode, self.discriminator.FloatTensor, opt=self.opt)

    def init_model_weights(self):
        for model in [self.generator, self.discriminator]:
            for layer in model.modules():
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    torch.nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, 0, 0.001)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

    def init_optimizers(self):
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=4e-4, betas=(0, 0.999))
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=4e-4, betas=(0, 0.999))

        self.generator.train()
        self.discriminator.train()

    def get_generator_loss(self, pred_fake, Z_att, Z_Y_att, Z_id, Z_Y_id, Y, X_t, same):

        adversarial_loss = self.criterionGAN(pred_fake, True,
                                                for_discriminator=False)

        attribute_loss = losses.attribute_loss(Z_att, Z_Y_att)

        identity_loss = losses.identity_loss(Z_id, Z_Y_id)

        reconstruction_loss = losses.reconstruction_loss(Y, X_t, same)

        loss_G = 1 * adversarial_loss + 10 * attribute_loss + 5 * identity_loss +\
                                                        10 * reconstruction_loss

        return loss_G

    def get_discriminator_loss(self, pred_real, pred_fake):
        loss_D = 0.5 * (self.criterionGAN(pred_real, True, for_discriminator=True) +\
                       self.criterionGAN(pred_fake, False, for_discriminator=True))



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

        # generator loss
        loss_G = self.get_generator_loss(pred_fake_g, Z_att, Z_Y_att, Z_id, Z_Y_id, Y, X_t, same)

        loss_G.backward()
        self.opt_G.step()

        # discriminator
        self.opt_D.zero_grad()
        pred_fake_d, pred_real_d = D(torch.cat([Y.detach(), X_s], dim=0))
        # discriminator loss
        loss_D = self.get_discriminator_loss()

        loss_D.backward()
        opt_D.step()

    def test_step(self, inputs, step):
        self.generator.eval()
        self.discriminator.eval()

        X_s, X_t, same = inputs
        Y, Z_att, Z_id = self.generator([X_s, X_t])

        pred_fake_g, pred_real_g = self.discriminator(torch.cat([Y, X_s], dim=0))

        Z_Y_att = self.generator.att_enc(Y)
        Z_Y_id = self.generator.identity_encoder.encode(Y, unsqueeze=False)

        loss_G = self.get_generator_loss(pred_fake_g, Z_att, Z_Y_att, Z_id, Z_Y_id, Y, X_t, same)

        pred_fake_d, pred_real_d = D(torch.cat([Y.detach(), X_s], dim=0))
        loss_D = self.get_discriminator_loss()

        self.generator.train()
        self.discriminator.train()


    def train_mockup():
#         for step, batch in enumerate(self.train_loader):
        while True:
            step += 1
            train_batch = next(self.train_loader)
            self.train_step(train_batch, step)
            if step % 100 == 0:
                test_batch = next(self.test_loader)
                self.test_step(test_batch, step)

            if step % 1000 == 0:

                save_checkpoint(
                    self.generator, self.discriminator, self.opt_g, self.opt_d, self.it
                )

            if step % 100 == 0:
                visualize_train

            if step % 100 == 0:
                visualize_test
