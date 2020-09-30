import torch

from model.Generator import AEINet
from model.Discriminator import Discriminator

from model import losses
from model.config import Config
from Dataset import FaceDataset

from model.IdentityEncoder import IdentityEncoder

identity_encoder = IdentityEncoder()
dataset = FaceDataset()

batch_size = 16
dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, drop_last=True)

config = Config()
G = AEINet(config, identity_encoder)

opt = losses.DummyOptions()
D = Discriminator(opt)

criterionGAN = losses.loss.GANLoss(opt.gan_mode, D.FloatTensor, opt=opt)

G.train()
D.train()

opt_G = torch.optim.Adam(G.parameters(), lr=4e-4, betas=(0, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=4e-4, betas=(0, 0.999))

for epoch in range(0, 2000):
    for i, (X_s, X_t, same) in enumerate(dl):
        print(i)
        opt_G.zero_grad()
        Y, Z_att, Z_id = G([X_s, X_t])

        pred_fake, pred_real = D(torch.cat([Y, X_s], dim=0))

        #adversarial loss
        adversarial_loss = criterionGAN(pred_fake, True,
                                        for_discriminator=False)

        #attribute loss
        Z_Y_att = G.encode_attributes(Y, return_as_dict=False)
        attribute_loss = losses.attribute_loss(Z_att, Z_Y_att)

        # identity loss
        Z_Y_id = identity_encoder.encode_identity(Y, unsqueeze=False)
        identity_loss = losses.identity_loss(Z_id, Z_Y_id)

        # reconstruction loss
        reconstruction_loss = losses.reconstruction_loss(Y, X_t, same)

        #total loss
        lossG = 1 * adversarial_loss + 10 * attribute_loss + 5 * identity_loss +\
                                                    10 * reconstruction_loss
        lossG.backward()
        opt_G.step()


        opt_D.zero_grad()
        pred_fake, pred_real = D(torch.cat([Y.detach(), X_s], dim=0))
        #discriminator loss
        lossD = 0.5 * (criterionGAN(pred_real, True, for_discriminator=True) +\
                       criterionGAN(pred_fake, False, for_discriminator=True))
        lossD.backward()
        opt_D.step()


        if i % 1000 == 0:
            torch.save(G.state_dict(), './save/G_latest.pth')
            torch.save(D.state_dict(), './save/D_latest.pth')
