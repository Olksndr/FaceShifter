import os
import sys

sys.path.append('SPADE/')

import torch

from SPADE.models.networks import discriminator
from SPADE.models.networks import loss


if __name__ =='__main__':

    opt = DummyOptions()

    MultiscaleDiscrimnator = discriminator.MultiscaleDiscriminator(opt)

    fake_x = torch.zeros([1, 3, 256, 256])
    true_x = torch.zeros([1, 3, 256, 256])

    discriminator_input = torch.cat([fake_x, true_x], dim=0)
    out = MultiscaleDiscrimnator(discriminator_input)

    pred_fake, pred_real = divide_pred(out)

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() \
            else torch.FloatTensor

    criterionGAN = loss.GANLoss(opt.gan_mode, FloatTensor, opt=opt)

    GanLoss = criterionGAN(pred_fake, True, for_discriminator=False)
