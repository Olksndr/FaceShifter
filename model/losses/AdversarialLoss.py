import os
import sys

sys.path.append('SPADE/')

import torch

from SPADE.models.networks import discriminator
from SPADE.models.networks import loss

class DummyOptions:
    num_D = 3

    netD_subarch = 'n_layer'

    ndf = 64

    label_nc = 2
    output_nc = 0
    contain_dontcare_label = False
    no_instance = False
    norm_D = 'spectralinstance'
    n_layers_D = 4

    no_ganFeat_loss = True

    gan_mode = 'hinge'

def divide_pred(pred):
    # the prediction contains the intermediate outputs of multiscale GAN,
    # so it's usually a list
    if type(pred) == list:
        fake = []
        real = []
        for p in pred:
            fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
            real.append([tensor[tensor.size(0) // 2:] for tensor in p])
    else:
        fake = pred[:pred.size(0) // 2]
        real = pred[pred.size(0) // 2:]

    return fake, real

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
