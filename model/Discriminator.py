# from model import losses
import sys
sys.path.append('SPADE/')

import torch
from SPADE.models.networks import discriminator
# from SPADE.models.networks import loss


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

class Discriminator(torch.nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() \
            else torch.FloatTensor

        self.MultiscaleDiscrimnator = discriminator.MultiscaleDiscriminator(opt)

    def forward(self, inputs):
        # discriminator_input = torch.cat([Y, X_s], dim=0)
        D_i = self.MultiscaleDiscrimnator(inputs)

        pred_fake, pred_real = divide_pred(D_i)

        return pred_fake, pred_real
