import sys
sys.path.append('SPADE/')

from SPADE.models.networks import discriminator
from SPADE.models.networks import loss
# from model.losses.AdversarialLoss import DummyOptions, divide_pred

from model.losses.AttributeLoss import attribute_loss
from model.losses.IdentityLoss import identity_loss
from model.losses.ReconstructionLoss import reconstruction_loss
