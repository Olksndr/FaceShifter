import torch

from model.IdentityEncoder import IdentityEncoder
from model.AAD import AADGenerator
from model.AttributeEncoder import MultiLevelAttEncoder

class AEINet(torch.nn.Module):
    def __init__(self, device):

        super(AEINet, self).__init__()
#         self.identity_encoder = IdentityEncoder(device)
        self.att_enc = MultiLevelAttEncoder().train().to(device)
        self.aad_generator = AADGenerator().train().to(device)

    def forward(self, inputs):
        Z_id, X_t = inputs

#         Z_id = self.identity_encoder.encode(X_s)
        Z_att = self.att_enc(X_t)
        Y = self.aad_generator([Z_att, Z_id])

        return Y, Z_att, Z_id
