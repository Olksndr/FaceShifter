import torch



from model.AdaptiveAttentionalDenorm import AADGenerator
from model.AttributeEncoder import MultiLevelAttEncoder

class AEINet(torch.nn.Module):
    def __init__(self, config, identity_encoder):

        super(AEINet, self).__init__()
        self.identity_encoder = identity_encoder
        self.att_enc = MultiLevelAttEncoder(config.MLAttConfig['input_channels'], config.MLAttConfig['output_channels'])
        self.aad_generator = AADGenerator(config.AADConfig)


        # for param in self.arcface.parameters():
        #     param.requires_grad = False



    def encode_attributes(self, X, return_as_dict=True):
        Z_att = self.att_enc(X)
        if return_as_dict:
            return Z_att, {k:Z_att[k] for k in range(8)}
        else:
            return Z_att

    def forward(self, inputs):
        X_s, X_t = inputs

        Z_id_, Z_id = self.identity_encoder.encode_identity(X_s)
        print(Z_id.shape)
        Z_att, Z_att_dict = self.encode_attributes(X_t)

        Y = self.aad_generator([Z_att_dict, Z_id])

        return Y, Z_att, Z_id_
