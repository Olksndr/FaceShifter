from InsightFace_Pytorch.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch.nn.functional as F
import torch

class IdentityEncoder():
    def __init__(self, device):
        self.arcface = Backbone(50, 0.6, 'ir_se').to(device)
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load('./InsightFace_Pytorch/model_ir_se50.pth', map_location='cpu'), strict=False)

    def encode(self, X):
        with torch.no_grad():
            Z_id = self.arcface(F.interpolate(X, [112, 112], mode='bilinear', align_corners=True))

#         if unsqueeze:
#             return Z_id, Z_id.unsqueeze(-1).unsqueeze(-1)
#         else:
#             return Z_id
        return Z_id.unsqueeze(-1).unsqueeze(-1)
