import torch

def identity_loss(z_id, z_Y_id):
    return (1 - torch.cosine_similarity(z_id, z_Y_id, dim=1)).mean()
