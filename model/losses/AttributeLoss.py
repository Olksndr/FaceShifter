import torch
def attribute_loss(z_att_x, z_att_y, batch_size=1):
    L_attr = 0

    for i in range(len(z_att_x)):
        L_attr += torch.mean(torch.pow(z_att_x[i] - z_att_y[i], 2).reshape(batch_size, -1), dim=1).mean()

    L_attr /= 2.0

    return L_attr
