import torch

def get_adversarial_loss(inputs, positive=True):
    def hinge_loss(X, positive=True):
        if positive:
            return torch.relu(1 - X).mean()
        else:
            return torch.relu(X + 1).mean()

    loss = 0
    for scores_map in inputs:
        loss += hinge_loss(d_scores_map, positive)
    return loss

def attribute_loss(z_att_x, z_att_y, batch_size=1):
    L_attr = 0

    for i in range(len(z_att_x)):
        L_attr += torch.mean(torch.pow(z_att_x[i] - z_att_y[i], 2).reshape(batch_size, -1), dim=1).mean()

    L_attr /= 2.0

    return L_attr

def identity_loss(z_id, z_Y_id):
    return (1 - torch.cosine_similarity(z_id, z_Y_id, dim=1)).mean()

def reconstruction_loss(Y, X_t, same_person, batch_size=1):
    return torch.sum(0.5 * torch.mean(torch.pow(Y - X_t, 2)\
        .reshape(batch_size, -1), dim=1) * same_person)\
                            / (same_person.sum() + 1e-6)


def get_generator_loss(fake_discr_scores, Z_att, Z_Y_att,
                                    Z_id, Z_Y_id, Y, X_t, same):


    adversarial_loss_G = get_adversarial_loss(fake_discr_scores, positive=True)

    attribute_loss = model_losses.attribute_loss(Z_att, Z_Y_att)

    identity_loss = model_losses.identity_loss(Z_id, Z_Y_id)

    reconstruction_loss = model_losses.reconstruction_loss(Y, X_t, same)

    loss_g = 1 * adversarial_loss + 10 * attribute_loss +\
                        5 * identity_loss + 10 * reconstruction_loss

    losses = {"adversarial_loss": adversarial_loss,
            "attribute_loss": attribute_loss,
            "identity_loss": identity_loss,
            "reconstruction_loss": reconstruction_loss,
            "loss_g": loss_g}

    return loss_g, losses

def get_discriminator_loss(fake_discr_scores, true_discr_scores):
    adversarial_loss_D_fake = get_adversarial_loss(fake_discr_scores,
                                                    positive=False)
    adversarial_loss_D_true = get_adversarial_loss(true_discr_scores,
                                                    positive=True)
    loss_d = 0.5 * (adversarial_loss_D_fake + adversarial_loss_D_true)

    losses = {"adv_loss_d_fake": adversarial_loss_D_fake,
                    "adv_loss_d_true": adversarial_loss_D_true,
                    "loss_d": loss_d}

    return loss_d, losses
