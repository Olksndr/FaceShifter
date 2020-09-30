import torch

def reconstruction_loss(Y, X_t, same_person, batch_size=1):

 return torch.sum(0.5 * torch.mean(torch.pow(Y - X_t, 2).reshape(batch_size, -1),
                            dim=1) * same_person) / (same_person.sum() + 1e-6)
