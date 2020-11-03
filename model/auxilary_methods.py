import torch
import os
import torchvision
import numpy as np

def save_checkpoint(step, step_test, generator, opt_g, discriminator, opt_d, ckpt_to_keep=10):
    if len(os.listdir("weights")) > ckpt_to_keep:
        ckpt_files = sorted(os.listdir("weights"),
            key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
        files_to_remove = [os.path.join("weights", file) for file in\
                                        ckpt_files[:-ckpt_to_keep]]
        [*map(os.remove, files_to_remove)]

    print("saving checkpoint")
    ckpt_path = os.path.join("weights", "model_{}.ckpt".format(self.step))
    torch.save({'step': self.step,
                'test_step': self.step_test,
                'generator_state_dict': self.generator.state_dict(),
                'generator_opt_state_dict': self.opt_G.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'discriminator_opt_state_dict': self.opt_D.state_dict()
                }, ckpt_path)

def load_checkpoint(generator, discriminator, opt_g, opt_d):
    latest_ckpt_path = sorted(os.listdir("weights"),
            key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))[-1]

    print("loading :{}".format(latest_ckpt_path))
    torch.cuda.empty_cache()
    ckpt = torch.load(os.path.join("weights", latest_ckpt_path))

    step = ckpt['step']
    step_test = ckpt['test_step']
    generator.load_state_dict(ckpt['generator_state_dict'])
    discriminator.load_state_dict(ckpt['discriminator_state_dict'])
    opt_g.load_state_dict(ckpt['generator_opt_state_dict'])
    opt_d.load_state_dict(ckpt['discriminator_opt_state_dict'])

    return step, step_test, generator, discriminator, opt_g, opt_d


def log_metrics(writer, losses, step, mode):
    writer.add_scalar("{}/adversarial_loss".format(mode), losses["adversarial_loss"], step)
    writer.add_scalar("{}/attribute_loss".format(mode), losses["attribute_loss"], step)
    writer.add_scalar("{}/identity_loss".format(mode), losses["identity_loss"], step)
    writer.add_scalar("{}/reconstruction_loss".format(mode), losses["reconstruction_loss"], step)
    writer.add_scalar("{}/adv_loss_d_fake".format(mode), losses["adv_loss_d_fake"], step)
    writer.add_scalar("{}/adv_loss_d_fake".format(mode), losses["adv_loss_d_true"], step)
    writer.add_scalar("{}/loss_G".format(mode), losses["loss_g"], step)
    writer.add_scalar("{}/loss_D".format(mode), losses["loss_d"], step)
    writer.flush()

def perform_visualization(batch_size, images, step, step_test=False):

    def smart_stack(X_s, X_t, Y):
        for i in range(batch_size):
            yield torch.stack([X_s[i], X_t[i], Y[i]])

    ss = smart_stack(images[0], images[1], images[2])

    tensors = torch.cat([*ss], axis=0)

    img = torchvision.utils.make_grid(tensors, nrow=3)
    np_img = np.clip(img.permute(1,2,0).cpu().numpy(), 0, 1)

    if test:
        plt.imsave("test_images/{}.png".format(step_test), np_img)
        return
    plt.imsave("train_images/{}.png".format(step), np_img)


def init_weights(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, 0, 0.001)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
