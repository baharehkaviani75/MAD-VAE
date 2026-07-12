import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.madvae_mnist import MADVAEMNIST
from models.madvae_resnet import MADVAEResNet
from utils.loss_function import *
from utils.dataset import Dataset2
from utils.scheduler import MinExponentialLR

# ==========================
# arguments
# ==========================

def parse_args():

    parser = argparse.ArgumentParser(
        description="MAD-VAE vanilla training"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='gtsrb',
        choices=[
            'mnist',
            'cifar',
            'svhn',
            'gtsrb',
            'celeb'
        ]
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=20
    )

    parser.add_argument(
        '--h_dim',
        type=int,
        default=4096
    )

    parser.add_argument(
        '--z_dim',
        type=int,
        default=256
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001
    )

    parser.add_argument(
        '--closs_weight',
        type=float,
        default=0.5
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='pretrained_model'
    )

    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=True
    )

    return parser.parse_args()

# ==========================
# dataset parameters
# ==========================

def set_params(args):

    if args.dataset == 'mnist':

        args.image_channels = 1
        args.image_size = 28
        args.num_classes = 10
        args.z_dim = 128

    elif args.dataset in ['cifar','svhn']:

        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 10
        args.z_dim = 256

    elif args.dataset == 'gtsrb':

        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 43
        args.z_dim = 256

    elif args.dataset == 'celeb':

        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 2
        args.z_dim = 256

    return args

# ==========================
# model
# ==========================

def init_model(args):

    if args.dataset == 'mnist':

        model = MADVAEMNIST(args)

    else:

        model = MADVAEResNet(args)

    if args.use_gpu and torch.cuda.is_available():

        model = model.cuda()

    model.train()

    return model

# ==========================
# training
# ==========================

def train(
        args,
        dataloader,
        model,
        optimizer,
        step
):
    recon_losses = []
    img_losses = []
    kl_losses = []
    c_losses = []

    for data, label in dataloader:

        step += 1

        if torch.cuda.is_available():

            data = data.cuda()
            label = label.cuda()

        # ======================
        # vanilla
        # no adversarial attack
        # ======================

        adv_data = data

        optimizer.zero_grad()

        # forward

        output, dsm, dss, z = model(
            adv_data
        )

        distribution = Normal(
            dsm,
            dss
        )

        # reconstruction loss

        r_loss, img_recon, kld = recon_loss_function(
            output,
            data,
            distribution,
            step,
            0.1
        )

        # classification loss

        c_loss = classification_loss(
            adv_data,
            label,
            model.aclassifier
        )

        loss = (
            r_loss
            +
            args.closs_weight*c_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1
        )

        optimizer.step()

        recon_losses.append(
            loss.item()
        )

        img_losses.append(
            img_recon.item()
        )

        kl_losses.append(
            kld.item()
        )

        c_losses.append(
            c_loss.item()
        )

    return (
        recon_losses,
        img_losses,
        kl_losses,
        c_losses,
        step
    )

# ==========================
# main
# ==========================

def main():

    args = parse_args()

    args = set_params(args)

    os.makedirs(
        args.model_dir,
        exist_ok=True
    )

    # data

    data = np.load(
        f'data/xs_{args.dataset}.npy'
    )

    labels = np.load(
        f'data/ys_{args.dataset}.npy'
    )

    dataset = Dataset2(
        data,
        labels
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    model = init_model(args)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scheduler = MinExponentialLR(
        optimizer,
        gamma=0.998,
        minimum=1e-5
    )

    step = 0

    for epoch in range(
        1,
        args.epochs+1
    ):

        print(
            "Epoch:",
            epoch
        )

        recon_losses, img_losses, kl_losses, c_losses, step = train(
            args,
            dataloader,
            model,
            optimizer,
            step
        )

        print(
            "img_recon: {:.5f}, recon: {:.5f}, kl: {:.5f}, cls: {:.5f}".format(
                np.mean(img_losses),
                np.mean(recon_losses),
                np.mean(kl_losses),
                np.mean(c_losses)
            )
        )

        scheduler.step()

        if epoch % 5 == 0:

            torch.save(
                model.state_dict(),
                f'{args.model_dir}/'
                f'{args.dataset}_vanilla_epoch{epoch}.pt'
            )

    torch.save(
        model.state_dict(),
        f'{args.model_dir}/'
        f'{args.dataset}_vanilla.pt'
    )

if __name__ == '__main__':

    main()