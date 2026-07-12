import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from models.madvae_mnist import MADVAEMNIST
from models.madvae_resnet import MADVAEResNet
from utils.dataset import Dataset2
from utils.loss_function import *
from utils.adversarial import add_adv
from utils.scheduler import MinExponentialLR

# ==========================
# arguments
# ==========================

def parse_args():

    parser = argparse.ArgumentParser(description="MAD-VAE training")
    parser.add_argument('--dataset', type=str, default='cifar', choices=['mnist','cifar','svhn','gtsrb','celeb'])
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Training epoch numbers')
    parser.add_argument('--z_dim', type=int, default=256, help='Latent dimensions for images')
    parser.add_argument('--h_dim', type=int, default=512, help='Hidden dimensions')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the Adam optimizer')
    parser.add_argument('--closs_weight', type=float, default=0.5, help='Weight for classification loss functions')
    parser.add_argument('--model_dir', type=str, default='pretrained_model', help='Pretrained model directory')
    parser.add_argument('--attack', type=str, default='pgd')

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

    elif args.dataset == 'gtsrb':

        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 43

    elif args.dataset == 'celeb':

        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 2

    return args

# ==========================
# model
# ==========================

def init_model(args):

    if args.dataset == 'mnist':

        model = MADVAEMNIST(args)

    else:

        model = MADVAEResNet(args)

    model.cuda()

    model.train()

    return model

# ==========================
# training
# ==========================

def train_epoch(
        args,
        model,
        loader,
        optimizer
):

    recon_losses=[]
    img_losses=[]
    kl_losses=[]

    for image,label in loader:

        image=image.cuda()
        label=label.cuda()

        # generate adversarial sample
        _,adv_image = add_adv(
            model.classifier,
            image,
            label,
            args.attack,
            0
        )

        adv_image = adv_image.cuda()

        optimizer.zero_grad()

        # MAD-VAE forward

        output,mu,std,z = model(
            adv_image
        )

        distribution = Normal(
            mu,
            std
        )

        # reconstruction + KL

        r_loss,img_recon,kld = recon_loss_function(
            output,
            image,
            distribution,
            0,
            0.1
        )

        # classifier loss

        c_loss = classification_loss(
            adv_image,
            label,
            model.classifier
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
            r_loss.item()
        )

        img_losses.append(
            img_recon.item()
        )

        kl_losses.append(
            kld.item()
        )

    return (
        np.mean(recon_losses),
        np.mean(img_losses),
        np.mean(kl_losses)
    )

# ==========================
# main
# ==========================

def main():

    args=parse_args()

    args=set_params(args)

    os.makedirs(
        args.model_dir,
        exist_ok=True
    )

    # load clean data

    x=np.load(
        f'data/xs_{args.dataset}.npy'
    )

    y=np.load(
        f'data/ys_{args.dataset}.npy'
    )

    dataset=Dataset2(
        x,
        y
    )

    loader=DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    model=init_model(args)

    optimizer=optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scheduler=MinExponentialLR(
        optimizer,
        gamma=0.998,
        minimum=1e-5
    )

    for epoch in range(
        1,
        args.epochs+1
    ):

        print(
            "Epoch:",
            epoch
        )

        r,img,kld=train_epoch(
            args,
            model,
            loader,
            optimizer
        )

        print(
            f"recon={r:.5f} "
            f"img={img:.5f} "
            f"kl={kld:.5f}"
        )

        scheduler.step()

        if epoch%5==0:

            torch.save(
                model.state_dict(),
                f'{args.model_dir}/'
                f'{args.dataset}_{args.attack}_epoch{epoch}.pt'
            )

    torch.save(
        model.state_dict(),
        f'{args.model_dir}/'
        f'{args.dataset}_{args.attack}.pt'
    )

if __name__=="__main__":

    main()