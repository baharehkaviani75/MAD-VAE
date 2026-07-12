import os
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap
from models.madvae_mnist import MADVAEMNIST
from models.madvae_resnet import MADVAEResNet

def parse_args():

    parser = argparse.ArgumentParser(
        description="UMAP visualization of MAD-VAE latent space"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=[
            'mnist',
            'cifar',
            'svhn',
            'gtsrb',
            'celeb'
        ]
    )

    parser.add_argument(
        '--attack',
        type=str,
        default='fgsm'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True
    )

    parser.add_argument(
        '--sample',
        type=int,
        default=3000
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=256
    )

    return parser.parse_args()

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

def load_model(args):

    if args.dataset == 'mnist':

        model = MADVAEMNIST(args)

    else:

        model = MADVAEResNet(args)

    checkpoint = torch.load(
        args.checkpoint,
        map_location='cuda'
    )

    model.load_state_dict(checkpoint)

    model.cuda()

    model.eval()

    return model

def get_latent(model, loader):

    z_list = []
    y_list = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.cuda()

            _, _, _, z = model(images)

            z_list.append(
                z.cpu().numpy()
            )

            y_list.append(
                labels.numpy()
            )

    z = np.concatenate(
        z_list,
        axis=0
    )

    y = np.concatenate(
        y_list,
        axis=0
    )

    return z, y

def main():

    args = parse_args()

    args = set_params(args)

    model = load_model(args)

    # load generated adversarial data
    adv = np.load(
        f'../data/advs_{args.dataset}_{args.attack}.npy'
    )

    labels = np.load(
        f'../data/ys_{args.dataset}.npy'
    )

    # random sampling
    n = min(
        args.sample,
        len(adv)
    )

    idx = np.random.choice(
        len(adv),
        n,
        replace=False
    )

    adv = adv[idx]
    labels = labels[idx]

    if adv.ndim == 2:
        adv = adv.reshape(
            adv.shape[0],
            args.image_channels,
            args.image_size,
            args.image_size
        )

    adv_tensor = torch.from_numpy(
        adv
    ).float()

    dataset = TensorDataset(
        adv_tensor,
        torch.from_numpy(labels).long()
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # ==========================
    # latent extraction
    # ==========================

    z, y = get_latent(
        model,
        loader
    )

    print(
        "latent shape:",
        z.shape
    )

    # ==========================
    # UMAP
    # ==========================

    reducer = umap.UMAP(
        n_components=2,
        random_state=42
    )

    u = reducer.fit_transform(
        z
    )

    plt.figure(
        figsize=(10,8)
    )

    plt.scatter(
        u[:,0],
        u[:,1],
        c=y,
        cmap='Spectral',
        s=14
    )

    clb = plt.colorbar(
        boundaries=np.arange(args.num_classes+1)-0.5
    )

    clb.set_ticks(
        np.arange(args.num_classes)
    )

    plt.xticks([])
    plt.yticks([])

    plt.title(
        f'UMAP embedding of {args.dataset.upper()} '
        f'classes under {args.attack.upper()}',
        fontsize=20
    )

    os.makedirs(
        'umap',
        exist_ok=True
    )

    plt.savefig(
        f'umap/{args.dataset}_{args.attack}.png',
        dpi=300,
        bbox_inches='tight'
    )

if __name__ == "__main__":

    main()