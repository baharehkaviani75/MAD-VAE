import sys
import os
import argparse
import warnings
sys.path.insert(0, os.path.abspath('..'))
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.adversarial import add_adv

# argument parser
def parse_args():
    def str2bool(s):
        return s.lower().startswith('t')
    desc = "MAD-VAE for adversarial defense"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Training epoch numbers')
    parser.add_argument('--h_dim', type=int, default=512, help='Hidden dimensions')
    parser.add_argument('--z_dim', type=int, default=256, help='Latent dimensions for images')
    parser.add_argument('--image_channels', type=int, default=3, help='Image channels')
    parser.add_argument('--image_size', type=int, default=32, help='Image size (default to be squared images)')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of image classes')
    parser.add_argument('--log_dir', type=str, default='v_logs', help='Logs directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the Adam optimizer')
    parser.add_argument('--data_root', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='pretrained_model', help='Pretrained model directory')
    parser.add_argument('--use_gpu', type=bool, default=True, help='If use GPU for training')
    parser.add_argument('--gpu_num', type=int, default=0, choices=range(0,5), help='GPU numbers available for parallel training')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--dataset', type=str, default='cifar', choices=['mnist','cifar','svhn','gtsrb','celeb'])
    parser.add_argument('--attack', type=str, default='pgd')
    parser.add_argument('--checkpoint', type=str, default='paramsit.pt')
    parser.add_argument('--celeb_root', type=str, default='CelebA')

    return parser.parse_args()

def set_dataset_params(args):

    if args.dataset == 'mnist':
        args.num_classes = 10
        args.image_channels = 1
        args.image_size = 28

    elif args.dataset == 'cifar':
        args.num_classes = 10

    elif args.dataset == 'svhn':
        args.num_classes = 10

    elif args.dataset == 'gtsrb':
        args.num_classes = 43

    elif args.dataset == 'celeb':
        args.num_classes = 2

    return args

def get_dataset(args):

    transform = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor()
    ])

    if args.dataset == 'mnist':

        return datasets.MNIST(
            root=args.data_root,
            train=True,
            download=True,
            transform=transform
        )

    elif args.dataset == 'cifar':
        return datasets.CIFAR10(
            root=args.data_root,
            train=True,
            download=True,
            transform=transform
        )

    elif args.dataset == 'svhn':
        return datasets.SVHN(
            root=args.data_root,
            split='train',
            download=True,
            transform=transform
        )

    elif args.dataset == 'gtsrb':
        return datasets.GTSRB(
            root=args.data_root,
            split='train',
            download=True,
            transform=transform
        )

    elif args.dataset == 'celeb':
        return datasets.ImageFolder(
            args.celeb_root,
            transform
        )

if __name__ == "__main__":

    args = parse_args()

    args = set_dataset_params(args)

    if args.dataset == 'mnist':
        from models.madvae_mnist import MADVAEMNIST
        model = MADVAEMNIST(args)
    else:
        from models.madvae_resnet import MADVAEResNet
        model = MADVAEResNet(args)

    checkpoint = torch.load(
        f'../plotting/pretrained_model/'
        f'{args.checkpoint}/'
        f'{args.checkpoint_file}'
    )

    model.load_state_dict(checkpoint)

    model.eval()
    model.cuda()

    dataset = get_dataset(args)

    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    xs=[]
    ys=[]
    advs=[]

    for image, label in trainloader:

        image = image.cuda()
        label = label.cuda()

        _, adv_out = add_adv(
            model.classifier,
            image,
            label,
            args.attack,
            0
        )

        xs.append(image.cpu().detach().numpy())
        ys.append(label.cpu().detach().numpy())
        advs.append(adv_out.cpu().detach().numpy())

    xt=np.concatenate(xs,axis=0)
    yt=np.concatenate(ys,axis=0)
    adv_x=np.concatenate(advs,axis=0)

    np.save(f'../data/xs_{args.dataset}.npy', xt)
    np.save(f'../data/ys_{args.dataset}.npy', yt)
    np.save(f'../data/advs_{args.dataset}_{args.attack}.npy', adv_x)
