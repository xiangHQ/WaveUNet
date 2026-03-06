import glob
import os, sys
import json
import time
import torch
from torch import optim
import torch.nn as nn
import timeit
import math
import numpy as np
import torch.nn.functional as F
from scipy.io import savemat
# from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from utils.utils import setup_seed, init_weight, netParams
from torch.utils.data import DataLoader

from builders.models import creat_model
from builders.datasets import create_dataset


def load_partial_weights(model, checkpoint_path, device='cpu', verbose=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    old_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = model.state_dict()
    matched_weights = {}
    unmatched_keys = []
    for k, v in old_state_dict.items():
        if k in new_state_dict and v.size() == new_state_dict[k].size():
            matched_weights[k] = v
        else:
            unmatched_keys.append(k)
    if verbose:
        print(f"✅ Successfully loaded {len(matched_weights)} / {len(new_state_dict)} Layer parameters")
        if unmatched_keys:
            print(f"⚠️ Skipped {len(unmatched_keys)} Mismatched parameters, such as：")
            for name in unmatched_keys[:5]:
                print(f"  - {name}")
    new_state_dict.update(matched_weights)
    model.load_state_dict(new_state_dict)
    return model



def train_model(args):

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"=====> Using equipment: {device}")

    # Process input size
    if isinstance(args.input_size, str):
        input_size = tuple(map(int, args.input_size.split(',')))
    else:
        input_size = args.input_size
    print(f"=====> input_size: {input_size}")

    cudnn.enabled = True
    

    # Create model save directory
    model_dir = os.path.join(args.savedir, args.dataset, args.model)
    os.makedirs(model_dir, exist_ok=True)
    print(f"=====> model_dir: {model_dir}")

    # Setup TensorBoard logging
    log_dir = os.path.join(model_dir, 'logs')
    # writer = SummaryWriter(log_dir=log_dir)

    # Save configuration parameters
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        config_dict = vars(args)
        json.dump(config_dict, f, indent=4)

    # Load datasets
    train_dataset = create_dataset(args, 'train')
    val_dataset = create_dataset(args, 'val')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize model
    model = creat_model(args.model)
    model = model.to(device)
    init_weight(model, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')
    print(args.model)
    
    total_paramters = netParams(model) 
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # Use multi-GPU if available
    if args.cuda and torch.cuda.device_count() > 1:
        print(f"=====> use {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Define loss function and optimizer
    if args.dataset == 'InSAR-DLPU':
        criterion = nn.L1Loss(reduction='mean')
    elif args.dataset == 'phaseUnwrapping':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError(
            "not support dataset: %s" % args.dataset)
    print(f"=====> criterion: {criterion}")

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-4)
    else:
        raise NotImplementedError(
            "not supported: %s" % args.optim)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.isfile(args.resume):
        print(f"=====> Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model = load_partial_weights(model, args.resume, device='cuda')
        # model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # best_val_loss = checkpoint['best_val_loss']
        print(f"=====> from epoch {start_epoch} Continue training")
        
    total_batches = len(train_loader)

    total_train_start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, args.max_epochs):
        print(f"=====> Epoch {epoch + 1}/{args.max_epochs}")

        # Training phase
        model.train()
        cudnn.benchmark = True
        train_loss = 0.0
        train_start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            args.per_iter = total_batches
            args.max_iter = args.max_epochs * args.per_iter
            args.cur_iter = epoch * args.per_iter + batch_idx

            # Learning rate scheduler
            lambda1 = lambda epoch: math.pow((1 - (args.cur_iter / args.max_iter)), args.poly_exp)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

            # Record current learning rate
            lr = optimizer.param_groups[0]['lr']
            
            current_iter = epoch * args.per_iter + batch_idx
            args.cur_iter = current_iter

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"=====> train_batch [{batch_idx}/{len(train_loader)}], loss: {loss.item():.4f}, "
                      f"lr: {lr:.6f}, "
                      f"Global Iteration: {current_iter}/{args.max_iter}")

        train_loss /= len(train_loader)
        train_time = time.time() - train_start_time
        final_lr = optimizer.param_groups[0]['lr']
    
        print(f"=====> Epoch {epoch+1} finish, "
              f"Average loss: {train_loss:.4f}, "
              f"lr: {final_lr:.2e}, "
              f"time: {train_time:.2f}s")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_start_time = time.time()

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                rmse_loss = torch.sqrt(loss)
                val_loss += rmse_loss.item()
        val_loss /= len(val_loader)
        val_time = time.time() - val_start_time

        # Print progress
        print(f"=====> train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        print(f"=====> train_time: {train_time:.2f}s, val_time: {val_time:.2f}s")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"=====> New best validation loss: {best_val_loss:.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        torch.save(checkpoint, os.path.join(model_dir, 'latest_checkpoint.pth'))

        if is_best:
            torch.save(checkpoint, os.path.join(model_dir, 'best_model.pth'))

        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(model_dir, f'model_{epoch + 1:04d}.pth'))

        with open(os.path.join(model_dir, args.logFile), 'a') as f:
            f.write(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Best Val Loss: {best_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.4f}, "
                    f"Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s\n")

    total_train_time = time.time() - total_train_start_time
    total_train_time_hours = total_train_time / 3600
    print(f"\=====> nTotal training time: {total_train_time:.2f} seconds ({total_train_time_hours:.2f} hours)")
    
    # Write total training time to log
    with open(os.path.join(model_dir, args.logFile), 'a') as f:
        f.write(f"\nTotal Training Time: {total_train_time:.2f}s ({total_train_time_hours:.2f}h)\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")

    print("=====> Training completed!")
    # writer.close()

    return best_val_loss


if __name__ == '__main__':
    parser = ArgumentParser(description='/PUNet/TransUNet2/PPMNet')
    # model and dataset
    parser.add_argument('--model', type=str, default="WaveUNet", help="model name")
    parser.add_argument('--dataRootDir', type=str, default=r"/root/autodl-tmp/checkpoint/phaseUnwrapping", help="dataset dir")
    parser.add_argument('--dataset', type=str, default="phaseUnwrapping", help="dataset")
    parser.add_argument('--input_size', type=str, default="256,256", help="input size of model")
    parser.add_argument('--num_workers', type=int, default=24, help=" the number of parallel threads")
    parser.add_argument('--num_channels', type=int, default=1, help="the number of input channels ")
    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=200, help="the number of epochs")
    parser.add_argument('--random_mirror', type=bool, default=True, help="input image random mirror")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size ")
    parser.add_argument('--optim', type=str.lower, default='adam', choices=['sgd', 'adam'], help="select optimizer")
    parser.add_argument('--poly_exp', type=float, default=0.95, help='polynomial LR exponent')
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")
    # checkpoint and log
    parser.add_argument('--resume', type=str,
                        default=r"/root/autodl-tmp/checkpoint/phaseUnwrapping/phaseUnwrapping/WaveUNet/latest_checkpoint.pth",
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--savedir', default=r"/root/autodl-tmp/checkpoint/phaseUnwrapping",
                        help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    args = parser.parse_args()
    train_model(args)