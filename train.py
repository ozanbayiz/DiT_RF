import yaml
import os
import os.path as osp
import logging
import argparse
from collections import OrderedDict
from copy import deepcopy
from time import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sample import sample_images_as_grid
from model.DiT import DiT
from model.RF import RF

### Heavily based on: https://github.com/facebookresearch/DiT/blob/main/train.py

### helper functions
### ### ### ### ### ### ### ### 

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if param.requires_grad: # prevent updates to positional embeddings
            ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)

def create_logger(logging_dir="./", rank=0):
    # https://github.com/facebookresearch/DiT/blob/main/train.py
    # real logger
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)

    # dummy logger
    else:  
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def train(config_path):

    ### initialization
    ### ### ### ### ### ### ### ### 

    # load config
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return
    data_transform = config['data_transform']
    dataset_config = config['dataset']
    model_config = config['model']
    train_config = config['training']
    sample_config = config['sampling']
    
    # make folders for checkpoints & samples
    os.makedirs(train_config['checkpoint_dir'], exist_ok=True)
    os.makedirs(sample_config['sample_dir'], exist_ok=True)

    train_ema = train_config['train_ema']
    train_distributed = train_config['distributed']

    # set device
    if train_distributed:
        assert train_config['num_workers'] > 0, "num_workers must be greater than 0 for distributed training"
        assert torch.cuda.is_available(), "CUDA is required for distributed training"
        # more stuff to do if distributed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler

        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = train_config['global_seed'] * rank
        torch.cuda.set_device(device)
        torch.manual_seed(seed)

        print(f"rank: {rank}, seed: {seed}")
        logger = create_logger(rank=rank)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        torch.manual_seed(train_config['global_seed'])
        logger = create_logger()

    # create dataset transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(data_transform['padding']),
        transforms.Normalize(
            mean=([data_transform['normalize']['mean']] * dataset_config['image_channels']), 
            std=([data_transform['normalize']['std']] * dataset_config['image_channels']), 
            inplace=True
        )
    ])

    # load dataset
    if dataset_config["data_dir"]:
        # load custom dataset
        from torchvision.datasets import ImageFolder
        try:
            dataset = ImageFolder(dataset_config["data_dir"], transform=transform)
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    else:
        # load from torchvision.datasets
        if dataset_config['name'] == 'MNIST':
            dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        elif dataset_config['name'] == 'FashionMNIST':
            dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
        elif dataset_config['name'] == 'CIFAR10':
            dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        else:
            raise ValueError(f"Dataset {dataset_config['name']} not supported")
    logger.info(f"loaded dataset {dataset_config['name']} with {len(dataset)} samples")
    
    # initialize dataloader
    if train_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=train_config['global_seed']
        )
        dataloader = DataLoader(
            dataset,
            batch_size=train_config['batch_size'] // dist.get_world_size(), 
            sampler=sampler,
            num_workers=train_config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
    else:
        dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True)

    # initialize DiT / RF
    model = DiT(**model_config).to(device)
    if train_distributed:
        model = DDP(model, device_ids=[rank])
        module = model.module
    else:
        module = model
    rf = RF(model, distributed=train_distributed)

    # initialize EMA    
    if train_ema:
        ema = deepcopy(model).to(device)
        # no gradients for EMA model
        for param in ema.parameters():
            param.requires_grad = False
        ema_rf = RF(ema)
        if train_distributed:
            update_ema(ema, model.module, decay=0)
        else:
            update_ema(ema, model, decay=0)
        ema.eval()

    # initialize optimizer
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config['learning_rate'], 
        weight_decay=train_config['weight_decay']
    )


    ### training loop
    ### ### ### ### ### ### ### ### 

    model.train()
    running_loss = 0.0
    train_steps = 0
    log_steps = 0
    start_time = time()

    for epoch in range(train_config['num_epochs']):
        logger.info(f"starting epoch {epoch+1}/{train_config['num_epochs']}")

        for x, y in dataloader:
            # forward pass
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = rf.forward(x, y) # criterion applied in RF's forward method
            loss.backward()
            optim.step()

            if train_ema:
                if train_distributed:
                    update_ema(ema, model.module, decay=0.999)
                else:
                    update_ema(ema, model, decay=0.999)

            # logging
            running_loss += loss.item()
            train_steps += 1
            log_steps += 1
            if (
                train_config['log_every'] and 
                log_steps % train_config['log_every'] == 0 and 
                log_steps > 0
            ):
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # reduce loss history over all processes
                mean_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss = mean_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) loss: {mean_loss:.4f}, train steps/sec: {steps_per_sec:.2f}")
                # reset monitoring variables
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            # take checkpoint
            if (
                train_config["checkpoint_every"] and 
                train_steps % train_config['checkpoint_every'] == 0 and 
                train_steps > 0
            ):
                checkpoint = {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "train_steps": train_steps,
                    "model_config": model_config,
                }
                if train_ema:
                    checkpoint['ema'] = ema.state_dict()
                torch.save(checkpoint, osp.join(train_config['checkpoint_dir'], f"checkpoint_{train_steps}.pt"))

        # sample images
        if train_config['sample_after_epoch']:
            # sample images from EMA model,
            if train_ema:
                sample_images_as_grid(
                    rf=ema_rf,
                    device=device,
                    image_shape=(model_config['in_channels'], model_config['in_height'], model_config['in_width']),
                    num_images=sample_config['num_images'],
                    num_rows=sample_config['num_rows'],
                    num_classes=model_config['num_classes'],
                    cond=sample_config['class_labels'],
                    steps=sample_config['sample_steps'],
                    cfg_scale=sample_config['cfg_scale'],
                    name=f"model_{epoch+1}_epochs",
                    sample_dir=sample_config['sample_dir'],
                    save_gif=sample_config['save_gif'],
                )

            # sample images from training model
            model.eval()
            sample_images_as_grid(
                rf=rf,
                device=device,
                image_shape=(model_config['in_channels'], model_config['in_height'], model_config['in_width']),
                num_images=sample_config['num_images'],
                num_rows=sample_config['num_rows'],
                num_classes=model_config['num_classes'],
                cond=sample_config['class_labels'],
                steps=sample_config['sample_steps'],
                cfg_scale=sample_config['cfg_scale'],
                name=f"model_{epoch+1}_epochs",
                sample_dir=sample_config['sample_dir'],
                save_gif=sample_config['save_gif'],
            )
            model.train()
            logger.info(f"sampled {sample_config['num_images']} images from model at epoch {epoch+1}")
            

    # training complete
    logger.info("training complete!")
    model.eval()
    model = model.to("cpu")
    final_checkpoint = {
        "config": model_config,
    }
    if train_distributed:
        final_checkpoint['model'] = model.module.state_dict()
    else:
        final_checkpoint['model'] = model.state_dict()
    if train_ema:
        final_checkpoint['ema'] = ema.state_dict()
    torch.save(final_checkpoint, osp.join(train_config['checkpoint_dir'], f"{train_config['checkpoint_name']}.pt"))

    if train_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml", help="path to config file")
    args = parser.parse_args()
    train(args.config)