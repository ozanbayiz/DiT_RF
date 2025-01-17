import torch
import yaml
import argparse
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

from model.DiT import DiT
from model.RF import RF

def sample_images_as_grid(
    rf, 
    device,
    image_shape, 
    num_images=16, 
    num_rows=4, 
    num_classes=10,
    cond = None,
    steps=50, 
    cfg_scale=2.0,
    name="", 
    sample_dir="", 
    save_gif=False,
):
    """ convenience method to sample images from the model"""

    # sample num_images images
    if cond is None:
        cond = torch.arange(num_images, device=device) % num_classes
    else:
        num_images = len(cond)
        cond = torch.tensor(cond, device=device)
        
    z_1 = torch.randn(num_images, *image_shape, device=device)
    imgs = rf.sample(z_1, cond, steps, cfg_scale)

    # create gif
    gif = []
    for img_t in imgs:
        # unnormalize image
        img = img_t * 0.5 + 0.5
        img = img.clamp(0, 1)
        # rearrange into grid
        img_grid = make_grid(img.float(), nrow=num_rows)  # (c, h, w)
        img = img_grid.permute(1, 2, 0).cpu().numpy()  # (h, w, c)
        img = (img * 255).astype(np.uint8)
        gif.append(Image.fromarray(img))

    if save_gif:
        gif[0].save(
            f"{sample_dir}/{name}_steps.gif",
            save_all=True,
            append_images=gif[1:],
            duration=1,
            loop=0,
        )

    # save final result
    last_img = gif[-1]
    last_img.save(f"{sample_dir}/{name}_result.png")


def sample(config_path):

    # load config
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return
    # load sampling config
    sample_config = config['sampling']
    # get checkpoint path
    checkpoint_path = sample_config['checkpoint_path']
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    # load model's config and state dict from checkpoint
    model_config = checkpoint['config']
    model_state_dict = checkpoint['model']

    # configure device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # initialize model
    model = DiT(**model_config).to(device)
    rf = RF(model)
    model.load_state_dict(model_state_dict)

    # sample images
    image_shape = (model_config['in_channels'], model_config['in_height'], model_config['in_width'])
    sample_images_as_grid(
        rf=rf, 
        device=device,
        image_shape=image_shape,
        num_images=sample_config['num_images'],
        num_rows=sample_config['num_rows'],
        num_classes=model_config['num_classes'],
        cfg_scale=sample_config['cfg_scale'],
        cond=sample_config['class_labels'],
        steps=sample_config['sample_steps'],
        sample_dir=sample_config['sample_dir'],
        name=sample_config['name'],
        save_gif=sample_config['save_gif'],
    )
    num_images = len(sample_config['class_labels']) if sample_config['class_labels'] else sample_config['num_images']
    print(f"successfully sampled {num_images} images and saved to {sample_config['sample_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml", help="path to config file")
    args = parser.parse_args()
    sample(args.config)