from argparse import Namespace
import os
import hashlib
from os.path import join as pjoin
import random
import sys
from typing import (
    Iterable,
    Optional,
)

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter

# Enable cuDNN auto-tuner for fixed-size convolutions (StyleGAN2 uses fixed resolutions).
# This benchmarks different algorithms on first call, then caches the fastest for subsequent calls.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
from torchvision.transforms import (
    Compose,
    Grayscale,
    Resize,
    ToTensor,
    Normalize,
)

from losses.joint_loss import JointLoss
from model import Generator
from tools.initialize import Initializer
from tools.match_skin_histogram import match_skin_histogram
from utils.projector_arguments import ProjectorArguments
from utils import torch_helpers as th
from utils.torch_helpers import make_image
from utils.misc import stem
from utils.optimize import Optimizer
from models.degrade import (
    Degrade,
    Downsample,
)


def set_random_seed(seed: int):
    # FIXME (xuanluo): this setup still allows randomness somehow
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def compact_run_tag(input_stem: str, opt_str: str, max_stem: int = 40) -> str:
    stem_part = (input_stem or "input")[:max_stem]
    return f"{stem_part}-cfg{short_hash(input_stem + '|' + opt_str)}"


def read_images(paths: Iterable[str], max_size: Optional[int] = None):
    transform = Compose(
        [
            Grayscale(),
            ToTensor(),
        ]
    )

    imgs = []
    for path in paths:
        img = Image.open(path)
        if max_size is not None and img.width > max_size:
            img = img.resize((max_size, max_size))
        img = transform(img)
        imgs.append(img)
    imgs = torch.stack(imgs, 0)
    return imgs


def normalize(img: torch.Tensor, mean=0.5, std=0.5):
    """[0, 1] -> [-1, 1]"""
    return (img - mean) / std


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def create_generator(args: Namespace, device: torch.device):
    generator = Generator(args.generator_size, 512, 8)
    ckpt = _load_checkpoint(args.ckpt)
    generator.load_state_dict(ckpt['g_ema'], strict=False)
    generator.eval()
    generator = generator.to(device)
    return generator


def save(
        path_prefixes: Iterable[str],
        imgs: torch.Tensor,  # BCHW
        latents: torch.Tensor,
        noises: torch.Tensor,
        imgs_rand: Optional[torch.Tensor] = None,
):
    assert len(path_prefixes) == len(imgs) and len(latents) == len(path_prefixes)
    if imgs_rand is not None:
        assert len(imgs) == len(imgs_rand)
    imgs_arr = make_image(imgs)
    for path_prefix, img, latent, noise in zip(path_prefixes, imgs_arr, latents, noises):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        cv2.imwrite(path_prefix + ".png", img[...,::-1].copy())
        torch.save({"latent": latent.detach().cpu(), "noise": noise.detach().cpu()},
                path_prefix + ".pt")

    if imgs_rand is not None:
        imgs_arr = make_image(imgs_rand)
        for path_prefix, img in zip(path_prefixes, imgs_arr):
            cv2.imwrite(path_prefix + "-rand.png", img[...,::-1].copy())


def main(args):
    opt_str = ProjectorArguments.to_string(args)
    print(opt_str)
    input_stem = stem(args.input)
    run_tag = compact_run_tag(input_stem, opt_str)

    if args.rand_seed is not None:
        set_random_seed(args.rand_seed)
    device = th.device()

    # read inputs. TODO imgs_orig has channel 1
    imgs_orig = read_images([args.input], max_size=args.generator_size).to(device)
    imgs = normalize(imgs_orig)  # actually this will be overwritten by the histogram matching result

    # initialize
    with torch.no_grad():
        init = Initializer(args).to(device)
        latent_init = init(imgs_orig)

    # create generator
    generator = create_generator(args, device)

    # init noises
    with torch.no_grad():
        noises_init = generator.make_noise()

    # create a new input by matching the input's histogram to the sibling image
    with torch.no_grad():
        sibling, _, sibling_rgbs = generator([latent_init], input_is_latent=True, noise=noises_init)
    mh_dir = pjoin(args.results_dir, f"mh_{short_hash(input_stem)}")
    imgs = match_skin_histogram(
        imgs, sibling,
        args.spectral_sensitivity,
        pjoin(mh_dir, "input_sibling"),
        pjoin(mh_dir, "skin_mask"),
        matched_hist_fn=pjoin(mh_dir, f"matched_{args.spectral_sensitivity}.png"),
        normalize=normalize,
    ).to(device)
    # TODO imgs has channel 3

    degrade = Degrade(args).to(device)

    rgb_levels = generator.get_latent_size(args.coarse_min) // 2 + len(args.wplus_step) - 1
    criterion = JointLoss(
            args, imgs,
            sibling=sibling.detach(), sibling_rgbs=sibling_rgbs[:rgb_levels]).to(device)

    # save initialization
    save(
        [pjoin(args.results_dir, f"{run_tag}-init")],
        sibling, latent_init, noises_init,
    )

    writer = SummaryWriter(pjoin(args.log_dir, run_tag))
    try:
        # start optimize
        latent, noises = Optimizer.optimize(generator, criterion, degrade, imgs, latent_init, noises_init, args, writer=writer)
    finally:
        writer.close()

    # generate output
    img_out, _, _ = generator([latent], input_is_latent=True, noise=noises)
    img_out_rand_noise, _, _ = generator([latent], input_is_latent=True)
    # save output
    save(
        [pjoin(args.results_dir, run_tag)],
        img_out, latent, noises,
        imgs_rand=img_out_rand_noise
    )


def parse_args():
    return ProjectorArguments().parse()

if __name__ == "__main__":
    sys.exit(main(parse_args()))

