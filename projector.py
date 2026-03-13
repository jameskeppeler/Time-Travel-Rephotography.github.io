from argparse import Namespace
import json
import os
import hashlib
from os.path import join as pjoin
import random
import sys
import time
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


class TimingLog:
    """Structured JSONL timing log for profiling pipeline stages."""
    def __init__(self, log_path: Optional[str] = None):
        self._path = log_path
        self._t0 = time.perf_counter()
        self._marks = {}
        self._fh = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            self._fh = open(log_path, "a", encoding="utf-8")

    def mark(self, event: str, **extra):
        now = time.perf_counter()
        elapsed = now - self._t0
        entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_s": round(elapsed, 4), "event": event}
        entry.update(extra)
        if self._fh:
            self._fh.write(json.dumps(entry) + "\n")
            self._fh.flush()
        self._marks[event] = now
        return now

    def since(self, event: str) -> float:
        return time.perf_counter() - self._marks.get(event, self._t0)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def save(
        path_prefixes: Iterable[str],
        imgs: torch.Tensor,  # BCHW
        latents: torch.Tensor,
        noises: torch.Tensor,
        imgs_rand: Optional[torch.Tensor] = None,
        png_compress: int = 3,
):
    assert len(path_prefixes) == len(imgs) and len(latents) == len(path_prefixes)
    if imgs_rand is not None:
        assert len(imgs) == len(imgs_rand)
    imgs_arr = make_image(imgs)
    for path_prefix, img, latent, noise in zip(path_prefixes, imgs_arr, latents, noises):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        cv2.imwrite(path_prefix + ".png", img[...,::-1].copy(),
                    [cv2.IMWRITE_PNG_COMPRESSION, png_compress])
        torch.save({"latent": latent.detach().cpu(), "noise": noise.detach().cpu()},
                path_prefix + ".pt")

    if imgs_rand is not None:
        imgs_arr = make_image(imgs_rand)
        for path_prefix, img in zip(path_prefixes, imgs_arr):
            cv2.imwrite(path_prefix + "-rand.png", img[...,::-1].copy(),
                        [cv2.IMWRITE_PNG_COMPRESSION, png_compress])


def _gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def main(args):
    opt_str = ProjectorArguments.to_string(args)
    print(opt_str)
    input_stem = stem(args.input)
    run_tag = compact_run_tag(input_stem, opt_str)

    timing_path = pjoin(args.results_dir, "run_timing_log.jsonl")
    tlog = TimingLog(timing_path)
    tlog.mark("projector_start", input=os.path.basename(args.input), gpu_mem_mb=_gpu_mem_mb())

    if args.rand_seed is not None:
        set_random_seed(args.rand_seed)
    device = th.device()

    # read inputs. TODO imgs_orig has channel 1
    imgs_orig = read_images([args.input], max_size=args.generator_size).to(device)
    imgs = normalize(imgs_orig)  # actually this will be overwritten by the histogram matching result
    tlog.mark("image_loaded", gpu_mem_mb=_gpu_mem_mb())

    # initialize
    with torch.no_grad():
        init = Initializer(args).to(device)
        tlog.mark("encoder_loaded", gpu_mem_mb=_gpu_mem_mb())
        latent_init = init(imgs_orig)
    tlog.mark("latent_initialized", gpu_mem_mb=_gpu_mem_mb())

    # create generator
    generator = create_generator(args, device)
    tlog.mark("generator_loaded", gpu_mem_mb=_gpu_mem_mb())

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
    tlog.mark("histogram_matched", gpu_mem_mb=_gpu_mem_mb())

    degrade = Degrade(args).to(device)

    rgb_levels = generator.get_latent_size(args.coarse_min) // 2 + len(args.wplus_step) - 1
    criterion = JointLoss(
            args, imgs,
            sibling=sibling.detach(), sibling_rgbs=sibling_rgbs[:rgb_levels]).to(device)
    tlog.mark("loss_initialized", gpu_mem_mb=_gpu_mem_mb())

    # save initialization
    save(
        [pjoin(args.results_dir, f"{run_tag}-init")],
        sibling, latent_init, noises_init,
    )

    writer = SummaryWriter(pjoin(args.log_dir, run_tag))
    try:
        # start optimize
        tlog.mark("optimization_start", total_steps=int(np.sum(args.wplus_step)), gpu_mem_mb=_gpu_mem_mb())
        latent, noises = Optimizer.optimize(
            generator, criterion, degrade, imgs, latent_init, noises_init, args,
            writer=writer, timing_log=tlog,
        )
        tlog.mark("optimization_end", gpu_mem_mb=_gpu_mem_mb())
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
    tlog.mark("projector_end", gpu_mem_mb=_gpu_mem_mb())
    total_s = tlog.since("projector_start")
    print(f"=== Total projector time: {total_s:.1f}s ===")
    tlog.close()


def parse_args():
    return ProjectorArguments().parse()

if __name__ == "__main__":
    sys.exit(main(parse_args()))

