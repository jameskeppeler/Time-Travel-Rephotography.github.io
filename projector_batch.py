"""Batch projector — loads models once and processes multiple faces sequentially.

Eliminates the ~5-8s conda activation overhead per face that occurs when
run_rephoto_with_facecrop.ps1 invokes projector.py once per crop.

Accepts a JSON manifest file listing faces to process.  Each entry specifies
the input image path and per-face results directory.  All other arguments
(loss weights, learning rate, etc.) are shared across faces.
"""
from argparse import ArgumentParser, Namespace
import json
import os
from os.path import join as pjoin
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

from losses.joint_loss import JointLoss
from projector import (
    TimingLog,
    compact_run_tag,
    create_generator,
    normalize,
    read_images,
    save,
    set_random_seed,
    short_hash,
    _gpu_mem_mb,
)
from tools.initialize import Initializer
from tools.match_skin_histogram import match_skin_histogram
from utils.projector_arguments import ProjectorArguments
from utils import torch_helpers as th
from utils.misc import stem
from utils.optimize import Optimizer
from models.degrade import Degrade


def _check_stop_flag(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _wait_for_pause(pause_path: str, stop_path: str):
    """Block while the pause flag file exists, checking for stop."""
    if not pause_path or not os.path.exists(pause_path):
        return
    print("=== Pause requested ===", flush=True)
    while os.path.exists(pause_path):
        time.sleep(0.25)
        if _check_stop_flag(stop_path):
            break
    print("=== Resume requested ===", flush=True)


def process_single_face(
    input_path: str,
    results_dir: str,
    args: Namespace,
    generator: torch.nn.Module,
    initializer: Initializer,
    device: torch.device,
):
    """Process one face crop using pre-loaded models.

    Emits the same stdout markers as the original per-face projector.py
    so that the GUI progress parser works unchanged.
    """
    opt_str = ProjectorArguments.to_string(args)
    input_stem = stem(input_path)
    run_tag = compact_run_tag(input_stem, opt_str)

    timing_path = pjoin(results_dir, "run_timing_log.jsonl")
    tlog = TimingLog(timing_path)
    tlog.mark("projector_start", input=os.path.basename(input_path), gpu_mem_mb=_gpu_mem_mb())

    # Read input
    imgs_orig = read_images([input_path], max_size=args.generator_size).to(device)
    imgs = normalize(imgs_orig)
    tlog.mark("image_loaded", gpu_mem_mb=_gpu_mem_mb())

    # Encode latent (models already loaded)
    with torch.no_grad():
        latent_init = initializer(imgs_orig)
    tlog.mark("latent_initialized", gpu_mem_mb=_gpu_mem_mb())

    # Init noises
    with torch.no_grad():
        noises_init = generator.make_noise()

    # Histogram matching
    with torch.no_grad():
        sibling, _, sibling_rgbs = generator([latent_init], input_is_latent=True, noise=noises_init)
    mh_dir = pjoin(results_dir, f"mh_{short_hash(input_stem)}")
    imgs = match_skin_histogram(
        imgs, sibling,
        args.spectral_sensitivity,
        pjoin(mh_dir, "input_sibling"),
        pjoin(mh_dir, "skin_mask"),
        matched_hist_fn=pjoin(mh_dir, f"matched_{args.spectral_sensitivity}.png"),
        normalize=normalize,
    ).to(device)
    tlog.mark("histogram_matched", gpu_mem_mb=_gpu_mem_mb())

    degrade = Degrade(args).to(device)

    rgb_levels = generator.get_latent_size(args.coarse_min) // 2 + len(args.wplus_step) - 1
    criterion = JointLoss(
        args, imgs,
        sibling=sibling.detach(), sibling_rgbs=sibling_rgbs[:rgb_levels],
    ).to(device)
    tlog.mark("loss_initialized", gpu_mem_mb=_gpu_mem_mb())

    # Save initialization
    save(
        [pjoin(results_dir, f"{run_tag}-init")],
        sibling, latent_init, noises_init,
    )

    writer = SummaryWriter(pjoin(args.log_dir, run_tag))
    try:
        tlog.mark("optimization_start", total_steps=int(np.sum(args.wplus_step)), gpu_mem_mb=_gpu_mem_mb())
        latent, noises = Optimizer.optimize(
            generator, criterion, degrade, imgs, latent_init, noises_init, args,
            writer=writer, timing_log=tlog,
        )
        tlog.mark("optimization_end", gpu_mem_mb=_gpu_mem_mb())
    finally:
        writer.close()

    # Generate and save output
    with torch.no_grad():
        img_out, _, _ = generator([latent], input_is_latent=True, noise=noises)
        img_out_rand_noise, _, _ = generator([latent], input_is_latent=True)
    save(
        [pjoin(results_dir, run_tag)],
        img_out, latent, noises,
        imgs_rand=img_out_rand_noise,
    )
    tlog.mark("projector_end", gpu_mem_mb=_gpu_mem_mb())
    total_s = tlog.since("projector_start")
    print(f"=== Total projector time: {total_s:.1f}s ===")
    tlog.close()

    # Free per-face tensors
    del criterion, degrade, imgs, imgs_orig, sibling, sibling_rgbs
    del latent, noises, latent_init, noises_init, img_out, img_out_rand_noise
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_parser() -> ArgumentParser:
    """Build argument parser with all projector args but 'input' as optional."""
    parser = ArgumentParser("Batch projector — process multiple faces with shared models")
    parser.add_argument("--manifest", required=True,
                        help="JSON file listing faces: [{\"input\": ..., \"results_dir\": ...}, ...]")
    # Add all the sub-argument groups (stylegan, loss, optimizer, degrade, initializer)
    # but skip the positional 'input' and '--results_dir' since those come from the manifest.
    parser.add_argument("--results_dir", default="results/projector", help="(unused in batch mode, per-face dirs come from manifest)")
    parser.add_argument("--stop_flag", type=str, default="", help="optional file path; if created, batch ends early")
    parser.add_argument("--pause_flag", type=str, default="", help="optional file path; if present, batch pauses")
    parser.add_argument('--rand_seed', type=int, default=None, help="random seed")

    ProjectorArguments.add_stylegan_args(parser)
    ProjectorArguments.add_preprocess_args(parser)

    from tools.initialize import InitializerArguments
    from losses.joint_loss import LossArguments
    from utils.optimize import OptimizerArguments
    from models.degrade import DegradeArguments
    InitializerArguments.add_arguments(parser)
    LossArguments.add_arguments(parser)
    OptimizerArguments.add_arguments(parser)
    DegradeArguments.add_arguments(parser)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    # Set a dummy 'input' so ProjectorArguments.to_string() doesn't error
    if not hasattr(args, "input"):
        args.input = "__batch__"

    # Load manifest
    with open(args.manifest, "r", encoding="utf-8-sig") as f:
        manifest = json.load(f)

    if not manifest:
        print("Empty manifest, nothing to process.")
        return

    print(f"=== Batch projector: {len(manifest)} face(s) ===", flush=True)

    if args.rand_seed is not None:
        set_random_seed(args.rand_seed)
    device = th.device()

    # Load models ONCE
    print("Loading shared models...", flush=True)
    t0 = time.perf_counter()

    with torch.no_grad():
        initializer = Initializer(args).to(device)
    generator = create_generator(args, device)

    model_load_s = time.perf_counter() - t0
    print(f"Models loaded in {model_load_s:.1f}s (GPU: {_gpu_mem_mb():.0f} MB)", flush=True)

    stop_path = getattr(args, "stop_flag", "") or ""
    pause_path = getattr(args, "pause_flag", "") or ""

    for i, entry in enumerate(manifest):
        _wait_for_pause(pause_path, stop_path)
        if _check_stop_flag(stop_path):
            print("Early-stop flag detected. Ending batch early.", flush=True)
            break

        input_path = entry["input"]
        results_dir = entry["results_dir"]
        crop_name = os.path.basename(input_path)

        # Emit the same markers the PS1 wrapper emits so GUI parsing works
        print(f"=== Rephoto step ===", flush=True)
        print(f"Crop: {input_path}", flush=True)
        print(f"Results: {results_dir}", flush=True)
        print("", flush=True)

        os.makedirs(results_dir, exist_ok=True)

        face_start = time.perf_counter()
        try:
            # Override per-face values
            args.input = input_path
            args.results_dir = results_dir

            process_single_face(input_path, results_dir, args, generator, initializer, device)
        except Exception as e:
            print(f"ERROR processing {crop_name}: {e}", flush=True)
            raise

        elapsed = time.perf_counter() - face_start
        print(f"=== Rephoto crop elapsed: {elapsed:.1f}s ===", flush=True)

        # Copy final output to simple name for GUI
        try:
            pngs = [
                f for f in os.listdir(results_dir)
                if f.endswith(".png")
                and not f.endswith("-init.png")
                and not f.endswith("-rand.png")
                and not f.endswith("_g.png")
                and f != "final.png"
            ]
            if pngs:
                pngs.sort(key=lambda p: os.path.getmtime(pjoin(results_dir, p)), reverse=True)
                src = pjoin(results_dir, pngs[0])
                dst = pjoin(results_dir, "final.png")
                shutil.copy2(src, dst)
                print(f"Simple final copy: {dst}", flush=True)
        except Exception as e:
            print(f"Warning: Simple final copy failed (non-fatal): {e}", flush=True)

        if _check_stop_flag(stop_path):
            print("Early-stop flag acknowledged. Finishing batch now.", flush=True)
            break

    # Free shared models
    del initializer, generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=== Batch projector complete ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
