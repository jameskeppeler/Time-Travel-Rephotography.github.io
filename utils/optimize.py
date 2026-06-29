import math
from argparse import (
    ArgumentParser,
    Namespace,
)
from typing import (
    Dict,
    Iterable,
    Optional,
    Tuple,
)

import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import Resize

#from optim import get_optimizer_class, OPTIMIZER_MAP
from losses.regularize_noise import NoiseRegularizer
from optim import RAdam
from utils.misc import (
    iterable_to_str,
    optional_string,
)

# ── AMP availability ──────────────────────────────────────────────
_AMP_AVAILABLE = hasattr(torch.cuda.amp, "autocast") and torch.cuda.is_available()


class OptimizerArguments:
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument('--coarse_min', type=int, default=32)
        parser.add_argument('--wplus_step', type=int, nargs="+", default=[250, 750], help="#step for optimizing w_plus")
        #parser.add_argument('--lr_rampup', type=float, default=0.05)
        #parser.add_argument('--lr_rampdown', type=float, default=0.25)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--noise_strength', type=float, default=.0)
        parser.add_argument('--noise_ramp', type=float, default=0.75)
        #parser.add_argument('--optimize_noise', action="store_true")
        parser.add_argument('--camera_lr', type=float, default=0.01)

        parser.add_argument("--log_dir", default="log/projector", help="tensorboard log directory")
        parser.add_argument("--log_freq", type=int, default=10, help="log frequency")
        parser.add_argument("--log_visual_freq", type=int, default=50, help="log frequency")
        parser.add_argument("--progress_freq", type=int, default=10, help="loss update print frequency (iterations)")

        # ── New optimization flags ────────────────────────────────
        parser.add_argument("--use_amp", action="store_true", default=False,
                            help="Enable mixed-precision (fp16) training for ~15-30%% speedup")
        parser.add_argument("--early_stop_patience", type=int, default=0,
                            help="Stop if loss doesn't improve for N iterations (0=disabled)")
        parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                            help="Minimum relative improvement to count as progress")
        parser.add_argument("--lr_decay", type=float, default=0.0,
                            help="Cosine LR decay factor: final_lr = lr * lr_decay. 0=disabled")

    @staticmethod
    def to_string(args: Namespace) -> str:
        return (
            f"lr{args.lr}_{args.camera_lr}-c{args.coarse_min}"
            + f"-wp({iterable_to_str(args.wplus_step)})"
            + optional_string(args.noise_strength, f"-n{args.noise_strength}")
        )


class LatentNoiser(nn.Module):
    def __init__(
            self, generator: torch.nn,
            noise_ramp: float = 0.75, noise_strength: float = 0.05,
            n_mean_latent: int = 10000
    ):
        super().__init__()

        self.noise_ramp = noise_ramp
        self.noise_strength = noise_strength

        with torch.no_grad():
            # TODO: get 512 from generator
            noise_sample = torch.randn(n_mean_latent, 512, device=generator.device)
            latent_out = generator.style(noise_sample)

            latent_mean = latent_out.mean(0)
            self.latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    def forward(self, latent: torch.Tensor, t: float) -> torch.Tensor:
        strength = self.latent_std * self.noise_strength * max(0, 1 - t / self.noise_ramp) ** 2
        noise = torch.randn_like(latent) * strength
        return latent + noise


class _EarlyStopTracker:
    """Tracks loss plateau for early stopping."""
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.enabled = patience > 0

    def should_stop(self, loss_val: float) -> bool:
        if not self.enabled:
            return False
        if not math.isfinite(loss_val):
            # Diverged — stop immediately rather than waiting for patience
            return True
        rel_improvement = (self.best_loss - loss_val) / max(abs(self.best_loss), 1e-12)
        if rel_improvement > self.min_delta:
            self.best_loss = loss_val
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience


class Optimizer:
    @staticmethod
    def _to_float_scalar(value) -> float:
        if torch.is_tensor(value):
            return float(value.detach().item())
        return float(value)

    @classmethod
    def optimize(
            cls,
            generator: torch.nn,
            criterion: torch.nn,
            degrade: torch.nn,
            target: torch.Tensor,  # only used in writer since it's mostly baked in criterion
            latent_init: torch.Tensor,
            noise_init: torch.Tensor,
            args: Namespace,
            writer: Optional[SummaryWriter] = None,
            timing_log=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # do not optimize generator
        generator = generator.eval()
        target = target.detach()
        # prepare parameters
        noises = []
        for n in noise_init:
            noise = n.detach().clone()
            noise.requires_grad = True
            noises.append(noise)


        def create_parameters(latent_coarse):
            parameters = [
                {'params': [latent_coarse], 'lr': args.lr},
                {'params': noises, 'lr': args.lr},
                {'params': degrade.parameters(), 'lr': args.camera_lr},
            ]
            return parameters


        device = target.device

        # start optimize
        total_steps = int(np.sum(args.wplus_step))
        noiser = None
        if args.noise_strength > 0:
            noiser = LatentNoiser(
                generator,
                noise_ramp=args.noise_ramp,
                noise_strength=args.noise_strength,
            ).to(device)
        latent = latent_init.detach().clone()
        milestone_hits = set()
        stop_flag_path = str(getattr(args, "stop_flag", "") or "").strip()
        pause_flag_path = str(getattr(args, "pause_flag", "") or "").strip()
        stop_requested = False
        pause_announced = False

        # ── Mixed precision setup ─────────────────────────────────
        use_amp = getattr(args, "use_amp", False) and _AMP_AVAILABLE
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
        amp_dtype = torch.float16

        # ── Early stopping setup ──────────────────────────────────
        early_stop_patience = getattr(args, "early_stop_patience", 0)
        early_stop_delta = getattr(args, "early_stop_min_delta", 1e-4)
        early_stopper = _EarlyStopTracker(early_stop_patience, early_stop_delta)

        # ── LR decay setup ────────────────────────────────────────
        lr_decay = getattr(args, "lr_decay", 0.0)
        use_lr_schedule = lr_decay > 0.0 and lr_decay < 1.0

        # ── Reduced logging: bump visual freq to save GPU→CPU copies ──
        log_freq = max(1, int(getattr(args, "log_freq", 10)))
        log_visual_freq = max(50, int(getattr(args, "log_visual_freq", 50)))

        def should_stop_early() -> bool:
            return bool(stop_flag_path) and os.path.exists(stop_flag_path)

        def should_pause() -> bool:
            return bool(pause_flag_path) and os.path.exists(pause_flag_path)

        def wait_if_paused() -> bool:
            nonlocal pause_announced, stop_requested
            while should_pause():
                if should_stop_early():
                    print("=== Early stop requested ===")
                    stop_requested = True
                    return False
                if not pause_announced:
                    print("=== Pause requested ===")
                    pause_announced = True
                time.sleep(0.2)
            if pause_announced:
                print("=== Resume requested ===")
                pause_announced = False
            return True

        level_t0 = time.perf_counter()

        for coarse_level, steps in enumerate(args.wplus_step):
            if not wait_if_paused():
                break
            if should_stop_early():
                print("=== Early stop requested ===")
                stop_requested = True
                break

            if criterion.weights["contextual"] > 0:
                with torch.no_grad():
                    # synthesize new sibling image using the current optimization results
                    # FIXME: update rgbs sibling
                    sibling, _, _ = generator([latent], input_is_latent=True, randomize_noise=True)
                    criterion.update_sibling(sibling)

            coarse_size = (2 ** coarse_level) * args.coarse_min
            latent_coarse, latent_fine = cls.split_latent(
                    latent, generator.get_latent_size(coarse_size))
            parameters = create_parameters(latent_coarse)
            optimizer = RAdam(parameters)
            completed_before_level = int(np.sum(args.wplus_step[:coarse_level]))
            progress_freq = max(1, int(getattr(args, "progress_freq", 10)))

            # ── LR scheduler for this coarse level ────────────────
            scheduler = None
            if use_lr_schedule:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=steps, eta_min=args.lr * lr_decay)

            # Reset early-stop tracker per coarse level
            early_stopper.best_loss = float("inf")
            early_stopper.wait = 0

            level_t0 = time.perf_counter()
            print(f"Optimizing {coarse_size}x{coarse_size}" +
                  (f" [AMP fp16]" if use_amp else "") +
                  (f" [early-stop patience={early_stop_patience}]" if early_stopper.enabled else ""))
            if timing_log:
                timing_log.mark(f"coarse_level_{coarse_level}_start",
                                resolution=f"{coarse_size}x{coarse_size}",
                                steps=steps, use_amp=use_amp)

            for si in range(steps):
                # Print iteration counter every 50 steps to reduce logging overhead
                if si % 50 == 0 or si == steps - 1:
                    print(f"{si+1}/{steps}", flush=True)
                if not wait_if_paused():
                    break
                if should_stop_early():
                    print("=== Early stop requested ===")
                    stop_requested = True
                    break

                latent = torch.cat((latent_coarse, latent_fine), dim=1)
                niters = si + completed_before_level

                # Emit milestone markers for shorter-run timing reuse
                completed_iters = niters + 1
                if completed_iters >= 1000:
                    current_milestone = (completed_iters // 1000) * 1000
                    if current_milestone not in milestone_hits and current_milestone < total_steps:
                        print(f"=== Rephoto milestone === {current_milestone}")
                        milestone_hits.add(current_milestone)
                        if timing_log:
                            timing_log.mark(f"milestone_{current_milestone}",
                                            iteration=completed_iters)

                if noiser is None:
                    latent_noisy = latent
                else:
                    latent_noisy = noiser(latent, niters / max(1, total_steps))

                # ── Forward + loss (with optional AMP) ────────────
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        img_gen, _, rgbs = generator([latent_noisy], input_is_latent=True, noise=noises)
                        loss, losses = criterion(img_gen, degrade=degrade, noises=noises, rgbs=rgbs)

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    img_gen, _, rgbs = generator([latent_noisy], input_is_latent=True, noise=noises)
                    # TODO: use coarse_size instead of args.coarse_size for rgb_level
                    loss, losses = criterion(img_gen, degrade=degrade, noises=noises, rgbs=rgbs)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                NoiseRegularizer.normalize(noises)

                # ── Early stopping check (sample every progress_freq to avoid GPU sync) ──
                if early_stopper.enabled and si % progress_freq == 0:
                    loss_val = cls._to_float_scalar(loss)
                    if early_stopper.should_stop(loss_val):
                        print(f"=== Early stop: loss plateaued at {loss_val:.4e} for {early_stop_patience} iters ===")
                        if timing_log:
                            timing_log.mark("early_stop_triggered",
                                            iteration=niters, loss=loss_val,
                                            coarse_level=coarse_level)
                        break

                # Print losses periodically (avoids frequent GPU->CPU sync for .item()).
                if si % progress_freq == 0 or si == (steps - 1):
                    desc_parts = []
                    for k, v in losses.items():
                        try:
                            scalar_v = cls._to_float_scalar(v)
                        except (TypeError, ValueError):
                            continue
                        desc_parts.append(f"{k}: {scalar_v: .3e}")
                    if desc_parts:
                        print("; ".join(desc_parts), flush=True)

                if writer is not None and niters % log_freq == 0:
                    cls.log_losses(writer, niters, loss, losses, criterion.weights)
                    cls.log_parameters(writer, niters, degrade.named_parameters())
                if writer is not None and niters % log_visual_freq == 0:
                    with torch.no_grad():
                        degraded_vis = degrade(img_gen)
                    cls.log_visuals(writer, niters, img_gen, target, degraded=degraded_vis, rgbs=rgbs)

            level_elapsed = time.perf_counter() - level_t0
            iters_per_sec = (si + 1) / max(level_elapsed, 0.001)
            print(f"Level {coarse_level} done: {si+1} iters in {level_elapsed:.1f}s ({iters_per_sec:.1f} it/s)")
            if timing_log:
                timing_log.mark(f"coarse_level_{coarse_level}_end",
                                iterations=si + 1, elapsed_s=round(level_elapsed, 2),
                                iters_per_sec=round(iters_per_sec, 1))

            latent = torch.cat((latent_coarse, latent_fine), dim=1).detach()
            if stop_requested:
                break

        return latent, noises

    @staticmethod
    def split_latent(latent: torch.Tensor, coarse_latent_size: int):
        latent_coarse = latent[:, :coarse_latent_size]
        latent_coarse.requires_grad = True
        latent_fine = latent[:, coarse_latent_size:]
        latent_fine.requires_grad = False
        return latent_coarse, latent_fine

    @staticmethod
    def log_losses(
            writer: SummaryWriter,
            niters: int,
            loss_total: torch.Tensor,
            losses: Dict[str, torch.Tensor],
            weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        writer.add_scalar("loss", Optimizer._to_float_scalar(loss_total), niters)

        for name, loss in losses.items():
            try:
                loss_scalar = Optimizer._to_float_scalar(loss)
            except (TypeError, ValueError):
                continue
            writer.add_scalar(name, loss_scalar, niters)
            if weights is not None and name in weights:
                weight_scalar = Optimizer._to_float_scalar(weights[name])
                writer.add_scalar(f"weighted_{name}", weight_scalar * loss_scalar, niters)

    @staticmethod
    def log_parameters(
            writer: SummaryWriter,
            niters: int,
            named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    ):
        for name, para in named_parameters:
            writer.add_scalar(name, para.item(), niters)

    @classmethod
    def log_visuals(
            cls,
            writer: SummaryWriter,
            niters: int,
            img: torch.Tensor,
            target: torch.Tensor,
            degraded=None,
            rgbs=None,
    ):
        if target.shape[-1] != img.shape[-1]:
            visual = make_grid(img, nrow=1, normalize=True, value_range=(-1, 1))
            writer.add_image("pred", visual, niters)

        def resize(img):
            return F.interpolate(img, size=target.shape[2:], mode="area")

        vis = resize(img)
        if degraded is not None:
            vis = torch.cat((resize(degraded), vis), dim=-1)
        visual = make_grid(torch.cat((target.repeat(1, vis.shape[1] // target.shape[1], 1, 1), vis), dim=-1), nrow=1, normalize=True, value_range=(-1, 1))
        writer.add_image("gnd[-degraded]-pred", visual, niters)

        # log to rgbs
        if rgbs is not None:
            cls.log_torgbs(writer, niters, rgbs)

    @staticmethod
    def log_torgbs(writer: SummaryWriter, niters: int, rgbs: Iterable[torch.Tensor], prefix: str = ""):
        for ri, rgb in enumerate(rgbs):
            scale = 2 ** (-(len(rgbs) - ri))
            visual = make_grid(torch.cat((rgb, rgb / scale), dim=-1), nrow=1, normalize=True, value_range=(-1, 1))
            writer.add_image(f"{prefix}to_rgb_{2 ** (ri + 2)}", visual, niters)
