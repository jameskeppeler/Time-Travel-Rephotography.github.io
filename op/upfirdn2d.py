import torch
import torch.nn.functional as F


def _parse_scaling(v):
    if isinstance(v, int):
        return int(v), int(v)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return int(v[0]), int(v[1])
    raise ValueError(f"up/down must be int or (x, y); got {v!r}")


def _parse_padding(pad):
    # Accept: int, (pad0, pad1), or (x0, x1, y0, y1)
    if isinstance(pad, int):
        return pad, pad, pad, pad
    if isinstance(pad, (tuple, list)):
        if len(pad) == 2:
            x0, x1 = int(pad[0]), int(pad[1])
            return x0, x1, x0, x1
        if len(pad) == 4:
            x0, x1, y0, y1 = map(int, pad)
            return x0, x1, y0, y1
    raise ValueError(f"pad must be int, (pad0,pad1) or (x0,x1,y0,y1); got {pad!r}")


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """
    Pure-PyTorch fallback for StyleGAN2 upfirdn2d.
    input:  (N, C, H, W)
    kernel: (kh, kw) float tensor
    up/down: int or (x, y)
    pad: int or (pad0, pad1) or (x0, x1, y0, y1)
    """
    up_x, up_y = _parse_scaling(up)
    down_x, down_y = _parse_scaling(down)
    pad_x0, pad_x1, pad_y0, pad_y1 = _parse_padding(pad)

    if not torch.is_tensor(kernel):
        kernel = torch.tensor(kernel, dtype=input.dtype, device=input.device)
    else:
        kernel = kernel.to(dtype=input.dtype, device=input.device)

    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2D (kh,kw); got shape {tuple(kernel.shape)}")

    n, c, in_h, in_w = input.shape

    # Work in (N*C, 1, H, W) so the same kernel is applied independently per channel.
    x = input.reshape(n * c, 1, in_h, in_w)

    # Upsample by inserting zeros.
    if up_x > 1 or up_y > 1:
        x = x.view(n * c, 1, in_h, 1, in_w, 1)
        x = F.pad(x, (0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1))
        x = x.view(n * c, 1, in_h * up_y, in_w * up_x)

    # Pad (positive) then crop (negative).
    x = F.pad(
        x,
        (max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)),
    )

    if pad_x0 < 0 or pad_x1 < 0 or pad_y0 < 0 or pad_y1 < 0:
        y0 = max(-pad_y0, 0)
        y1 = x.shape[2] - max(-pad_y1, 0)
        x0 = max(-pad_x0, 0)
        x1 = x.shape[3] - max(-pad_x1, 0)
        x = x[:, :, y0:y1, x0:x1]

    # Convolution with flipped kernel (as in the CUDA op).
    kh, kw = kernel.shape
    w = torch.flip(kernel, dims=[0, 1]).view(1, 1, kh, kw)
    x = F.conv2d(x, w)

    # Downsample.
    x = x[:, :, ::down_y, ::down_x]

    out_h, out_w = x.shape[2], x.shape[3]
    return x.view(n, c, out_h, out_w)

# === SAFE_UPFIRDN2D_OVERRIDE ===
# Windows-safe reference implementation (no custom CUDA extension, no fragile view()-based upsample).
import torch
import torch.nn.functional as F

def _to_2tuple(v):
    if isinstance(v, (tuple, list)):
        if len(v) == 2:
            return int(v[0]), int(v[1])
        if len(v) == 1:
            return int(v[0]), int(v[0])
    return int(v), int(v)

def _to_4pad(p):
    if isinstance(p, (tuple, list)):
        if len(p) == 4:
            return int(p[0]), int(p[1]), int(p[2]), int(p[3])
        if len(p) == 2:
            return int(p[0]), int(p[1]), int(p[0]), int(p[1])
        if len(p) == 1:
            return int(p[0]), int(p[0]), int(p[0]), int(p[0])
    p = int(p)
    return p, p, p, p

def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    # Expect NCHW
    if x.ndim != 4:
        raise RuntimeError(f"upfirdn2d expected a 4D NCHW tensor, got shape {tuple(x.shape)}")

    up_x, up_y = _to_2tuple(up)
    down_x, down_y = _to_2tuple(down)
    pad_x0, pad_x1, pad_y0, pad_y1 = _to_4pad(pad)

    n, c, in_h, in_w = x.shape

    # Upsample by zero-insertion
    if up_x > 1 or up_y > 1:
        y = x.new_zeros((n, c, in_h * up_y, in_w * up_x))
        y[:, :, ::up_y, ::up_x] = x
        x = y

    # Handle negative padding as cropping first (F.pad does not support negative pad)
    if pad_x0 < 0:
        x = x[:, :, :, (-pad_x0):]
        pad_x0 = 0
    if pad_x1 < 0:
        x = x[:, :, :, : (x.shape[3] + pad_x1)]
        pad_x1 = 0
    if pad_y0 < 0:
        x = x[:, :, (-pad_y0):, :]
        pad_y0 = 0
    if pad_y1 < 0:
        x = x[:, :, : (x.shape[2] + pad_y1), :]
        pad_y1 = 0

    # Positive padding
    if pad_x0 or pad_x1 or pad_y0 or pad_y1:
        x = F.pad(x, [pad_x0, pad_x1, pad_y0, pad_y1])

    # Prepare 2D filter
    k = torch.as_tensor(kernel, device=x.device, dtype=x.dtype)
    if k.ndim == 1:
        k = torch.outer(k, k)
    if k.ndim != 2:
        raise RuntimeError(f"kernel must be 1D or 2D, got shape {tuple(k.shape)}")

    # Conv2d is correlation; flip to emulate convolution-style filtering
    k = k.flip([0, 1]).contiguous().view(1, 1, k.shape[0], k.shape[1])

    # Filter per-channel by collapsing N and C
    x = x.reshape(n * c, 1, x.shape[2], x.shape[3])
    x = F.conv2d(x, k, stride=1, padding=0)
    x = x.reshape(n, c, x.shape[2], x.shape[3])

    # Downsample
    if down_x > 1 or down_y > 1:
        x = x[:, :, ::down_y, ::down_x]

    return x
# === END SAFE_UPFIRDN2D_OVERRIDE ===


# === E4E_UPFIRDN2D_DELEGATE ===
# Prefer the compiled StyleGAN2 op from encoder4editing if it imports successfully.
try:
    from models.encoder4editing.models.stylegan2.op.upfirdn2d import upfirdn2d as _e4e_upfirdn2d
except Exception:
    _e4e_upfirdn2d = None

_upfirdn2d_fallback = upfirdn2d

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if _e4e_upfirdn2d is not None and getattr(input, 'is_cuda', False):
        return _e4e_upfirdn2d(input, kernel, up=up, down=down, pad=pad)
    return _upfirdn2d_fallback(input, kernel, up=up, down=down, pad=pad)
# === END E4E_UPFIRDN2D_DELEGATE ===

