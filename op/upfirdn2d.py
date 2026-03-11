import torch
import torch.nn.functional as F


def _to_2tuple(value):
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            return int(value[0]), int(value[1])
        if len(value) == 1:
            return int(value[0]), int(value[0])
    return int(value), int(value)


def _to_4pad(value):
    if isinstance(value, (tuple, list)):
        if len(value) == 4:
            return int(value[0]), int(value[1]), int(value[2]), int(value[3])
        if len(value) == 2:
            return int(value[0]), int(value[1]), int(value[0]), int(value[1])
        if len(value) == 1:
            p = int(value[0])
            return p, p, p, p
    p = int(value)
    return p, p, p, p


def _upfirdn2d_fallback_impl(x, kernel, up=1, down=1, pad=(0, 0)):
    # NCHW expected by StyleGAN2 call sites.
    if x.ndim != 4:
        raise RuntimeError(f"upfirdn2d expected a 4D NCHW tensor, got shape {tuple(x.shape)}")

    up_x, up_y = _to_2tuple(up)
    down_x, down_y = _to_2tuple(down)
    pad_x0, pad_x1, pad_y0, pad_y1 = _to_4pad(pad)

    n, c, in_h, in_w = x.shape

    if up_x > 1 or up_y > 1:
        y = x.new_zeros((n, c, in_h * up_y, in_w * up_x))
        y[:, :, ::up_y, ::up_x] = x
        x = y

    # F.pad does not accept negative values, so crop first.
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

    if pad_x0 or pad_x1 or pad_y0 or pad_y1:
        x = F.pad(x, [pad_x0, pad_x1, pad_y0, pad_y1])

    k = torch.as_tensor(kernel, device=x.device, dtype=x.dtype)
    if k.ndim == 1:
        k = torch.outer(k, k)
    if k.ndim != 2:
        raise RuntimeError(f"kernel must be 1D or 2D, got shape {tuple(k.shape)}")

    # Conv2d is correlation; flip for convolution-equivalent filtering.
    k = k.flip([0, 1]).contiguous().view(1, 1, k.shape[0], k.shape[1])

    x = x.reshape(n * c, 1, x.shape[2], x.shape[3])
    x = F.conv2d(x, k, stride=1, padding=0)
    x = x.reshape(n, c, x.shape[2], x.shape[3])

    if down_x > 1 or down_y > 1:
        x = x[:, :, ::down_y, ::down_x]

    return x


try:
    from models.encoder4editing.models.stylegan2.op.upfirdn2d import upfirdn2d as _e4e_upfirdn2d
except Exception:
    _e4e_upfirdn2d = None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if _e4e_upfirdn2d is not None and getattr(input, "is_cuda", False):
        return _e4e_upfirdn2d(input, kernel, up=up, down=down, pad=pad)
    return _upfirdn2d_fallback_impl(input, kernel, up=up, down=down, pad=pad)
