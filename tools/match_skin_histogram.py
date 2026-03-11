from argparse import Namespace
import os
from os.path import join as pjoin
from typing import Optional

import cv2
import torch

from tools import (
    parse_face,
    match_histogram,
)
from utils.torch_helpers import make_image
from utils.misc import stem


def _has_valid_mask(mask_path: str) -> bool:
    if not os.path.isfile(mask_path):
        return False
    mask = cv2.imread(mask_path, 0)
    return mask is not None and mask.size > 0


def match_skin_histogram(
        imgs: torch.Tensor,
        sibling_img: torch.Tensor,
        spectral_sensitivity,
        im_sibling_dir: str,
        mask_dir: str,
        matched_hist_fn: Optional[str] = None,
        normalize=None,  # normalize the range of the tensor
):
    """
    Extract the skin of the input and sibling images. Create a new input image by matching
    its histogram to the sibling.
    """
    # TODO: Currently only allows imgs of batch size 1
    im_sibling_dir = os.path.abspath(im_sibling_dir)
    mask_dir = os.path.abspath(mask_dir)

    # make_image returns RGB; OpenCV expects BGR when writing/reading files.
    img_np = make_image(imgs)[0][..., ::-1]
    sibling_np = make_image(sibling_img)[0][..., ::-1]

    # save img, sibling
    os.makedirs(im_sibling_dir, exist_ok=True)
    im_name, sibling_name = 'input.png', 'sibling.png'
    cv2.imwrite(pjoin(im_sibling_dir, im_name), img_np)
    cv2.imwrite(pjoin(im_sibling_dir, sibling_name), sibling_np)

    # face parsing
    try:
        parse_face.main(
            Namespace(in_dir=im_sibling_dir, out_dir=mask_dir, include_hair=False)
        )
    except Exception as e:
        print(f"WARNING: face parsing failed; skipping histogram match. ({e})")
        return imgs

    src_mask_path = pjoin(mask_dir, im_name)
    ref_mask_path = pjoin(mask_dir, sibling_name)
    if not (_has_valid_mask(src_mask_path) and _has_valid_mask(ref_mask_path)):
        print("WARNING: skin masks were not generated; skipping histogram match.")
        return imgs

    # match_histogram
    mh_args = match_histogram.parse_args(
        args=[
            pjoin(im_sibling_dir, im_name),
            pjoin(im_sibling_dir, sibling_name),
        ],
        namespace=Namespace(
            out=matched_hist_fn if matched_hist_fn else pjoin(im_sibling_dir, "match_histogram.png"),
            src_mask=src_mask_path,
            ref_mask=ref_mask_path,
            spectral_sensitivity=spectral_sensitivity,
        )
    )
    try:
        matched_np = match_histogram.main(mh_args) / 255.0  # [0, 1]
    except Exception as e:
        print(f"WARNING: histogram match failed; using original input image. ({e})")
        return imgs
    matched = torch.FloatTensor(matched_np).permute(2, 0, 1)[None,...]  #BCHW

    if normalize is not None:
        matched = normalize(matched)

    return matched
