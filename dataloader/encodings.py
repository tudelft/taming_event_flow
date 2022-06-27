"""
Adapted from Monash University https://github.com/TimoStoff/events_contrast_maximization
"""

import torch


def events_to_image(xs, ys, ps, sensor_size=(180, 240), accumulate=True):
    """
    Accumulate events into an image.
    :param xs: event x coordinates
    :param ys: event y coordinates
    :param ps: event polarity
    :param sensor_size: sensor size
    :param accumulate: flag indicating whether to accumulate events into the image
    :return img: image containing per-pixel event counts
    """

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size, device=device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=accumulate)

    return img


def events_to_voxel(xs, ys, ts, ps, num_bins, sensor_size=(180, 240)):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    :param xs: event x coordinates
    :param ys: event y coordinates
    :param ts: event timestamps
    :param ps: event polarity
    :param num_bins: number of bins in the voxel grid
    :param sensor_size: sensor size
    :return: voxel grid representation
    """

    assert len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps)

    voxel = []
    ts = ts * (num_bins - 1)
    device = xs.device

    zeros = torch.zeros(ts.size(), device=device)
    for b_idx in range(num_bins):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
        voxel_bin = events_to_image(xs, ys, ps * weights, sensor_size=sensor_size)
        voxel.append(voxel_bin)

    return torch.stack(voxel)


def events_to_channels(xs, ys, ps, sensor_size=(180, 240)):
    """
    Generate a two-channel event image containing per-pixel event counters.
    :param xs: event x coordinates
    :param ys: event y coordinates
    :param ps: event polarity
    :param sensor_size: sensor size
    :return: event image containing per-pixel and per-polarity event counts
    """

    assert len(xs) == len(ys) and len(ys) == len(ps)

    mask_pos = ps.clone()
    mask_neg = ps.clone()
    mask_pos[ps < 0] = 0
    mask_neg[ps > 0] = 0
    mask_pos[ps > 0] = 1
    mask_neg[ps < 0] = -1

    pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
    neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

    return torch.stack([pos_cnt, neg_cnt])
