import os
import sys

import torch

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from utils.iwe import event_propagation, get_event_flow, purge_unfeasible, get_interpolation, interpolate


class BaseValidation(torch.nn.Module):
    """
    Base class for validation metrics.
    """

    def __init__(self, config, device):
        super(BaseValidation, self).__init__()
        self.res = config["loader"]["resolution"]
        self.device = device
        self.config = config

        self._passes = 0
        self._event_ts = None
        self._event_loc = None
        self._event_pol_mask = None

        self._flow_maps_x = None
        self._flow_maps_y = None
        self._event_mask = None

        # warping indices (for forward-propagated flow)
        my, mx = torch.meshgrid(torch.arange(self.res[0]), torch.arange(self.res[1]))
        indices = torch.stack([my, mx], dim=0).unsqueeze(0)
        self.indices = indices.float().view(1, 2, -1).permute(0, 2, 1).to(device)
        self.indices_map = self.indices.clone().permute(0, 2, 1).view(1, 2, self.res[0], self.res[1])
        self.indices_mask = torch.ones((1, self.res[0] * self.res[1], 1)).to(device)

    @property
    def num_passes(self):
        return self._passes

    def forward_prop_flow(self, i, tref, flow_maps_x, flow_maps_y):
        """
        Forward propagation of the estimated optical flow using bilinear interpolation.
        :param i: time at which the flow map to be warped is defined
        :param tref: reference time for the forward propagation
        :return warped_flow_x: [[batch_size x 1 x H x W]] warped, horizontal optical flow map
        :return warped_flow_y: [batch_size x 1 x H x W] warped, vertical optical flow map
        """

        # sample per-pixel optical flow
        indices_mask = self.indices_mask.clone()
        indices_flow = get_event_flow(flow_maps_x[:, i, ...], flow_maps_y[:, i, ...], self.indices)

        # optical flow (forward) propagation
        warped_indices = event_propagation(i, self.indices, indices_flow, tref)
        warped_indices, indices_mask = purge_unfeasible(warped_indices, indices_mask, self.res)

        # (bilinearly) interpolate forward
        indices_mask = torch.cat([indices_mask for _ in range(4)], dim=1)
        indices_flow = torch.cat([indices_flow for _ in range(4)], dim=1)
        interp_warped_indices, interp_weights = get_interpolation(warped_indices, self.res)
        warped_weights = interpolate(interp_warped_indices, interp_weights, self.res, polarity_mask=indices_mask)
        warped_flow_y = interpolate(
            interp_warped_indices, interp_weights * indices_flow[..., 0:1], self.res, polarity_mask=indices_mask
        )
        warped_flow_x = interpolate(
            interp_warped_indices, interp_weights * indices_flow[..., 1:2], self.res, polarity_mask=indices_mask
        )
        warped_flow_y /= warped_weights + 1e-9
        warped_flow_x /= warped_weights + 1e-9

        return warped_flow_x, warped_flow_y

    def update_base(self, flow_list, event_list, pol_mask, event_mask):
        """
        Initialize/Update container lists of events and flow maps for forward warping.
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
        :param pol_mask: [batch_size x N x 2] polarity mask (pos, neg)
        :param event_mask: [batch_size x 1 x H x W] event mask
        """

        # update event timestamps
        event_list[:, :, 0:1] += self._passes  # only nonzero second time
        event_ts = event_list[:, :, 0:1].clone()
        if self.config["loss"]["round_ts"]:
            event_ts[...] = event_ts.min() + 0.5

        if self._event_ts is None:
            self._event_ts = event_ts
            self._event_loc = event_list[:, :, 1:3].clone()
            self._event_pol_mask = pol_mask.clone()
        else:
            self._event_ts = torch.cat([self._event_ts, event_ts], dim=1)
            self._event_loc = torch.cat([self._event_loc, event_list[:, :, 1:3].clone()], dim=1)
            self._event_pol_mask = torch.cat([self._event_pol_mask, pol_mask.clone()], dim=1)

        # update optical flow maps
        flow = flow_list[-1]  # only highest resolution flow
        if self._flow_maps_x is None:
            self._flow_maps_x = flow[:, 0:1, :, :]
            self._flow_maps_y = flow[:, 1:2, :, :]
        else:
            self._flow_maps_x = torch.cat([self._flow_maps_x, flow[:, 0:1, :, :]], dim=1)
            self._flow_maps_y = torch.cat([self._flow_maps_y, flow[:, 1:2, :, :]], dim=1)

        # update internal smoothing mask
        if self._event_mask is None:
            self._event_mask = event_mask
        else:
            self._event_mask = torch.cat([self._event_mask, event_mask], dim=1)

    def reset_base(self):
        """
        Reset lists.
        """

        self._passes = 0
        self._event_ts = None
        self._event_loc = None
        self._event_pol_mask = None

        self._flow_maps_x = None
        self._flow_maps_y = None
        self._event_mask = None

    def window_events_base(self, round_idx=False):
        """
        :param round_idx: if True, round the event coordinates to the nearest integer.
        :return: image-like representation of all the events in the validation time/event window.
        """

        pol_mask_list = self._event_pol_mask
        if not round_idx:
            pol_mask_list = torch.cat([pol_mask_list for i in range(4)], dim=1)

        fw_idx, fw_weights = get_interpolation(self._event_loc, self.res, round_idx=round_idx)
        events_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask_list[:, :, 0:1])
        events_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask_list[:, :, 1:2])

        return torch.cat([events_pos, events_neg], dim=1)

    def window_flow_base(self, flow_maps_x, flow_maps_y, mask=False):
        """
        :param flow_maps_x: [batch_size x num_passes x H x W] horizontal flow maps to be averaged
        :param flow_maps_y: [batch_size x num_passes x H x W] vertical flow maps to be averaged
        :return avg_flow: image-like representation of the per-pixel average flow in the validation time/event window.
        """

        flow_x = flow_maps_x[:, 0:1, :, :]
        flow_y = flow_maps_y[:, 0:1, :, :]
        avg_flow = torch.cat([flow_x, flow_y], dim=1)
        flow_mask = (flow_x != 0.0) + (flow_y != 0.0)
        cnt = flow_mask.float()

        for i in range(1, flow_maps_x.shape[1]):
            flow_x = flow_maps_x[:, i : i + 1, :, :]
            flow_y = flow_maps_y[:, i : i + 1, :, :]
            avg_flow += torch.cat([flow_x, flow_y], dim=1)
            flow_mask = (flow_x != 0.0) + (flow_y != 0.0)
            cnt += flow_mask.float()

        if mask:
            mask = torch.sum(self._event_mask, dim=1, keepdim=True) > 0.0
            avg_flow *= mask.float()

        return avg_flow / (cnt + 1e-9)

    def window_iwe_base(self, round_idx=False):
        """
        Assumption: events have NOT been previously warped in a forward fashion in the update() method.
        :param round_idx: if True, round the event coordinates to the nearest integer.
        :return: image-like representation of the IWE of all the events in the validation time/event window.
        """

        pol_mask_list = self._event_pol_mask
        if not round_idx:
            pol_mask_list = torch.cat([pol_mask_list for i in range(4)], dim=1)

        fw_events = event_propagation(self._event_ts, self._event_loc, self._event_flow, self._passes)
        fw_idx, fw_weights = get_interpolation(fw_events, self.res, round_idx=round_idx)
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask_list[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask_list[:, :, 1:2])

        return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)

    def compute_fwl(self, fw_events, zero_events, fw_pol_mask, zero_pol_mask):
        """
        The Flow Warp Loss (FWL) metric is the ratio of the variance of the image of warped events
        and that of the image of (non-warped) events; hence, the higher the value of this metric,
        the better the optical flow estimate.
        See 'Reducing the Sim-to-Real Gap for Event Cameras',
        Stoffregen et al., ECCV 2020.
        """

        # interpolate forward
        fw_idx, fw_weights = get_interpolation(fw_events, self.res, round_idx=True)

        # image of (forward) warped averaged timestamps
        fw_iwe_pos = interpolate(fw_idx, fw_weights, self.res, polarity_mask=fw_pol_mask[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx, fw_weights, self.res, polarity_mask=fw_pol_mask[:, :, 1:2])
        fw_iwe = fw_iwe_pos + fw_iwe_neg

        # image of non-warped averaged timestamps
        zero_idx, zero_weights = get_interpolation(zero_events, self.res, round_idx=True)
        zero_iwe_pos = interpolate(zero_idx, zero_weights, self.res, polarity_mask=zero_pol_mask[:, :, 0:1])
        zero_iwe_neg = interpolate(zero_idx, zero_weights, self.res, polarity_mask=zero_pol_mask[:, :, 1:2])
        zero_iwe = zero_iwe_pos + zero_iwe_neg

        return fw_iwe.var() / zero_iwe.var()

    def compute_rsat(self, fw_events, zero_events, fw_pol_mask, zero_pol_mask, ts_list):
        """
        The Ratio of the Squared Averaged Timestamps (RSAT) metric is the ratio of the squared sum of the per-pixel and
        per-polarity average timestamp of the image of warped events and that of the image of (non-warped) events; hence,
        the lower the value of this metric, the better the optical flow estimate.
        See 'Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks',
        Hagenaars and Paredes-Valles et al., NeurIPS 2021.
        """

        # interpolate forward
        fw_idx, fw_weights = get_interpolation(fw_events, self.res, round_idx=True)

        # image of (forward) warped averaged timestamps
        fw_iwe_pos = interpolate(fw_idx, fw_weights, self.res, polarity_mask=fw_pol_mask[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx, fw_weights, self.res, polarity_mask=fw_pol_mask[:, :, 1:2])
        fw_iwe_pos_ts = interpolate(fw_idx, fw_weights * ts_list, self.res, polarity_mask=fw_pol_mask[:, :, 0:1])
        fw_iwe_neg_ts = interpolate(fw_idx, fw_weights * ts_list, self.res, polarity_mask=fw_pol_mask[:, :, 1:2])
        fw_iwe_pos_ts /= fw_iwe_pos + 1e-9
        fw_iwe_neg_ts /= fw_iwe_neg + 1e-9
        fw_iwe_pos_ts = fw_iwe_pos_ts / self._passes
        fw_iwe_neg_ts = fw_iwe_neg_ts / self._passes

        # image of non-warped averaged timestamps
        zero_idx, zero_weights = get_interpolation(zero_events, self.res, round_idx=True)
        zero_iwe_pos = interpolate(zero_idx, zero_weights, self.res, polarity_mask=zero_pol_mask[:, :, 0:1])
        zero_iwe_neg = interpolate(zero_idx, zero_weights, self.res, polarity_mask=zero_pol_mask[:, :, 1:2])
        zero_iwe_pos_ts = interpolate(
            zero_idx, zero_weights * ts_list, self.res, polarity_mask=zero_pol_mask[:, :, 0:1]
        )
        zero_iwe_neg_ts = interpolate(
            zero_idx, zero_weights * ts_list, self.res, polarity_mask=zero_pol_mask[:, :, 1:2]
        )
        zero_iwe_pos_ts /= zero_iwe_pos + 1e-9
        zero_iwe_neg_ts /= zero_iwe_neg + 1e-9
        zero_iwe_pos_ts = zero_iwe_pos_ts / self._passes
        zero_iwe_neg_ts = zero_iwe_neg_ts / self._passes

        # (scaled) sum of the squares of the per-pixel and per-polarity average timestamps
        fw_iwe_pos_ts = fw_iwe_pos_ts.view(fw_iwe_pos_ts.shape[0], -1)
        fw_iwe_neg_ts = fw_iwe_neg_ts.view(fw_iwe_neg_ts.shape[0], -1)
        fw_iwe_pos_ts = torch.sum(fw_iwe_pos_ts**2, dim=1)
        fw_iwe_neg_ts = torch.sum(fw_iwe_neg_ts**2, dim=1)
        fw_ts_sum = fw_iwe_pos_ts + fw_iwe_neg_ts

        fw_nonzero_px = fw_iwe_pos + fw_iwe_neg
        fw_nonzero_px[fw_nonzero_px > 0] = 1
        fw_nonzero_px = fw_nonzero_px.view(fw_nonzero_px.shape[0], -1)
        fw_ts_sum /= torch.sum(fw_nonzero_px, dim=1)

        zero_iwe_pos_ts = zero_iwe_pos_ts.view(zero_iwe_pos_ts.shape[0], -1)
        zero_iwe_neg_ts = zero_iwe_neg_ts.view(zero_iwe_neg_ts.shape[0], -1)
        zero_iwe_pos_ts = torch.sum(zero_iwe_pos_ts**2, dim=1)
        zero_iwe_neg_ts = torch.sum(zero_iwe_neg_ts**2, dim=1)
        zero_ts_sum = zero_iwe_pos_ts + zero_iwe_neg_ts

        zero_nonzero_px = zero_iwe_pos + zero_iwe_neg
        zero_nonzero_px[zero_nonzero_px > 0] = 1
        zero_nonzero_px = zero_nonzero_px.view(zero_nonzero_px.shape[0], -1)
        zero_ts_sum /= torch.sum(zero_nonzero_px, dim=1)

        return fw_ts_sum / zero_ts_sum

    def compute_aee(self, pred, gt, mask=None):
        """
        Average endpoint error (i.e., Euclidean distance).
        """

        # compute AEE
        batch_size = pred.shape[0]
        error = (pred - gt).pow(2).sum(1).sqrt()

        # AEE not computed in pixels without valid ground truth
        gtflow_mask = (gt[:, 0, :, :] == 0.0) * (gt[:, 1, :, :] == 0.0)
        gtflow_mask = ~gtflow_mask

        # AEE not computed in pixels without input events (MVSEC)
        if mask is not None:
            mask = torch.sum(mask, axis=1)
            mask = mask > 0

            if "res_aee" in self.config["metrics"].keys():
                yoff = (self.res[0] - self.config["metrics"]["res_aee"][0]) // 2
                xoff = (self.res[1] - self.config["metrics"]["res_aee"][1]) // 2
                mask = mask[:, yoff:-yoff, xoff:-xoff].contiguous()
                error = error[:, yoff:-yoff, xoff:-xoff].contiguous()
                gtflow_mask = gtflow_mask[:, yoff:-yoff, xoff:-xoff].contiguous()

            if "vertical_crop_aee" in self.config["metrics"].keys():
                mask = mask[:, : self.config["metrics"]["vertical_crop_aee"], :]
                error = error[:, : self.config["metrics"]["vertical_crop_aee"], :]
                gtflow_mask = gtflow_mask[:, : self.config["metrics"]["vertical_crop_aee"], :]

            gtflow_mask = gtflow_mask * mask

        # compute AEE
        error = error.view(batch_size, -1)
        gtflow_mask = gtflow_mask.view(batch_size, -1)
        error = error[gtflow_mask]
        aee = torch.mean(error, dim=0)

        return aee


class Linear(BaseValidation):
    """
    Linear event warping validation class.
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        self._event_flow = None

    def update(self, flow_list, event_list, pol_mask, event_mask):
        """
        Initialize/Update container lists of events and flow maps for forward warping.
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
        :param pol_mask: [batch_size x N x 2] polarity mask (pos, neg)
        :param event_mask: [batch_size x 1 x H x W] event mask
        """

        # update base lists (event data, flow maps, event masks)
        self.update_base(flow_list, event_list, pol_mask, event_mask)

        # get flow for every event in the list
        event_flow = get_event_flow(
            self._flow_maps_x[:, -1, ...],
            self._flow_maps_y[:, -1, ...],
            event_list[:, :, 1:3],
        )

        if self._event_flow is None:
            self._event_flow = event_flow
        else:
            self._event_flow = torch.cat([self._event_flow, event_flow], dim=1)

        # update timestamp index
        self._passes += 1

    def reset(self):
        """
        Reset lists.
        """

        self.reset_base()
        self._event_flow = None

    def window_events(self, round_idx=False):
        """
        :param round_idx: if True, round the event coordinates to the nearest integer.
        :return: image-like representation of all the events in the validation time/event window.
        """

        return self.window_events_base(round_idx)

    def window_flow(self, mode=None, mask=None):
        """
        :return avg_flow: image-like representation of the per-pixel average flow in the validation time/event window.
        """

        if mask is None:
            mask = self.config["vis"]["mask_output"]

        # copy flow tensors to prevent overwriting
        flow_maps_x = self._flow_maps_x.clone()
        flow_maps_y = self._flow_maps_y.clone()

        # forward propagation of the estimated optical flow
        for i in range(self._passes - 1):
            warped_flow_x, warped_flow_y = self.forward_prop_flow(
                i, self._passes - 1, self._flow_maps_x, self._flow_maps_y
            )

            # update lists
            flow_maps_x[:, i : i + 1, ...] = warped_flow_x
            flow_maps_y[:, i : i + 1, ...] = warped_flow_y

        return self.window_flow_base(flow_maps_x, flow_maps_y, mask=mask)

    def window_iwe(self, mode=None, round_idx=False):
        """
        Assumption: events have NOT been previously warped in a forward fashion in the update() method.
        :param round_idx: if True, round the event coordinates to the nearest integer.
        :return: image-like representation of the IWE of all the events in the validation time/event window.
        """

        return self.window_iwe_base(round_idx)

    def rsat(self):
        """
        :return rsat: deblur metric for validation of the estimated optical flow.
        """

        fw_events = event_propagation(self._event_ts, self._event_loc, self._event_flow, self._passes)
        return self.compute_rsat(fw_events, self._event_loc, self._event_pol_mask, self._event_pol_mask, self._event_ts)

    def fwl(self):
        """
        :return fwl: deblur metric for validation of the estimated optical flow (Stoffregen et al, ECCV 2020).
        """

        fw_events = event_propagation(self._event_ts, self._event_loc, self._event_flow, self._passes)
        return self.compute_fwl(fw_events, self._event_loc, self._event_pol_mask, self._event_pol_mask)


class Iterative(BaseValidation):
    """
    Iterative event warping validation class.
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        self._fw_event_loc = None
        self._fw_event_warp_ts = None
        self._fw_event_pol_mask = None

        self._bw_event_loc = None
        self._bw_event_pol_mask = None

        self._fw_prop_flow_maps_x = None
        self._fw_prop_flow_maps_y = None

        self._accum_flow_map_x = None
        self._accum_flow_map_y = None
        self._flow_warping_indices = None
        self._flow_out_mask = torch.zeros(1, 1, self.res[0], self.res[1]).to(device)

    def update_fw_event_lists(self, event_list, event_pol_mask):
        """
        Initialize/Update container lists of events to be udpated during foward warping
        :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
        :param event_pol_mask: [batch_size x N x 2] event polarity mask
        """

        event_ts = event_list[:, :, 0:1].clone()
        if self.config["loss"]["round_ts"]:
            event_ts[...] = event_ts.min() + 0.5

        if self._fw_event_loc is None:
            self._fw_event_warp_ts = event_ts
            self._fw_event_loc = event_list[:, :, 1:3].clone()
            self._fw_event_pol_mask = event_pol_mask.clone()

        else:
            self._fw_event_warp_ts = torch.cat([self._fw_event_warp_ts, event_ts], dim=1)
            self._fw_event_loc = torch.cat([self._fw_event_loc, event_list[:, :, 1:3].clone()], dim=1)
            self._fw_event_pol_mask = torch.cat([self._fw_event_pol_mask, event_pol_mask.clone()], dim=1)

    def update_bw_event_lists(self, event_loc, event_pol_mask):
        """
        Initialize/Update container lists of events to be udpated during foward warping
        :param event_list: [batch_size x N x 2] input events (ts, y, x, p)
        :param event_pol_mask: [batch_size x N x 2] event polarity mask
        """

        if self._bw_event_loc is None:
            self._bw_event_loc = event_loc.clone()
            self._bw_event_pol_mask = event_pol_mask.clone()

        else:
            self._bw_event_loc = torch.cat([self._bw_event_loc, event_loc.clone()], dim=1)
            self._bw_event_pol_mask = torch.cat([self._bw_event_pol_mask, event_pol_mask.clone()], dim=1)

    def update(self, flow_list, event_list, pol_mask, event_mask):
        """
        Initialize/Update container lists of events and flow maps for forward warping.
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
        :param pol_mask: [batch_size x N x 2] polarity mask (pos, neg)
        :param event_mask: [batch_size x 1 x H x W] event mask
        """

        # update base lists (event data, flow maps, event masks)
        self.update_base(flow_list, event_list, pol_mask, event_mask)

        ############
        # FORWARD WARPING
        ############

        # initialize and update event lists for fw warping
        self.update_fw_event_lists(event_list, pol_mask)

        # sample optical flow
        fw_event_flow = get_event_flow(
            self._flow_maps_x[:, -1, ...],
            self._flow_maps_y[:, -1, ...],
            self._fw_event_loc,
        )

        # event warping process
        self._fw_event_loc = event_propagation(
            self._fw_event_warp_ts,
            self._fw_event_loc,
            fw_event_flow,
            self._passes + 1,
        )
        self._fw_event_loc, self._fw_event_pol_mask = purge_unfeasible(
            self._fw_event_loc,
            self._fw_event_pol_mask,
            self.res,
        )

        # update warping times
        self._fw_event_warp_ts[...] = self._passes + 1

        ############
        # BACKWARD WARPING
        ############

        bw_event_loc = event_list[:, :, 1:3].clone()
        bw_event_pol_mask = pol_mask.clone()
        bw_event_warp_ts = event_list[:, :, 0:1].clone()
        if self.config["loss"]["round_ts"]:
            bw_event_warp_ts[...] = bw_event_warp_ts.min() + 0.5

        cnt = 0
        while self._passes + cnt >= 0:

            # sample optical flow
            bw_event_flow = get_event_flow(
                self._flow_maps_x[:, self._passes + cnt, ...],
                self._flow_maps_y[:, self._passes + cnt, ...],
                bw_event_loc,
            )

            # event warping process
            bw_event_loc = event_propagation(
                bw_event_warp_ts,
                bw_event_loc,
                bw_event_flow,
                self._passes + cnt,
            )
            bw_event_loc, bw_event_pol_mask = purge_unfeasible(
                bw_event_loc,
                bw_event_pol_mask,
                self.res,
            )

            # update warping times
            bw_event_warp_ts[...] = self._passes + cnt
            cnt -= 1

        self.update_bw_event_lists(bw_event_loc, bw_event_pol_mask)

        ########################
        # FORWARD-PROPAGATED FLOW
        ########################

        # forward propagation of the estimated optical flow
        flow = flow_list[-1]  # only highest resolution flow
        if self._fw_prop_flow_maps_x is None:
            self._fw_prop_flow_maps_x = flow[:, 0:1, :, :]
            self._fw_prop_flow_maps_y = flow[:, 1:2, :, :]
        else:
            self._fw_prop_flow_maps_x = torch.cat([self._fw_prop_flow_maps_x, flow[:, 0:1, :, :]], dim=1)
            self._fw_prop_flow_maps_y = torch.cat([self._fw_prop_flow_maps_y, flow[:, 1:2, :, :]], dim=1)

        for i in range(self._passes):
            warped_flow_x, warped_flow_y = self.forward_prop_flow(
                i, i + 1, self._fw_prop_flow_maps_x, self._fw_prop_flow_maps_y
            )
            self._fw_prop_flow_maps_x[:, i : i + 1, ...] = warped_flow_x
            self._fw_prop_flow_maps_y[:, i : i + 1, ...] = warped_flow_y

        ########################
        # ACCUMULATED FLOW (BACKWARD WARPING)
        ########################

        indices = self.indices_map.clone()
        if self._flow_warping_indices is not None:
            indices = self._flow_warping_indices.clone()

        mask_valid = (
            (indices[:, 0:1, ...] >= 0)
            * (indices[:, 0:1, ...] <= self.res[0] - 1.0)
            * (indices[:, 1:2, ...] >= 0)
            * (indices[:, 1:2, ...] <= self.res[1] - 1.0)
        )
        self._flow_out_mask += mask_valid.float()

        curr_flow = get_event_flow(
            self._flow_maps_x[:, -1, ...],
            self._flow_maps_y[:, -1, ...],
            indices.view(1, 2, -1).permute(0, 2, 1),
        )
        curr_flow = curr_flow.permute(0, 2, 1).view(1, 2, self.res[0], self.res[1])

        warped_indices = indices + curr_flow * mask_valid.float()
        self._accum_flow_map_x = warped_indices[:, 1:2, :, :] - self.indices_map[:, 1:2, :, :]
        self._accum_flow_map_y = warped_indices[:, 0:1, :, :] - self.indices_map[:, 0:1, :, :]
        self._flow_warping_indices = warped_indices

        # update timestamp index
        self._passes += 1

    def reset(self):
        """
        Reset lists.
        """

        self.reset_base()
        self._fw_event_loc = None
        self._fw_event_warp_ts = None
        self._fw_event_pol_mask = None

        self._bw_event_loc = None
        self._bw_event_pol_mask = None

        self._fw_prop_flow_maps_x = None
        self._fw_prop_flow_maps_y = None

        self._accum_flow_map_x = None
        self._accum_flow_map_y = None
        self._flow_warping_indices = None
        self._flow_out_mask = torch.zeros(1, 1, self.res[0], self.res[1]).to(self.device)

    def window_events(self, round_idx=False):
        """
        :param round_idx: if True, round the event coordinates to the nearest integer.
        :return: image-like representation of all the events in the validation time/event window.
        """

        return self.window_events_base(round_idx)

    def window_flow(self, mode=None, mask=None):
        """
        :return avg_flow: image-like representation of the per-pixel average flow in the validation time/event window.
        """

        if mask is None:
            mask = self.config["vis"]["mask_output"]

        if mode == "forward":
            return self.window_flow_base(self._fw_prop_flow_maps_x, self._fw_prop_flow_maps_y, mask=mask)
        elif mode == "backward":
            return self.window_flow_base(
                self._accum_flow_map_x / self._flow_out_mask, self._accum_flow_map_y / self._flow_out_mask, mask=mask
            )
        else:
            return self.window_flow_base(self._flow_maps_x, self._flow_maps_y, mask=mask)

    def window_iwe(self, mode="forward", round_idx=False):
        """
        Assumption: events have been warped in a forward fashion in the update() method.
        :param round_idx: if True, round the event coordinates to the nearest integer.
        :return: image-like representation of the IWE of all the events in the validation time/event window.
        """

        if mode == "forward":
            event_loc = self._fw_event_loc
            pol_mask_list = self._fw_event_pol_mask
        elif mode == "backward":
            event_loc = self._bw_event_loc
            pol_mask_list = self._bw_event_pol_mask
        else:
            raise ValueError("Invalid IWE mode: {}".format(mode))

        if not round_idx:
            pol_mask_list = torch.cat([pol_mask_list for _ in range(4)], dim=1)

        idx, weights = get_interpolation(event_loc, self.res, round_idx=round_idx)
        iwe_pos = interpolate(idx.long(), weights, self.res, polarity_mask=pol_mask_list[:, :, 0:1])
        iwe_neg = interpolate(idx.long(), weights, self.res, polarity_mask=pol_mask_list[:, :, 1:2])

        return torch.cat([iwe_pos, iwe_neg], dim=1)

    def rsat(self):
        """
        :return: deblur metric for validation of the estimated optical flow.
        """

        return self.compute_rsat(
            self._fw_event_loc, self._event_loc, self._fw_event_pol_mask, self._event_pol_mask, self._event_ts
        )

    def fwl(self):
        """
        :return fwl: deblur metric for validation of the estimated optical flow (Stoffregen et al, ECCV 2020).
        """

        return self.compute_fwl(self._fw_event_loc, self._event_loc, self._fw_event_pol_mask, self._event_pol_mask)
