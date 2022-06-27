from abc import abstractmethod
import os
import sys

import math
import torch

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from utils.iwe import event_propagation, get_event_flow, purge_unfeasible, get_interpolation, interpolate


class BaseEventWarping(torch.nn.Module):
    """
    Base class for the contrast maximization loss.
    """

    def __init__(self, config, device, loss_scaling=True, border_compensation=True):
        super(BaseEventWarping, self).__init__()
        self.device = device
        self.config = config
        self.loss_scaling = loss_scaling
        self.border_compensation = border_compensation
        self.res = config["loader"]["resolution"]
        self.batch_size = config["loader"]["batch_size"]
        self.flow_spat_smooth_weight = config["loss"]["flow_spat_smooth_weight"]
        self.flow_temp_smooth_weight = config["loss"]["flow_temp_smooth_weight"]

        self._passes = 0
        self._num_flows = None
        self._flow_maps_x = None
        self._flow_maps_y = None

        # warping indices (for temporal consistency)
        my, mx = torch.meshgrid(torch.arange(self.res[0]), torch.arange(self.res[1]))
        indices = torch.stack([my, mx], dim=0).unsqueeze(0)
        self.indices = indices.float().to(device)
        self.indices_mask = torch.ones(self.indices.shape).to(device)

        # timescales for loss computation
        self.passes_loss = []
        for s in range(config["data"]["scales_loss"]):
            self.passes_loss.append(config["data"]["passes_loss"] // (2**s))

    def update_base(self, flow_list):
        """
        Initialize/Update container lists of events and flow maps for forward warping.
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        """

        if self._num_flows is None:
            self._num_flows = len(flow_list)

        # update optical flow maps
        if self._flow_maps_x is None:
            self._flow_maps_x = []
            self._flow_maps_y = []

        for i, flow in enumerate(flow_list):
            if i == len(self._flow_maps_x):
                self._flow_maps_x.append(flow[:, 0:1, :, :])
                self._flow_maps_y.append(flow[:, 1:2, :, :])
            else:
                self._flow_maps_x[i] = torch.cat([self._flow_maps_x[i], flow[:, 0:1, :, :]], dim=1)
                self._flow_maps_y[i] = torch.cat([self._flow_maps_y[i], flow[:, 1:2, :, :]], dim=1)

    def reset_base(self):
        """
        Reset lists.
        """

        self._passes = 0
        self._flow_maps_x = None
        self._flow_maps_y = None

    @property
    def num_passes(self):
        return self._passes

    def iwe_formatting(self, warped_events, pol_mask, ts_list, tref, ts_scaling, interp_zeros=None, iwe_zeros=None):
        """
        Compues the images of warped events with event count and (accumulated) event timestamp information.
        :param warped_events: [batch_size x N x 2] warped events
        :param pol_mask: [batch_size x 4*N x 2] polarity mask of the (forward) warped events
        :param ts_list: [batch_size x N x 1] event timestamp [0, max_ts]
        :param tref: reference timestamp
        :param ts_scaling: with of the timestamp normalization window
        :return iwe: image of warped events
        :return iwe_ts: image of averaged timestamps
        """

        # normalize timestamps (the ref timestamp is 1, the rest decreases linearly towards 0 on the extremes)
        norm_ts = ts_list.clone()
        norm_ts = 1 - torch.abs(tref - norm_ts) / ts_scaling

        # interpolate forward
        idx, weights = get_interpolation(warped_events, self.res, zeros=interp_zeros)

        # per-polarity image of (forward) warped events
        iwe_pos = interpolate(idx, weights, self.res, polarity_mask=pol_mask[:, :, 0:1], zeros=iwe_zeros)
        iwe_neg = interpolate(idx, weights, self.res, polarity_mask=pol_mask[:, :, 1:2], zeros=iwe_zeros)
        iwe = torch.cat([iwe_pos, iwe_neg], dim=1)

        # image of (forward) warped averaged timestamps
        iwe_pos_ts = interpolate(idx, weights * norm_ts, self.res, polarity_mask=pol_mask[:, :, 0:1], zeros=iwe_zeros)
        iwe_neg_ts = interpolate(idx, weights * norm_ts, self.res, polarity_mask=pol_mask[:, :, 1:2], zeros=iwe_zeros)
        iwe_ts = torch.cat([iwe_pos_ts, iwe_neg_ts], dim=1)

        return iwe, iwe_ts

    def focus_loss(self, iwe, iwe_ts):
        """
        Scaling of the loss function based on the number of events in the image space.
        See "Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks",
        Hagenaars and Paredes-Valles et al., NeurIPS 2021
        :param iwe: [batch_size x N x 2] image of warped events
        :param iwe_ts: [batch_size x N x 2] image of averaged timestamps
        :return loss: loss value
        """

        iwe_ts = iwe_ts.view(iwe_ts.shape[0], 2, -1)
        loss = torch.sum(iwe_ts[:, 0, :] ** 2, dim=1) + torch.sum(iwe_ts[:, 1, :] ** 2, dim=1)
        if self.loss_scaling:
            nonzero_px = torch.sum(iwe, dim=1, keepdim=True).bool()
            nonzero_px = nonzero_px.view(nonzero_px.shape[0], -1)
            loss /= torch.sum(nonzero_px, dim=1) + 1e-9

        return torch.sum(loss)

    def flow_temporal_smoothing(self):
        """
        (Temporal) Scaled Charbonnier smoothness prior on the estimated optical flow vectors.
        :return loss: smoothing loss value
        """

        loss = 0
        for i in range(self._num_flows):
            for j in range(self._flow_maps_x[i].shape[1] - 1):

                # compute (backward) warping indices
                flow = torch.stack([self._flow_maps_y[i][:, j, ...], self._flow_maps_x[i][:, j, ...]], dim=1)
                warped_indices = self.indices + flow
                warped_indices = warped_indices.view(self.batch_size, 2, -1).permute(0, 2, 1)

                # ignore pixels that go out of the image space
                warping_mask = (
                    (warped_indices[..., 0] >= 0)
                    * (warped_indices[..., 0] <= self.res[0] - 1.0)
                    * (warped_indices[..., 1] >= 0)
                    * (warped_indices[..., 1] <= self.res[1] - 1.0)
                )

                # (backward) warp the next flow
                warped_flow = get_event_flow(
                    self._flow_maps_x[i][:, j + 1, ...], self._flow_maps_y[i][:, j + 1, ...], warped_indices
                )
                warped_flow = warped_flow.permute(0, 2, 1).view(self.batch_size, 2, self.res[0], self.res[1])

                # compute flow temporal consistency (charbonnier)
                flow_dt = torch.sqrt((flow - warped_flow) ** 2 + 1e-9)
                flow_dt = torch.sum(flow_dt, dim=1, keepdim=True).view(self.batch_size, -1)
                loss += torch.sum(flow_dt * warping_mask, dim=1) / (torch.sum(warping_mask, dim=1) + 1e-9)

        loss /= self._num_flows
        loss /= self._passes - 1

        return self.flow_temp_smooth_weight * loss.sum()

    def flow_spatial_smoothing(self):
        """
        (Spatial) Scaled Charbonnier smoothness prior on the estimated optical flow vectors.
        :return loss: smoothing loss value
        """

        loss = 0
        for i in range(self._num_flows):

            # forward differences (horizontal, vertical, and diagonals)
            flow_x_dx = self._flow_maps_x[i][:, :, :, :-1] - self._flow_maps_x[i][:, :, :, 1:]
            flow_y_dx = self._flow_maps_y[i][:, :, :, :-1] - self._flow_maps_y[i][:, :, :, 1:]
            flow_x_dy = self._flow_maps_x[i][:, :, :-1, :] - self._flow_maps_x[i][:, :, 1:, :]
            flow_y_dy = self._flow_maps_y[i][:, :, :-1, :] - self._flow_maps_y[i][:, :, 1:, :]
            flow_x_dxdy_dr = self._flow_maps_x[i][:, :, :-1, :-1] - self._flow_maps_x[i][:, :, 1:, 1:]
            flow_y_dxdy_dr = self._flow_maps_y[i][:, :, :-1, :-1] - self._flow_maps_y[i][:, :, 1:, 1:]
            flow_x_dxdy_ur = self._flow_maps_x[i][:, :, 1:, :-1] - self._flow_maps_x[i][:, :, :-1, 1:]
            flow_y_dxdy_ur = self._flow_maps_y[i][:, :, 1:, :-1] - self._flow_maps_y[i][:, :, :-1, 1:]

            # compute flow spatial consistency (charbonnier)
            flow_dx = torch.sqrt((flow_x_dx) ** 2 + 1e-6) + torch.sqrt((flow_y_dx) ** 2 + 1e-6)
            flow_dy = torch.sqrt((flow_x_dy) ** 2 + 1e-6) + torch.sqrt((flow_y_dy) ** 2 + 1e-6)
            flow_dxdy_dr = torch.sqrt((flow_x_dxdy_dr) ** 2 + 1e-6) + torch.sqrt((flow_y_dxdy_dr) ** 2 + 1e-6)
            flow_dxdy_ur = torch.sqrt((flow_x_dxdy_ur) ** 2 + 1e-6) + torch.sqrt((flow_y_dxdy_ur) ** 2 + 1e-6)

            flow_dx = flow_dx.view(self.batch_size, self._passes, -1)
            flow_dy = flow_dy.view(self.batch_size, self._passes, -1)
            flow_dxdy_dr = flow_dxdy_dr.view(self.batch_size, self._passes, -1)
            flow_dxdy_ur = flow_dxdy_ur.view(self.batch_size, self._passes, -1)

            flow_dx = flow_dx.mean(2).mean(1)
            flow_dy = flow_dy.mean(2).mean(1)
            flow_dxdy_ur = flow_dxdy_ur.mean(2).mean(1)
            flow_dxdy_dr = flow_dxdy_dr.mean(2).mean(1)

            loss += (flow_dx + flow_dy + flow_dxdy_dr + flow_dxdy_ur) / 4

        loss /= self._num_flows

        return self.flow_spat_smooth_weight * loss.sum()

    @abstractmethod
    def forward(self):
        raise NotImplementedError


class Linear(BaseEventWarping):
    """
    Contrast maximization loss from Hagenaars and Paredes-Valles et al. (NeurIPS 2021).
    """

    def __init__(self, config, device, loss_scaling=True):
        super().__init__(config, device, loss_scaling=loss_scaling)
        self._event_ts = []
        self._event_loc = []
        self._event_flow = []
        self._event_pol_mask = []

        self._d_event_ts = []
        self._d_event_loc = []
        self._d_event_flow = []
        self._d_event_pol_mask = []

    def update(self, flow_list, event_list, pol_mask, d_event_list, d_pol_mask):
        """
        Initialize/Update container lists of events and flow maps for forward warping.
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
        :param pol_mask: [batch_size x N x 2] polarity mask (pos, neg)
        :param d_event_list: [batch_size x N x 4] detached input events (ts, y, x, p)
        :param d_pol_mask: [batch_size x N x 2] detached polarity mask (pos, neg)
        """

        # update base lists (flow maps)
        self.update_base(flow_list)

        # update event timestamps
        event_list[:, :, 0:1] += self._passes
        d_event_list[:, :, 0:1] += self._passes
        event_ts = event_list[:, :, 0:1].clone()
        d_event_ts = d_event_list[:, :, 0:1].clone()
        if self.config["loss"]["round_ts"]:
            event_ts[...] = event_ts.min() + 0.5
            d_event_ts[...] = d_event_ts.min() + 0.5
        self._event_ts.append(event_ts)
        self._d_event_ts.append(d_event_ts)

        # update event location
        self._event_loc.append(event_list[:, :, 1:3].clone())
        self._d_event_loc.append(d_event_list[:, :, 1:3].clone())

        # update polarity mask
        self._event_pol_mask.append(pol_mask.clone())
        self._d_event_pol_mask.append(d_pol_mask.clone())

        # update per-event flow vector
        event_flow = []
        d_event_flow = []
        for i in range(self._num_flows):
            event_flow.append(
                get_event_flow(
                    self._flow_maps_x[i][:, -1, ...],
                    self._flow_maps_y[i][:, -1, ...],
                    event_list[:, :, 1:3],
                )
            )
            with torch.no_grad():
                d_event_flow.append(
                    get_event_flow(
                        self._flow_maps_x[i][:, -1, ...],
                        self._flow_maps_y[i][:, -1, ...],
                        d_event_list[:, :, 1:3],
                    )
                )
        self._event_flow.append(event_flow)
        self._d_event_flow.append(d_event_flow)

        # update timestamp index
        self._passes += 1

    def reset(self):
        """
        Reset lists.
        """

        self.reset_base()
        self._event_ts = []
        self._event_loc = []
        self._event_flow = []
        self._event_pol_mask = []

        self._d_event_ts = []
        self._d_event_loc = []
        self._d_event_flow = []
        self._d_event_pol_mask = []

    def forward(self):

        loss = 0
        for s, scale in enumerate(self.passes_loss):

            loss_update = 0
            for w in range(2**s):
                low_pass = w * scale
                high_pass = (w + 1) * scale

                event_ts = torch.cat(self._event_ts[low_pass:high_pass], dim=1)
                event_loc = torch.cat(self._event_loc[low_pass:high_pass], dim=1)
                ts_list = torch.cat([event_ts for _ in range(4)], dim=1)

                d_event_ts = torch.cat(self._d_event_ts[low_pass:high_pass], dim=1)
                d_event_loc = torch.cat(self._d_event_loc[low_pass:high_pass], dim=1)
                d_ts_list = torch.cat([d_event_ts for _ in range(4)], dim=1)

                if not self.border_compensation:
                    event_pol_mask = torch.cat(self._event_pol_mask[low_pass:high_pass], dim=1)
                    event_pol_mask = torch.cat([event_pol_mask for _ in range(4)], dim=1)
                    d_event_pol_mask = torch.cat(self._d_event_pol_mask[low_pass:high_pass], dim=1)
                    d_event_pol_mask = torch.cat([d_event_pol_mask for _ in range(4)], dim=1)

                for i in range(self._num_flows):
                    if self.border_compensation:
                        event_pol_mask = torch.cat(self._event_pol_mask[low_pass:high_pass], dim=1)
                        d_event_pol_mask = torch.cat(self._d_event_pol_mask[low_pass:high_pass], dim=1)

                    # event propagation (with grads)
                    event_flow = torch.cat([flow[i] for flow in self._event_flow[low_pass:high_pass]], dim=1)
                    fw_events = event_propagation(event_ts, event_loc, event_flow, high_pass)
                    bw_events = event_propagation(event_ts, event_loc, event_flow, low_pass)

                    if self.border_compensation:
                        fw_events, event_pol_mask = purge_unfeasible(fw_events, event_pol_mask, self.res)
                        bw_events, event_pol_mask = purge_unfeasible(bw_events, event_pol_mask, self.res)
                        event_pol_mask = torch.cat([event_pol_mask for _ in range(4)], dim=1)

                    fw_iwe, fw_iwe_ts = self.iwe_formatting(
                        fw_events,
                        event_pol_mask,
                        ts_list,
                        high_pass,
                        scale,
                    )
                    bw_iwe, bw_iwe_ts = self.iwe_formatting(
                        bw_events,
                        event_pol_mask,
                        ts_list,
                        low_pass,
                        scale,
                    )

                    # event propagation (without grads)
                    d_event_flow = torch.cat([flow[i] for flow in self._d_event_flow[low_pass:high_pass]], dim=1)
                    d_fw_events = event_propagation(d_event_ts, d_event_loc, d_event_flow, high_pass)
                    d_bw_events = event_propagation(d_event_ts, d_event_loc, d_event_flow, low_pass)
                    if self.border_compensation:
                        d_fw_events, d_event_pol_mask = purge_unfeasible(d_fw_events, d_event_pol_mask, self.res)
                        d_bw_events, d_event_pol_mask = purge_unfeasible(d_bw_events, d_event_pol_mask, self.res)
                        d_event_pol_mask = torch.cat([d_event_pol_mask for _ in range(4)], dim=1)

                    d_fw_iwe, d_fw_iwe_ts = self.iwe_formatting(
                        d_fw_events,
                        d_event_pol_mask,
                        d_ts_list,
                        high_pass,
                        scale,
                    )
                    d_bw_iwe, d_bw_iwe_ts = self.iwe_formatting(
                        d_bw_events,
                        d_event_pol_mask,
                        d_ts_list,
                        low_pass,
                        scale,
                    )

                    # compute loss (forward)
                    fw_iwe = fw_iwe + d_fw_iwe
                    fw_iwe_ts = fw_iwe_ts + d_fw_iwe_ts
                    fw_iwe_ts = fw_iwe_ts / (fw_iwe + 1e-9)  # per-pixel and per-polarity timestamps
                    loss_update += self.focus_loss(fw_iwe, fw_iwe_ts)

                    # compute loss (backward)
                    bw_iwe = bw_iwe + d_bw_iwe
                    bw_iwe_ts = bw_iwe_ts + d_bw_iwe_ts
                    bw_iwe_ts = bw_iwe_ts / (bw_iwe + 1e-9)  # per-pixel and per-polarity timestamps
                    loss_update += self.focus_loss(bw_iwe, bw_iwe_ts)

            loss_update /= 2**s  # number of deblurring windows for a given scale
            loss_update /= 2  # number of deblurring points per deblurring window
            loss += loss_update

        # average loss over all flow predictions
        loss /= self.config["data"]["scales_loss"]
        loss /= self._num_flows

        # spatial smoothing of predicted flow vectors
        if self.flow_spat_smooth_weight is not None:
            loss += self.flow_spatial_smoothing()

        # temporal consistency of predicted flow vectors
        if self.flow_temp_smooth_weight is not None and self._passes > 1:
            loss += self.flow_temporal_smoothing()

        return loss


class Iterative(BaseEventWarping):
    """
    Contrast maximization loss from Hagenaars and Paredes-Valles et al. (NeurIPS 2021) but augmented
    with iterative event warping, loss computation at all intermediate (time) points, and multiple temporal scales.
    """

    def __init__(self, config, device, loss_scaling=True):
        if config["loss"]["iterative_mode"] == "four":
            config["data"]["passes_loss"] *= 2

        super().__init__(config, device, loss_scaling=loss_scaling)
        self._event_ts = []
        self._event_loc = []
        self._event_pol_mask = []

        self._d_event_ts = []
        self._d_event_loc = []
        self._d_event_pol_mask = []

        self.delta_passes = []
        for passes in self.passes_loss:
            if config["loss"]["iterative_mode"] == "one":
                self.delta_passes.append(passes // 1)
            elif config["loss"]["iterative_mode"] == "two":
                self.delta_passes.append(passes // 2)
            elif config["loss"]["iterative_mode"] == "four":
                self.delta_passes.append(passes // 4)

    def update(self, flow_list, event_list, pol_mask, d_event_list, d_pol_mask):
        """
        Initialize/Update container lists of events and flow maps for forward warping.
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        :param event_list: [batch_size x N x 4] input events (ts, y, x, p)
        :param pol_mask: [batch_size x N x 2] polarity mask (pos, neg)
        :param d_event_list: [batch_size x N x 4] detached input events (ts, y, x, p)
        :param d_pol_mask: [batch_size x N x 2] detached polarity mask (pos, neg)
        """

        # update base lists (event data, flow maps, event masks)
        self.update_base(flow_list)

        # update event timestamps
        event_list[:, :, 0:1] += self._passes
        d_event_list[:, :, 0:1] += self._passes
        event_ts = event_list[:, :, 0:1].clone()
        d_event_ts = d_event_list[:, :, 0:1].clone()
        if self.config["loss"]["round_ts"]:
            event_ts[...] = event_ts.min() + 0.5
            d_event_ts[...] = d_event_ts.min() + 0.5
        self._event_ts.append(event_ts)
        self._d_event_ts.append(d_event_ts)

        # update event locations
        self._event_loc.append(event_list[:, :, 1:3].clone())
        self._d_event_loc.append(d_event_list[:, :, 1:3].clone())

        # update event polarity masks
        self._event_pol_mask.append(pol_mask.clone())
        self._d_event_pol_mask.append(d_pol_mask.clone())

        # update timestamp index
        self._passes += 1

    def reset(self):
        """
        Reset lists.
        """

        self.reset_base()
        self._event_ts = []
        self._event_loc = []
        self._event_pol_mask = []

        self._d_event_ts = []
        self._d_event_loc = []
        self._d_event_pol_mask = []

    def update_warping_indices(self, tref, t, cnt, mode="forward"):
        """
        Updates indices for forward/backward iterative event warping.
        :param tref: reference timestamp for deblurring
        :param t: current (starting) timestamp
        :param cnt: counter that counts the number of warping passes a set of events have undergone
        :param mode: "forward" or "backward" warping mode
        :return cnt: updated counter
        :return break_flag: flag indicating whether the warping should be performed
        :return sampling_idx: indices for flow sampling
        :return warping_ts: reference time for current warping
        """

        warping_ts = t + cnt
        sampling_idx = t + cnt
        if mode == "forward":
            warping_ts += 1
            break_flag = t + cnt < tref
            cnt += 1

        elif mode == "backward":
            break_flag = t + cnt >= tref
            cnt -= 1

        else:
            raise ValueError("Unknown warping mode: {}".format(mode))

        return cnt, break_flag, sampling_idx, warping_ts

    def event_warping(
        self,
        tref,
        t,
        i,
        mode,
        buffer_event_loc,
        buffer_event_ts,
        buffer_event_pol_mask,
        warped_events,
        warped_events_ts,
        warped_events_mask,
    ):
        """
        Perform forward/backward iterative event warping.
        :param tref: reference timestamp for deblurring
        :param t: current (starting) timestamp
        :param i: index indicating which optical flow map to use
        :param mode: "forward" or "backward" warping mode
        :param buffer_event_loc: [batch_size x N x 2] buffer for event locations
        :param buffer_event_ts: [batch_size x N x 1] buffer for event timestamps
        :param buffer_event_pol_mask: [batch_size x N x 2] buffer for event polarity masks
        :param warped_events: [[batch_size x N x 2]] list containing warped events at all trefs
        :param warped_events_ts: [[batch_size x N x 1]] list containing warped events timestamp at all trefs
        :param warped_events_mask: [[batch_size x N x 2]] list containing polarity masks for warped events at all trefs
        :return warped_events: [[batch_size x N x 2]] updated list containing warped events at all trefs
        :return warped_events_ts: [[batch_size x N x 1]] updated list containing warped events timestamp at all trefs
        :return warped_events_mask: [[batch_size x N x 2]] updated list containing polarity masks for warped events at all trefs
        """

        cnt = 0
        event_loc = buffer_event_loc[t].clone()
        event_warp_ts = buffer_event_ts[t].clone()
        event_pol_mask = buffer_event_pol_mask[t].clone()
        while True:
            cnt, break_flag, sampling_idx, warping_ts = self.update_warping_indices(tref, t, cnt, mode=mode)
            if not break_flag:
                break

            # sample optical flow
            event_flow = get_event_flow(
                self._flow_maps_x[i][:, sampling_idx, ...],
                self._flow_maps_y[i][:, sampling_idx, ...],
                event_loc,
            )

            # event warping
            event_loc = event_propagation(
                event_warp_ts,
                event_loc,
                event_flow,
                warping_ts,
            )
            event_warp_ts[...] = warping_ts
            event_loc, event_pol_mask = purge_unfeasible(
                event_loc,
                event_pol_mask,
                self.res,
            )

            # update warping information (when in range)
            warped_events[warping_ts][t] = event_loc.clone()
            warped_events_ts[warping_ts][t] = buffer_event_ts[t].clone()
            warped_events_mask[warping_ts][t] = event_pol_mask.clone()

        return warped_events, warped_events_ts, warped_events_mask

    def forward(self):

        loss = 0
        max_passes = max(self.passes_loss)
        for i in range(self._num_flows):
            none_list = [None for _ in range(max_passes)]

            # iterative event warping
            event_ts = [none_list.copy() for _ in range(max_passes + 1)]
            event_loc = [none_list.copy() for _ in range(max_passes + 1)]
            event_pol_mask = [none_list.copy() for _ in range(max_passes + 1)]
            for t in range(max_passes):
                event_loc, event_ts, event_pol_mask = self.event_warping(
                    max_passes,
                    t,
                    i,
                    "forward",
                    self._event_loc,
                    self._event_ts,
                    self._event_pol_mask,
                    event_loc,
                    event_ts,
                    event_pol_mask,
                )
                event_loc, event_ts, event_pol_mask = self.event_warping(
                    0,
                    t,
                    i,
                    "backward",
                    self._event_loc,
                    self._event_ts,
                    self._event_pol_mask,
                    event_loc,
                    event_ts,
                    event_pol_mask,
                )

            # detached iterative event warping
            d_event_ts = [none_list.copy() for _ in range(max_passes + 1)]
            d_event_loc = [none_list.copy() for _ in range(max_passes + 1)]
            d_event_pol_mask = [none_list.copy() for _ in range(max_passes + 1)]
            with torch.no_grad():
                for t in range(max_passes):
                    d_event_loc, d_event_ts, d_event_pol_mask = self.event_warping(
                        max_passes,
                        t,
                        i,
                        "forward",
                        self._d_event_loc,
                        self._d_event_ts,
                        self._d_event_pol_mask,
                        d_event_loc,
                        d_event_ts,
                        d_event_pol_mask,
                    )
                    d_event_loc, d_event_ts, d_event_pol_mask = self.event_warping(
                        0,
                        t,
                        i,
                        "backward",
                        self._d_event_loc,
                        self._d_event_ts,
                        self._d_event_pol_mask,
                        d_event_loc,
                        d_event_ts,
                        d_event_pol_mask,
                    )

            # learning from multiple temporal scales (i.e., different amounts of blur)
            for s, scale in enumerate(self.passes_loss):

                loss_update = 0
                for w in range(2**s):
                    low_pass = w * scale
                    high_pass = (w + 1) * scale

                    low_tref = low_pass
                    high_tref = high_pass + 1
                    if self.config["loss"]["iterative_mode"] == "four":
                        low_tref = low_pass + self.delta_passes[s]
                        high_tref = low_pass + 3 * self.delta_passes[s] + 1

                    # merge event masks (to deal with partially-observable edges)
                    if self.border_compensation:
                        shared_event_pol_mask = none_list.copy()
                        shared_d_event_pol_mask = none_list.copy()
                        for t in range(low_tref, high_tref - 1):
                            tmp_event_pol_mask = event_pol_mask[low_tref][t].clone()
                            tmp_d_event_pol_mask = d_event_pol_mask[low_tref][t].clone()
                            for tref in range(low_tref + 1, high_tref):
                                tmp_event_pol_mask *= event_pol_mask[tref][t].clone()
                                tmp_d_event_pol_mask *= d_event_pol_mask[tref][t].clone()
                            shared_event_pol_mask[t] = tmp_event_pol_mask
                            shared_d_event_pol_mask[t] = tmp_d_event_pol_mask

                    # compute loss at all intermediate points
                    for tref in range(low_tref, high_tref):
                        low_extreme = max(low_pass, tref - self.delta_passes[s])
                        high_extreme = min(high_pass, tref + self.delta_passes[s])

                        # image of warped events (with grads)
                        ts_list = torch.cat(event_ts[tref][low_extreme:high_extreme], dim=1)
                        warped_events = torch.cat(event_loc[tref][low_extreme:high_extreme], dim=1)
                        if self.border_compensation:
                            warped_pol_mask = torch.cat(shared_event_pol_mask[low_extreme:high_extreme], dim=1)
                        else:
                            warped_pol_mask = torch.cat(event_pol_mask[tref][low_extreme:high_extreme], dim=1)

                        ts_list = torch.cat([ts_list for _ in range(4)], dim=1)
                        warped_pol_mask = torch.cat([warped_pol_mask for _ in range(4)], dim=1)
                        iwe, iwe_ts = self.iwe_formatting(
                            warped_events,
                            warped_pol_mask,
                            ts_list,
                            tref,
                            self.delta_passes[s],
                        )

                        # image of warped events (without grads)
                        d_ts_list = torch.cat(d_event_ts[tref][low_extreme:high_extreme], dim=1)
                        d_warped_events = torch.cat(d_event_loc[tref][low_extreme:high_extreme], dim=1)
                        if self.border_compensation:
                            d_warped_pol_mask = torch.cat(shared_d_event_pol_mask[low_extreme:high_extreme], dim=1)
                        else:
                            d_warped_pol_mask = torch.cat(d_event_pol_mask[tref][low_extreme:high_extreme], dim=1)

                        d_ts_list = torch.cat([d_ts_list for _ in range(4)], dim=1)
                        d_warped_pol_mask = torch.cat([d_warped_pol_mask for _ in range(4)], dim=1)
                        d_iwe, d_iwe_ts = self.iwe_formatting(
                            d_warped_events,
                            d_warped_pol_mask,
                            d_ts_list,
                            tref,
                            self.delta_passes[s],
                        )

                        # combination of images of warped events (with and without grads)
                        iwe = iwe + d_iwe
                        iwe_ts = iwe_ts + d_iwe_ts
                        iwe_ts = iwe_ts / (iwe + 1e-9)  # per-pixel and per-polarity timestamps
                        loss_update += self.focus_loss(iwe, iwe_ts)

                loss_update /= 2**s  # number of deblurring windows for a given scale
                loss_update /= 2 * self.delta_passes[s] + 1  # number of deblurring points per deblurring window
                loss += loss_update

        # average loss over all flow predictions and deblurring points
        loss /= self.config["data"]["scales_loss"]
        loss /= self._num_flows

        # spatial smoothing of predicted flow vectors
        if self.flow_spat_smooth_weight is not None:
            loss += self.flow_spatial_smoothing()

        # temporal consistency of predicted flow vectors
        if self.flow_temp_smooth_weight is not None and self._passes > 1:
            loss += self.flow_temporal_smoothing()

        return loss
