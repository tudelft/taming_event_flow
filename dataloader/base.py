from abc import abstractmethod

import cv2
import numpy as np
import random
import torch

from .encodings import events_to_voxel, events_to_channels


class BaseDataLoader(torch.utils.data.Dataset):
    """
    Base class for dataloader.
    """

    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.seq_num = 0
        self.samples = 0
        self.new_seq = False
        self.rectify = False
        self.device = self.config["loader"]["device"]
        self.res = self.config["loader"]["resolution"]
        self.batch_size = self.config["loader"]["batch_size"]

        # batch-specific data augmentation mechanisms
        self.batch_augmentation = {}
        for mechanism in self.config["loader"]["augment"]:
            self.batch_augmentation[mechanism] = [False for i in range(self.config["loader"]["batch_size"])]

        for i, mechanism in enumerate(self.config["loader"]["augment"]):
            for batch in range(self.config["loader"]["batch_size"]):
                if np.random.random() < self.config["loader"]["augment_prob"][i]:
                    self.batch_augmentation[mechanism][batch] = True

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def get_events(self, history):
        raise NotImplementedError

    def reset_sequence(self, batch):
        """
        Reset sequence-specific variables.
        :param batch: batch index
        """

        self.seq_num += 1

        # data augmentation
        for i, mechanism in enumerate(self.config["loader"]["augment"]):
            if np.random.random() < self.config["loader"]["augment_prob"][i]:
                self.batch_augmentation[mechanism][batch] = True
            else:
                self.batch_augmentation[mechanism][batch] = False

    def rectification_mapping(self, batch):
        """
        Compute the backward rectification map for the input representations.
        See https://github.com/uzh-rpg/DSEC/issues/14 for details.
        :param batch: batch index
        :return K_rect: intrinsic matrix of rectified image
        :return mapping: rectification map
        :return Q_rect: scaling matrix to convert disparity to depth
        """

        # distorted image
        K_dist = eval(self.open_files[batch]["calibration/intrinsics"][()])["cam0"]["camera_matrix"]

        # rectified image
        K_rect = eval(self.open_files[batch]["calibration/intrinsics"][()])["camRect0"]["camera_matrix"]
        R_rect = eval(self.open_files[batch]["calibration/extrinsics"][()])["R_rect0"]
        dist_coeffs = eval(self.open_files[batch]["calibration/intrinsics"][()])["cam0"]["distortion_coeffs"]

        # formatting
        K_dist = np.array([[K_dist[0], 0, K_dist[2]], [0, K_dist[1], K_dist[3]], [0, 0, 1]])
        K_rect = np.array([[K_rect[0], 0, K_rect[2]], [0, K_rect[1], K_rect[3]], [0, 0, 1]])
        R_rect = np.array(
            [
                [R_rect[0][0], R_rect[0][1], R_rect[0][2]],
                [R_rect[1][0], R_rect[1][1], R_rect[1][2]],
                [R_rect[2][0], R_rect[2][1], R_rect[2][2]],
            ]
        )
        dist_coeffs = np.array([dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3]])

        # backward mapping
        mapping = cv2.initUndistortRectifyMap(
            K_dist,
            dist_coeffs,
            R_rect,
            K_rect,
            (self.res[1], self.res[0]),
            cv2.CV_32FC2,
        )[0]

        # disparity to depth (onyl used for evaluation)
        Q_rect = eval(self.open_files[batch]["calibration/disparity_to_depth"][()])["cams_03"]
        Q_rect = np.array(
            [
                [Q_rect[0][0], Q_rect[0][1], Q_rect[0][2], Q_rect[0][3]],
                [Q_rect[1][0], Q_rect[1][1], Q_rect[1][2], Q_rect[1][3]],
                [Q_rect[2][0], Q_rect[2][1], Q_rect[2][2], Q_rect[2][3]],
                [Q_rect[3][0], Q_rect[3][1], Q_rect[3][2], Q_rect[3][3]],
            ]
        ).astype(np.float32)

        for _, mechanism in enumerate(self.config["loader"]["augment"]):

            if mechanism == "Horizontal":
                if self.batch_augmentation["Horizontal"][batch]:
                    K_rect[0, 2] = self.res[1] - 1 - K_rect[0, 2]
                    mapping[:, :, 0] = self.res[1] - 1 - mapping[:, :, 0]
                    mapping = np.flip(mapping, axis=1)
                    Q_rect[0, 3] = -K_rect[0, 2]

            elif mechanism == "Vertical":
                if self.batch_augmentation["Vertical"][batch]:
                    K_rect[1, 2] = self.res[0] - 1 - K_rect[1, 2]
                    mapping[:, :, 1] = self.res[0] - 1 - mapping[:, :, 1]
                    mapping = np.flip(mapping, axis=0)
                    Q_rect[1, 3] = -K_rect[1, 2]

        return K_rect, mapping, Q_rect

    @staticmethod
    def format_intrinsics(K_rect):
        """
        Format camera matrices.
        :param K_rect: [3 x 3] intrinsic matrix (numpy) of rectified image
        :return K_rect: [4 x 4] intrinsic matrix (tensor) of rectified image
        :return inv_K_rect: [4 x 4] inverse of the intrinsic matrix (tensor) of rectified image
        """

        K_rect = np.c_[K_rect, np.zeros(3)]
        K_rect = np.concatenate((K_rect, np.array([[0, 0, 0, 1]])), axis=0)
        inv_K_rect = np.linalg.pinv(K_rect)

        K_rect = torch.from_numpy(K_rect.astype(np.float32))
        inv_K_rect = torch.from_numpy(inv_K_rect.astype(np.float32))

        return K_rect, inv_K_rect

    def event_formatting(self, xs, ys, ts, ps):
        """
        Format input events as torch tensors.
        :param xs: [N] numpy array with event x location
        :param ys: [N] numpy array with event y location
        :param ts: [N] numpy array with event timestamp
        :param ps: [N] numpy array with event polarity ([-1, 1])
        :return rectified_xs: [N] numpy array with rectified event x location
        :return rectified_ys: [N] numpy array with rectified event y location
        :return xs: [N] tensor with event x location
        :return ys: [N] tensor with event y location
        :return ts: [N] tensor with normalized event timestamp
        :return ps: [N] tensor with event polarity ([-1, 1])
        """

        assert len(xs) == len(ys) == len(ts) == len(ps)

        xs = torch.from_numpy(xs.astype(np.float32)).to(self.device)
        ys = torch.from_numpy(ys.astype(np.float32)).to(self.device)
        ts = torch.from_numpy(ts.astype(np.float32)).to(self.device)
        ps = torch.from_numpy(ps.astype(np.float32)).to(self.device) * 2 - 1
        if ts.shape[0] > 0:
            ts = (ts - ts[0]) / (ts[-1] - ts[0])

        return xs, ys, ts, ps

    @staticmethod
    def rectify_events(rectify_map, xs, ys):
        """
        Rectify (and undistort) input events.
        :param rectify_map: map used to rectify events
        :param xs: [N] numpy array with event x location
        :param ys: [N] numpy array with event y location
        :return rectified_xs: [N] numpy array with rectified event x location
        :return rectified_ys: [N] numpy array with rectified event y location
        """

        rectified_events = rectify_map[ys.long(), xs.long()]
        rectified_xs = rectified_events[:, 0]
        rectified_ys = rectified_events[:, 1]

        return rectified_xs, rectified_ys

    def augment_events(self, xs, ys, ps, rec_xs, rec_ys, batch):
        """
        Augment event sequence with horizontal, vertical, and polarity flips.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ps: [N] tensor with event polarity ([-1, 1])
        :param rec_xs: [N] tensor with rectified event x location
        :param rec_ys: [N] tensor with rectified event y location
        :param batch: batch index
        :return xs: [N] tensor with augmented event x location
        :return ys: [N] tensor with augmented event y location
        :return ps: [N] tensor with augmented event polarity ([-1, 1])
        :return rec_xs: [N] tensor with augmented rectified event x location
        :return rec_ys: [N] tensor with augmented rectified event y location
        """

        for _, mechanism in enumerate(self.config["loader"]["augment"]):

            if mechanism == "Horizontal":
                if self.batch_augmentation["Horizontal"][batch]:
                    xs = self.res[1] - 1 - xs
                    if rec_xs is not None:
                        rec_xs = self.res[1] - 1 - rec_xs

            elif mechanism == "Vertical":
                if self.batch_augmentation["Vertical"][batch]:
                    ys = self.res[0] - 1 - ys
                    if rec_ys is not None:
                        rec_ys = self.res[0] - 1 - rec_ys

            elif mechanism == "Polarity":
                if self.batch_augmentation["Polarity"][batch]:
                    ps *= -1

        return xs, ys, ps, rec_xs, rec_ys

    def augment_gt(self, gt, batch):
        """
        Augment ground truth data with horizontal and vertical.
        :param gt: dictionary containing ground truth data
        :param batch: batch index
        """

        for _, mechanism in enumerate(self.config["loader"]["augment"]):

            if mechanism == "Horizontal":
                if self.batch_augmentation["Horizontal"][batch]:
                    for key in gt.keys():
                        gt[key] = torch.flip(gt[key], dims=[2])
                        if key == "gtflow":
                            gt[key][0, ...] *= -1

            elif mechanism == "Vertical":
                if self.batch_augmentation["Vertical"][batch]:
                    for key in gt.keys():
                        gt[key] = torch.flip(gt[key], dims=[1])
                        if key == "gtflow":
                            gt[key][1, ...] *= -1

        return gt

    @staticmethod
    def create_list_encoding(xs, ys, ts, ps):
        """
        Creates a four channel tensor with all the events in the input partition.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 4] list event representation
        """

        return torch.stack([ts, ys, xs, ps])

    @staticmethod
    def create_polarity_mask(ps):
        """
        Creates a two channel tensor that acts as a mask for the input event list.
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 2] polarity list event representation
        """

        event_list_pol_mask = torch.stack([ps, ps])
        event_list_pol_mask[0, :][event_list_pol_mask[0, :] < 0] = 0
        event_list_pol_mask[0, :][event_list_pol_mask[0, :] > 0] = 1
        event_list_pol_mask[1, :][event_list_pol_mask[1, :] < 0] = -1
        event_list_pol_mask[1, :][event_list_pol_mask[1, :] > 0] = 0
        event_list_pol_mask[1, :] *= -1
        return event_list_pol_mask

    def create_cnt_encoding(self, xs, ys, ps, rect_mapping):
        """
        Creates a per-pixel and per-polarity event count representation.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ps: [N] tensor with event polarity ([-1, 1])
        :param rect_mapping: map used to rectify events
        :return [2 x H x W] rectified event count representation
        """

        # create event count representation and rectify it using backward mapping
        event_cnt = events_to_channels(xs, ys, ps, sensor_size=self.res)
        if rect_mapping is not None:
            event_cnt = event_cnt.permute(1, 2, 0)
            event_cnt = cv2.remap(event_cnt.cpu().numpy(), rect_mapping, None, cv2.INTER_NEAREST)
            event_cnt = torch.from_numpy(event_cnt.astype(np.float32)).to(self.device)
            event_cnt = event_cnt.permute(2, 0, 1)

        return event_cnt

    @staticmethod
    def create_mask_encoding(event_cnt):
        """
        Creates per-pixel event mask based on event count.
        :param event_cnt: [2 x H x W] event count
        :return [H x W] rectified event mask representation
        """

        event_mask = event_cnt.clone()
        event_mask = torch.sum(event_mask, dim=0, keepdim=True)
        event_mask[event_mask > 0.0] = 1.0

        return event_mask

    def create_voxel_encoding(self, xs, ys, ts, ps, rect_mapping, num_bins=5):
        """
        Creates a spatiotemporal voxel grid tensor representation with a certain number of bins,
        as described in Section 3.1 of the paper 'Unsupervised Event-based Learning of Optical Flow,
        Depth, and Egomotion', Zhu et al., CVPR'19..
        Events are distributed to the spatiotemporal closest bins through bilinear interpolation.
        Positive events are added as +1, while negative as -1.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :param rect_mapping: map used to rectify events
        :param num_bins: number of bins in the voxel grid
        :return [B x H x W] rectified voxel grid event representation
        """

        event_voxel = events_to_voxel(
            xs,
            ys,
            ts,
            ps,
            num_bins,
            sensor_size=self.res,
        )

        if rect_mapping is not None:
            event_voxel = event_voxel.permute(1, 2, 0)
            event_voxel = cv2.remap(event_voxel.cpu().numpy(), rect_mapping, None, cv2.INTER_NEAREST)
            event_voxel = torch.from_numpy(event_voxel.astype(np.float32)).to(self.device)
            event_voxel = event_voxel.permute(2, 0, 1)

        return event_voxel

    @staticmethod
    def split_event_list(event_list, event_list_pol_mask, max_num_grad_events):
        """
        Splits the event list into two lists, one of them (with max. length) to be used for backprop.
        This helps reducing (VRAM) memory consumption.
        :param event_list: [4 x N] list event representation
        :param event_list_pol_mask: [2 x N] polarity list event representation
        :param max_num_grad_events: maximum number of events to be used for backprop
        :return event_list: [4 x N] list event representation to be used for backprop
        :return event_list_pol_mask: [2 x N] polarity list event representation to be used for backprop
        :return d_event_list: [4 x N] list event representation
        :return d_event_list_pol_mask: [2 x N] polarity list event representation
        """

        d_event_list = torch.zeros((4, 0))
        d_event_list_pol_mask = torch.zeros((2, 0))
        if max_num_grad_events is not None and event_list.shape[1] > max_num_grad_events:
            probs = torch.ones(event_list.shape[1], dtype=torch.float32) / event_list.shape[1]
            sampled_indices = probs.multinomial(
                max_num_grad_events, replacement=False
            )  # sample indices with equal prob.

            unsampled_indices = torch.ones(event_list.shape[1], dtype=torch.bool)
            unsampled_indices[sampled_indices] = False
            d_event_list = event_list[:, unsampled_indices]
            d_event_list_pol_mask = event_list_pol_mask[:, unsampled_indices]

            event_list = event_list[:, sampled_indices]
            event_list_pol_mask = event_list_pol_mask[:, sampled_indices]

        return event_list, event_list_pol_mask, d_event_list, d_event_list_pol_mask

    def __len__(self):
        return 1000  # not used

    def shuffle(self, flag=True):
        """
        Shuffles the training data.
        :param flag: if true, shuffles the data
        """

        if flag:
            random.shuffle(self.files)

    @staticmethod
    def custom_collate(batch):
        """
        Collects the different event representations and stores them together in a dictionary.
        :param batch: batch index
        :return batch_dict: dictionary with the output of a dataloader iteration
        """

        # create dictionary
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = []

        # collect data
        for entry in batch:
            for key in entry.keys():
                batch_dict[key].append(entry[key])

        # create batches
        for key in batch_dict.keys():

            if batch_dict[key][0] is not None:

                # pad entries of different size
                N = 0
                if key in ["event_list", "event_list_pol_mask", "d_event_list", "d_event_list_pol_mask"]:
                    for i in range(len(batch_dict[key])):
                        if N < batch_dict[key][i].shape[1]:
                            N = batch_dict[key][i].shape[1]

                    for i in range(len(batch_dict[key])):
                        zeros = torch.zeros((batch_dict[key][i].shape[0], N - batch_dict[key][i].shape[1]))
                        batch_dict[key][i] = torch.cat((batch_dict[key][i], zeros), dim=1)

                # create tensor
                item = torch.stack(batch_dict[key])
                if len(item.shape) == 3:
                    item = item.transpose(2, 1)
                batch_dict[key] = item

            else:
                batch_dict[key] = None

        return batch_dict
