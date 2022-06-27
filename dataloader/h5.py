import os
import sys
import cv2
import hdf5plugin
import h5py
import numpy as np

import torch

from .base import BaseDataLoader
from .cache import CacheDataset
from .utils import ProgressBar

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from utils.utils import binary_search_array


class FlowMaps:
    """
    Utility class for reading the ground truth optical flow maps encoded in the HDF5 files.
    """

    def __init__(self):
        self.ts_from = []
        self.ts_to = []
        self.names = []

    def __call__(self, name, h5obj):
        if hasattr(h5obj, "dtype") and name not in self.names:
            self.names += [name]
            self.ts_from += [h5obj.attrs["timestamp_from"]]
            self.ts_to += [h5obj.attrs["timestamp_to"]]


class H5Loader(BaseDataLoader):
    def __init__(self, config, shuffle=False, path_cache=""):
        super().__init__(config)
        self.input_window = self.config["data"]["window"]
        if self.config["data"]["mode"] in ["gtflow"] and self.input_window > 1:
            print("DataLoader error: Ground truth data mode cannot be used with window > 1.")
            raise AttributeError

        self.ts_jump = False
        self.ts_jump_reset = False  # used in the inference loops to reset model states

        self.gt_avg_dt = None
        self.gt_avg_idx = 0
        self.last_proc_timestamp = 0

        # "memory" that goes from forward pass to the next
        self.batch_idx = [i for i in range(self.batch_size)]  # event sequence
        self.batch_row = [0 for i in range(self.batch_size)]  # event_idx / time_idx
        self.batch_pass = [0 for i in range(self.batch_size)]  # forward passes

        # input event sequences
        self.files = []
        for root, dirs, files in os.walk(config["data"]["path"]):
            for file in files:
                if file.endswith(".h5"):
                    self.files.append(os.path.join(root, file))

        # shuffle files
        if shuffle:
            self.shuffle()

        # initialize cache
        if self.config["data"]["cache"]:
            self.cache = CacheDataset(config, path_cache)

        # open first files
        self.open_files = []
        self.batch_rectify_map = []
        self.batch_K_rect = []
        self.batch_Q_rect = []
        self.batch_rect_mapping = []
        for batch in range(self.config["loader"]["batch_size"]):
            self.open_files.append(h5py.File(self.files[self.batch_idx[batch] % len(self.files)], "r"))
            if "rectification" in self.open_files[-1].keys():
                self.batch_rectify_map.append(self.open_files[-1]["rectification/rectify_map"][:])
                self.batch_rectify_map[-1] = torch.from_numpy(self.batch_rectify_map[-1]).float().to(self.device)

                K_rect, mapping, Q_rect = self.rectification_mapping(-1)
                self.batch_K_rect.append(K_rect)
                self.batch_Q_rect.append(Q_rect)
                self.batch_rect_mapping.append(mapping)
                self.rectify = True
            else:
                self.batch_rect_mapping.append(None)

        # load GT optical flow maps from open files
        self.open_files_flowmaps = []
        if config["data"]["mode"] == "gtflow":
            for batch in range(self.batch_size):
                flowmaps = FlowMaps()
                if "flow" in self.open_files[batch].keys():
                    self.open_files[batch]["flow"].visititems(flowmaps)
                self.open_files_flowmaps.append(flowmaps)

        # progress bars
        if self.config["vis"]["bars"]:
            self.open_files_bar = []
            for batch in range(self.config["loader"]["batch_size"]):
                max_iters = self.get_iters(batch)
                self.open_files_bar.append(ProgressBar(self.files[batch].split("/")[-1], max=max_iters))

    def get_iters(self, batch):
        """
        Compute the number of forward passes given a sequence and an input mode and window.
        :param batch: batch index
        :return: number of forward passes
        """

        if self.config["data"]["mode"] == "events":
            max_iters = len(self.open_files[batch]["events/xs"])
        elif self.config["data"]["mode"] == "time":
            max_iters = self.open_files[batch].attrs["duration"]
        elif self.config["data"]["mode"] == "gtflow":
            max_iters = len(self.open_files_flowmaps[batch].ts_to) - 1
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError

        return max_iters // self.input_window

    def get_events(self, file, idx0, idx1):
        """
        Get all the events in between two indices.
        :param file: file to read from
        :param idx0: start index
        :param idx1: end index
        :return xs: [N] numpy array with event x location
        :return ys: [N] numpy array with event y location
        :return ts: [N] numpy array with event timestamp
        :return ps: [N] numpy array with event polarity ([-1, 1])
        """

        xs = file["events/xs"][idx0:idx1]
        ys = file["events/ys"][idx0:idx1]
        ts = file["events/ts"][idx0:idx1]
        ps = file["events/ps"][idx0:idx1]
        ts -= file.attrs["t0"]  # sequence starting at t0 = 0

        # check if temporal discontinuity in gt data modes
        self.ts_jump = False
        if self.config["data"]["mode"] in ["gtflow"]:
            dt = ts[-1] - self.last_proc_timestamp
            if self.gt_avg_dt is None:
                self.gt_avg_dt = dt
                self.gt_avg_idx += 1

            if dt >= 2 * self.gt_avg_dt / self.gt_avg_idx:
                self.ts_jump = True
                self.ts_jump_reset = True
            else:
                self.gt_avg_dt += dt
                self.gt_avg_idx += 1

        if ts.shape[0] > 0:
            self.last_proc_timestamp = ts[-1]
        return xs, ys, ts, ps

    def get_event_index(self, batch, window=0):
        """
        Get all the event indices to be used for reading.
        :param batch: batch index
        :param window: input window
        :return event_idx0: event index (from)
        :return event_idx1: event index (to)
        """

        restart = False
        event_idx0 = None
        event_idx1 = None
        if self.config["data"]["mode"] == "events":
            event_idx0 = self.batch_row[batch]
            event_idx1 = self.batch_row[batch] + window

        elif self.config["data"]["mode"] == "time":
            event_idx0 = self.find_ts_index(
                self.open_files[batch], self.batch_row[batch] + self.open_files[batch].attrs["t0"]
            )
            event_idx1 = self.find_ts_index(
                self.open_files[batch], self.batch_row[batch] + self.open_files[batch].attrs["t0"] + window
            )

        elif self.config["data"]["mode"] == "gtflow":
            idx1 = int(np.ceil(self.batch_row[batch] + window))
            if np.isclose(self.batch_row[batch] + window, idx1 - 1):
                idx1 -= 1
            event_idx0 = self.find_ts_index(self.open_files[batch], self.open_files_flowmaps[batch].ts_from[idx1])
            event_idx1 = self.find_ts_index(self.open_files[batch], self.open_files_flowmaps[batch].ts_to[idx1])
            if self.open_files_flowmaps[batch].ts_to[idx1] > self.open_files[batch].attrs["tk"]:
                restart = True

        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError

        return event_idx0, event_idx1, restart

    def find_ts_index(self, file, timestamp, dataset="events/ts"):
        """
        Find closest event index for a given timestamp through binary search.
        :param file: file to read from
        :param timestamp: timestamp to find
        :param dataset: dataset to search in
        :return: event index
        """

        return binary_search_array(file[dataset], timestamp)

    def open_new_h5(self, batch):
        """
        Open new H5 event sequence.
        :param batch: batch index
        """

        self.ts_jump = False
        self.ts_jump_reset = False

        self.gt_avg_dt = None
        self.gt_avg_idx = 0
        self.last_proc_timestamp = 0

        self.open_files[batch] = h5py.File(self.files[self.batch_idx[batch] % len(self.files)], "r+")

        if self.rectify:
            self.batch_rectify_map[batch] = self.open_files[batch]["rectification/rectify_map"][:]
            self.batch_rectify_map[batch] = torch.from_numpy(self.batch_rectify_map[batch]).float().to(self.device)

            K_rect, mapping, Q_rect = self.rectification_mapping(batch)
            self.batch_K_rect[batch] = K_rect
            self.batch_Q_rect[batch] = Q_rect
            self.batch_rect_mapping[batch] = mapping

        if self.config["data"]["mode"] == "gtflow":
            flowmaps = FlowMaps()
            if "flow" in self.open_files[batch].keys():
                self.open_files[batch]["flow"].visititems(flowmaps)
            self.open_files_flowmaps[batch] = flowmaps

        if self.config["vis"]["bars"]:
            self.open_files_bar[batch].finish()
            max_iters = self.get_iters(batch)
            self.open_files_bar[batch] = ProgressBar(
                self.files[self.batch_idx[batch] % len(self.files)].split("/")[-1], max=max_iters
            )

        if "Playback" in self.batch_augmentation.keys() and self.batch_augmentation["Playback"][batch]:
            file = self.open_files[batch]
            xs = np.flip(file["events/xs"][:])
            ys = np.flip(file["events/ys"][:])
            ps = np.flip(file["events/ps"][:])

            ts = np.flip(file["events/ts"][:])
            min_ts = ts[-1]
            max_ts = ts[0]
            ts = np.absolute((ts - min_ts) / (max_ts - min_ts) - 1)
            ts = ts * (max_ts - min_ts) + min_ts

            file["events/xs"][:] = xs
            file["events/ys"][:] = ys
            file["events/ts"][:] = ts
            file["events/ps"][:] = ps

    def __getitem__(self, index):
        while True:
            batch = index % self.config["loader"]["batch_size"]

            # try loading cached data
            if self.config["data"]["cache"]:
                output, success = self.cache.load(
                    self.files[self.batch_idx[batch] % len(self.files)], self.batch_pass[batch]
                )
                if success:
                    self.batch_row[batch] += self.input_window
                    self.batch_pass[batch] += 1
                    return output

            # trigger sequence change
            len_frames = 0
            restart = False
            if self.config["data"]["mode"] == "gtflow":
                len_frames = len(self.open_files_flowmaps[batch].ts_to)
                if int(np.ceil(self.batch_row[batch] + self.input_window)) >= len_frames:
                    restart = True

            # load events
            xs = np.zeros((0))
            ys = np.zeros((0))
            ts = np.zeros((0))
            ps = np.zeros((0))
            if not restart:
                idx0, idx1, restart = self.get_event_index(batch, window=self.input_window)

                if self.config["data"]["mode"] in ["gtflow"] and self.input_window < 1.0:
                    floor_row = int(np.floor(self.batch_row[batch]))
                    ceil_row = int(np.ceil(self.batch_row[batch] + self.input_window))

                    if np.isclose(self.batch_row[batch], floor_row + 1):
                        floor_row += 1
                    if np.isclose(self.batch_row[batch] + self.input_window, ceil_row - 1):
                        ceil_row -= 1

                    idx0_change = self.batch_row[batch] - floor_row
                    idx1_change = self.batch_row[batch] + self.input_window - floor_row

                    delta_idx = idx1 - idx0
                    idx1 = int(idx0 + idx1_change * delta_idx)
                    idx0 = int(idx0 + idx0_change * delta_idx)

                if not restart:
                    xs, ys, ts, ps = self.get_events(self.open_files[batch], idx0, idx1)

            # skip gt sample if temporal discontinuity in gt
            if self.config["data"]["mode"] in ["gtflow"] and self.ts_jump:
                self.batch_row[batch] += self.input_window
                self.batch_pass[batch] += 1
                continue

            # trigger sequence change
            if (self.config["data"]["mode"] == "events" and xs.shape[0] < self.input_window) or (
                self.config["data"]["mode"] == "time"
                and self.batch_row[batch] + self.input_window >= self.open_files[batch].attrs["duration"]
            ):
                restart = True

            # reset sequence if not enough input events
            if restart:
                self.new_seq = True
                self.reset_sequence(batch)
                self.batch_row[batch] = 0
                self.batch_idx[batch] = max(self.batch_idx) + 1
                self.batch_pass[batch] = 0
                self.open_files[batch].close()
                self.open_new_h5(batch)
                continue

            # handle case with very few events
            if xs.shape[0] <= 10:
                xs = np.empty([0])
                ys = np.empty([0])
                ts = np.empty([0])
                ps = np.empty([0])

            # event formatting and timestamp normalization
            xs, ys, ts, ps = self.event_formatting(xs, ys, ts, ps)

            # rectify input events
            rec_xs, rec_ys = None, None
            if self.rectify:
                rec_xs, rec_ys = self.rectify_events(self.batch_rectify_map[batch], xs, ys)

            # data augmentation
            xs, ys, ps, rec_xs, rec_ys = self.augment_events(xs, ys, ps, rec_xs, rec_ys, batch)

            # events to lists
            if self.rectify:
                event_list = self.create_list_encoding(rec_xs, rec_ys, ts, ps)
            else:
                event_list = self.create_list_encoding(xs, ys, ts, ps)
            event_list_pol_mask = self.create_polarity_mask(ps)

            # create event representations
            event_cnt = self.create_cnt_encoding(xs, ys, ps, self.batch_rect_mapping[batch])
            event_mask = self.create_mask_encoding(event_cnt)
            if self.config["data"]["voxel"] is not None:
                event_voxel = self.create_voxel_encoding(
                    xs, ys, ts, ps, self.batch_rect_mapping[batch], num_bins=self.config["data"]["voxel"]
                )

            # voxel is the preferred representation for the network's input
            if self.config["data"]["voxel"] is None:
                net_input = event_cnt.clone()
            else:
                net_input = event_voxel.clone()

            # load (and augment) GT maps when required
            gt = {}
            if self.config["data"]["mode"] == "gtflow":
                idx = int(np.ceil(self.batch_row[batch] + self.input_window))
                if np.isclose(self.batch_row[batch] + self.input_window, idx - 1):
                    idx -= 1
                flowmap = self.open_files[batch]["flow"][self.open_files_flowmaps[batch].names[idx]][:]
                flowmap = flowmap.astype(np.float32)
                flowmap = torch.from_numpy(flowmap).permute(2, 0, 1)
                gt["gtflow"] = flowmap
                gt["gtflow_dt"] = (
                    self.open_files_flowmaps[batch].ts_to[idx] - self.open_files_flowmaps[batch].ts_from[idx]
                )
                gt["gtflow_dt"] = torch.from_numpy(np.asarray(gt["gtflow_dt"])).float()

            gt = self.augment_gt(gt, batch)

            # update window
            self.batch_row[batch] += self.input_window
            self.batch_pass[batch] += 1

            # break while loop if everything went well
            break

        # camera matrix for rectified and cropped events
        if self.rectify:
            K_rect, inv_K_rect = self.format_intrinsics(self.batch_K_rect[batch].copy())

        # split event list (events with and without gradients)
        event_list, event_list_pol_mask, d_event_list, d_event_list_pol_mask = self.split_event_list(
            event_list, event_list_pol_mask, self.config["loader"]["max_num_grad_events"]
        )

        # prepare output
        output = {}
        output["net_input"] = net_input.cpu()
        output["event_cnt"] = event_cnt.cpu()
        output["event_mask"] = event_mask.cpu()
        output["event_list"] = event_list.cpu()
        output["event_list_pol_mask"] = event_list_pol_mask.cpu()
        output["d_event_list"] = d_event_list.cpu()
        output["d_event_list_pol_mask"] = d_event_list_pol_mask.cpu()
        if self.rectify:
            output["K_rect"] = K_rect
            output["inv_K_rect"] = inv_K_rect
        for key in gt.keys():
            output[key] = gt[key]

        if self.config["data"]["cache"]:
            self.cache.update(self.files[self.batch_idx[batch] % len(self.files)], output)

        return output
