import os

import cv2
import matplotlib
import numpy as np


class Visualization:
    """
    Utility class for the visualization and storage of rendered image-like representation of multiple elements of the pipeline.
    """

    def __init__(self, kwargs, eval_id=-1, path_results=None):
        self.img_idx = 0
        self.px = kwargs["vis"]["px"]
        self.res = kwargs["loader"]["resolution"]
        self.color_scheme = "green_red"  # gray / blue_red / green_red
        self.show_rendered = kwargs["vis"]["enabled"]
        self.store_rendered = kwargs["vis"]["store"]

        if eval_id >= 0 and path_results is not None:
            self.store_dir = path_results + "results/"
            self.store_dir = self.store_dir + "eval_" + str(eval_id) + "/"
            if not os.path.exists(self.store_dir):
                os.makedirs(self.store_dir)
            self.store_file = None

        self.data = {}
        self.keys = [
            "events",
            "events_window",
            "events_dynamic",
            "flow",
            "flow_window",
            "flow_dynamic",
            "flow_bw",
            "iwe",
            "iwe_fw_window",
            "iwe_bw_window",
            "iwe_fw_dynamic",
            "iwe_bw_dynamic",
            "flow_gt",
            "depth_gt",
        ]
        self.title = [
            "Input events",
            "Input events - Eval window",
            "Input events - Dynamic window",
            "Estimated flow",
            "Estimated flow - Eval window",
            "Estimated flow - Dynamic window",
            "Estimated flow - Backward",
            "IWE",
            "Forward IWE - Eval window",
            "Backward IWE - Eval window",
            "Forward IWE - Dynamic window",
            "Backward IWE - Dynamic window",
            "Ground truth flow",
            "Ground truth depth",
        ]

        self.reset_image_ph()

    def step(self, inputs, sequence=None, ts=None, show=None):
        """
        Main function of the visualization workflow.
        :param inputs: input data (output of the dataloader)
        """

        # render images
        self.render(inputs, show)

        # live display
        if self.show_rendered:
            self.update(show)

        # store rendered images
        if self.store_rendered and sequence is not None:
            self.store(sequence, ts, show)

        # reset image placeholders
        self.reset_image_ph()

    def reset_image_ph(self):
        """
        Initialize/Reset image placeholders.
        """
        for key in self.keys:
            self.data[key] = None

    def render(self, inputs, show=None):
        """
        Rendering tool.
        :param inputs: input data (output of the dataloader)
        """

        self.data["events"] = inputs["event_cnt"] if "event_cnt" in inputs.keys() else None
        if self.data["events"] is None:
            self.data["events"] = inputs["net_input"] if "net_input" in inputs.keys() else None

        self.data["flow_gt"] = inputs["gtflow"] if "gtflow" in inputs.keys() else None
        self.data["depth_gt"] = inputs["gtdepth"] if "gtdepth" in inputs.keys() else None

        # optical flow error
        if self.data["flow_bw"] is not None and self.data["flow_gt"] is not None:
            self.data["error_flow"] = (
                (self.data["flow_bw"].cpu() - self.data["flow_gt"]).pow(2).sum(1).sqrt().unsqueeze(1)
            )
            gtflow_mask = (self.data["flow_gt"][:, 0:1, :, :] == 0.0) * (self.data["flow_gt"][:, 1:2, :, :] == 0.0)
            self.data["error_flow"] *= ~gtflow_mask
            if "error_flow" not in self.keys:
                self.keys.append("error_flow")
                self.title.append("AEE (capped at 30px)")

        for key in self.keys:
            if show is not None:
                if key not in show:
                    continue

            if self.data[key] is not None:
                self.data[key] = self.data[key].detach()

            # input events
            if key.split("_")[0] == "events" and self.data[key] is not None:
                self.data[key] = (
                    self.data[key]
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                    .reshape((self.data[key].shape[2], self.data[key].shape[3], 2))
                )
                self.data[key] = self.events_to_image(self.data[key])

            # optical flow
            elif key.split("_")[0] == "flow" and self.data[key] is not None:
                self.data[key] = (
                    self.data[key]
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                    .reshape((self.data[key].shape[2], self.data[key].shape[3], 2))
                )
                if key != "flow_bw":
                    self.data[key] = self.flow_to_image(self.data[key])
                else:
                    self.data[key] = self.data[key] * 128 + 2**15
                    self.data[key] = self.data[key].astype(np.uint16)
                    self.data[key] = np.pad(self.data[key], ((0, 0), (0, 0), (0, 1)), constant_values=0)
                    self.data[key] = np.flip(self.data[key], axis=-1)

            # optical flow error
            elif key == "error_flow" and self.data[key] is not None:
                self.data[key] = (
                    self.data[key]
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                    .reshape((self.data[key].shape[2], self.data[key].shape[3], 1))
                )
                self.data[key] = self.minmax_norm(self.data[key], max=30, min=0)
                self.data[key] *= 255
                self.data[key] = self.data[key].astype(np.uint8)
                self.data[key] = cv2.applyColorMap(self.data[key], cv2.COLORMAP_VIRIDIS)

            # image of warped events
            elif key.split("_")[0] == "iwe" and self.data[key] is not None:
                self.data[key] = (
                    self.data[key]
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                    .reshape((self.data[key].shape[2], self.data[key].shape[3], 2))
                )
                self.data[key] = self.events_to_image(self.data[key])

    def update(self, show=None):
        """
        Live visualization of the previously-rendered images.
        """

        for i, key in enumerate(self.keys):
            if show is not None:
                if key not in show:
                    continue

            if key not in ["flow_bw"] and self.data[key] is not None:
                cv2.namedWindow(self.title[i], cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.title[i], int(self.px), int(self.px))
                cv2.imshow(self.title[i], self.data[key])

        cv2.waitKey(1)

    def store(self, sequence, ts=None, show=None):
        """
        Store previously-rendered images.
        :param sequence: name of the sequence
        :param ts: timestamp of the images to be stored
        """

        # check if new sequence
        path_to = self.store_dir + sequence + "/"
        if not os.path.exists(path_to):
            os.makedirs(path_to)
            for key in self.keys:
                os.makedirs(path_to + key + "/")
            if self.store_file is not None:
                self.store_file.close()
            self.store_file = open(path_to + "timestamps.txt", "w")
            self.img_idx = 0

        # store images
        for key in self.keys:
            if show is not None:
                if key not in show:
                    continue

            if not os.path.exists(path_to + key + "/"):
                os.makedirs(path_to + key + "/")
            if self.data[key] is not None:
                filename = path_to + key + "/%09d.png" % self.img_idx
                cv2.imwrite(filename, self.data[key])

        # store timestamps
        if ts is not None:
            self.store_file.write(str(ts) + "\n")
            self.store_file.flush()

        self.img_idx += 1
        cv2.waitKey(1)

    @staticmethod
    def flow_to_image(flow):
        """
        Use the optical flow color scheme from the supplementary materials of the paper 'Back to Event
        Basics: Self-Supervised Image Reconstruction for Event Cameras via Photometric Constancy',
        Paredes-Valles et al., CVPR'21.
        :param flow: [H x W x 2] optical flow map (horizontal, vertical in dim=2)
        :return: [H x W x 3] color-encoded optical flow in BGR format
        """
        mag = np.linalg.norm(flow, axis=2)
        min_mag = np.min(mag)
        mag_range = np.max(mag) - min_mag

        ang = np.arctan2(flow[:, :, 1], flow[:, :, 0]) + np.pi
        ang *= 1.0 / np.pi / 2.0

        hsv = np.zeros([flow.shape[0], flow.shape[1], 3])
        hsv[:, :, 0] = ang
        hsv[:, :, 1] = 1.0
        hsv[:, :, 2] = mag - min_mag
        if mag_range != 0.0:
            hsv[:, :, 2] /= mag_range

        flow_rgb = matplotlib.colors.hsv_to_rgb(hsv)
        flow_rgb = (255 * flow_rgb).astype(np.uint8)
        return cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def events_to_image(event_cnt, color_scheme="green_red"):
        """
        Format events into an image.
        :param event_cnt: [H x W x 2] event count map
        :param color_scheme: gray / blue_red / green_red
        """
        pos = event_cnt[:, :, 0]
        neg = event_cnt[:, :, 1]
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if pos_min != max:
            pos = (pos - pos_min) / (max - pos_min)
        if neg_min != max:
            neg = (neg - neg_min) / (max - neg_min)

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((event_cnt.shape[0], event_cnt.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 0][mask_neg] = 0
            event_image[:, :, 1][mask_neg * mask_not_pos] = 0

        elif color_scheme == "rpg":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            mask_pos = pos > 0
            mask_neg = neg > 0

            event_image[:, :, 0][mask_neg] = 1
            event_image[:, :, 1][mask_neg] = 0
            event_image[:, :, 2][mask_neg] = 0
            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = 0
            event_image[:, :, 2][mask_pos] = 1

        elif color_scheme == "prophesee":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            mask_pos = pos > 0
            mask_neg = neg > 0

            event_image[:, :, 0][mask_neg] = 0.24313725490196078
            event_image[:, :, 1][mask_neg] = 0.11764705882352941
            event_image[:, :, 2][mask_neg] = 0.047058823529411764
            event_image[:, :, 0][mask_pos] = 0.6352941176470588
            event_image[:, :, 1][mask_pos] = 0.4235294117647059
            event_image[:, :, 2][mask_pos] = 0.23529411764705882

        else:
            print("Visualization error: Unknown color scheme for event images.")
            raise AttributeError

        event_image = (255 * event_image).astype(np.uint8)
        return event_image

    @staticmethod
    def minmax_norm(x, max=None, min=None):
        """
        Robust min-max normalization.
        :param x: [H x W x 1]
        :return x: [H x W x 1] normalized x
        """

        if max is None:
            max = np.percentile(x, 99)
        if min is None:
            min = np.percentile(x, 1)

        den = max - min
        if den != 0:
            x = (x - min) / den
        return np.clip(x, 0, 1)
