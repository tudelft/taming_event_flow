import argparse

import mlflow
import torch

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow_val import *
from models.model import *
from utils.iwe import compute_pol_iwe
from utils.utils import load_model, create_model_dir, initialize_quant_results
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization


def test(args, config_parser):
    """
    Main function of the evaluation pipeline for event-based optical flow estimation.
    :param args: arguments of the script
    :param config_parser: YAMLParser object with config data
    """

    mlflow.set_tracking_uri(args.path_mlflow)

    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)
    config = config_parser.combine_entries(config)

    # configs
    config["loader"]["batch_size"] = 1

    # create directory for inference results
    path_results = create_model_dir(args.path_results, args.runid)

    # store validation settings
    eval_id = log_config(path_results, args.runid, config)

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs
    config["loader"]["device"] = device

    # visualization tool
    vis = Visualization(config, eval_id=eval_id, path_results=path_results)

    # data loader
    data = H5Loader(config, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # model initialization and settings
    num_bins = 2 if config["data"]["voxel"] is None else config["data"]["voxel"]
    model = eval(config["model"]["name"])(config["model"].copy(), num_bins)
    model = model.to(device)
    model, _ = load_model(args.runid, model, device)
    model.eval()

    # validation metric
    criteria = eval(config["metrics"]["warping"])(config, device)
    val_results = {}

    # inference loop
    end_test = False
    with torch.no_grad():
        while not end_test:
            for inputs in dataloader:
                sequence = data.files[data.batch_idx[0] % len(data.files)].split("/")[-1].split(".")[0]

                if data.new_seq:
                    data.new_seq = False
                    model.reset_states()
                    criteria.reset()

                if config["data"]["mode"] in ["gtflow"] and data.ts_jump_reset:
                    data.ts_jump_reset = False
                    model.reset_states()

                # finish inference loop
                if data.seq_num >= len(data.files):
                    end_test = True
                    break

                # forward pass
                x = model(inputs["net_input"].to(device))
                for i in range(len(x["flow"])):
                    x["flow"][i] = x["flow"][i] * config["loss"]["flow_scaling"]

                # mask flow for visualization
                flow_vis = x["flow"][-1].clone()
                if config["vis"]["mask_output"]:
                    flow_vis *= inputs["event_mask"].to(device)

                # image of warped events
                iwe = None
                if (config["vis"]["enabled"] or config["vis"]["store"]) and (
                    config["vis"]["show"] is None or "iwe" in config["vis"]["show"]
                ):
                    iwe = compute_pol_iwe(
                        flow_vis,
                        inputs["event_list"].to(device),
                        config["loader"]["resolution"],
                        inputs["event_list_pol_mask"].to(device),
                        round_idx=False,
                        round_flow=False,
                    )

                # update validation criteria
                criteria.update(
                    x["flow"],
                    inputs["event_list"].to(device),
                    inputs["event_list_pol_mask"].to(device),
                    inputs["event_mask"].to(device),
                )

                # prepare for visualization
                if config["vis"]["enabled"] or config["vis"]["store"]:

                    # dynamic windows
                    if config["data"]["passes_loss"] > 1 and config["vis"]["dynamic"]:
                        vis.data["events_dynamic"] = criteria.window_events()
                        vis.data["iwe_fw_dynamic"] = criteria.window_iwe(mode="forward")
                        vis.data["iwe_bw_dynamic"] = criteria.window_iwe(mode="backward")
                        vis.data["flow_dynamic"] = criteria.window_flow(mode="forward")

                    # accumulated windows
                    if criteria.num_passes > 1 and criteria.num_passes == config["data"]["passes_loss"]:
                        vis.data["events_window"] = criteria.window_events()
                        vis.data["iwe_fw_window"] = criteria.window_iwe(mode="forward")
                        vis.data["iwe_bw_window"] = criteria.window_iwe(mode="backward")
                        vis.data["flow_window"] = criteria.window_flow(mode="forward")

                # compute error metrics
                vis.data["flow_bw"] = None
                val_results = initialize_quant_results(val_results, sequence, config["metrics"]["name"])
                if criteria.num_passes == config["data"]["passes_loss"]:

                    compute_metrics = True
                    if "eval_time" in config["metrics"].keys():
                        if (
                            data.last_proc_timestamp < config["metrics"]["eval_time"][0]
                            or data.last_proc_timestamp > config["metrics"]["eval_time"][1]
                        ):
                            compute_metrics = False

                    if compute_metrics:

                        # AEE
                        if config["data"]["mode"] == "gtflow" and "AEE" in config["metrics"]["name"]:
                            mask_aee = None
                            if "mask_aee" in config["metrics"].keys() and config["metrics"]["mask_aee"]:
                                mask_aee = criteria.window_events().clone().to(device)

                            vis.data["flow_bw"] = (
                                criteria.window_flow(mode="backward", mask=False) * config["data"]["passes_loss"]
                            )
                            aee = criteria.compute_aee(vis.data["flow_bw"], inputs["gtflow"].to(device), mask=mask_aee)
                            val_results[sequence]["AEE"]["it"] += 1
                            val_results[sequence]["AEE"]["metric"] += aee.cpu().numpy()

                        # deblurring metrics
                        for metric in config["metrics"]["name"]:
                            if metric == "RSAT":
                                rsat = criteria.rsat()
                                val_results[sequence][metric]["metric"] += rsat[0].cpu().numpy()
                                val_results[sequence][metric]["it"] += 1

                            elif metric == "FWL":
                                fwl = criteria.fwl()
                                val_results[sequence][metric]["metric"] += fwl.cpu().numpy()
                                val_results[sequence][metric]["it"] += 1

                    # reset criteria
                    criteria.reset()

                # visualization
                if config["vis"]["bars"]:
                    for bar in data.open_files_bar:
                        bar.next()
                if config["vis"]["enabled"] or config["vis"]["store"]:
                    vis.data["iwe"] = iwe
                    vis.data["flow"] = flow_vis
                    vis.step(
                        inputs,
                        sequence=sequence,
                        ts=data.last_proc_timestamp,
                        show=config["vis"]["show"],
                    )

    if config["vis"]["bars"]:
        for bar in data.open_files_bar:
            bar.finish()

    # store validation config and results
    results = {}
    for metric in config["metrics"]["name"]:
        results[metric] = {}
        for key in val_results.keys():
            if val_results[key][metric]["it"] > 0:
                results[metric][key] = str(val_results[key][metric]["metric"] / val_results[key][metric]["it"])
        log_results(args.runid, results, path_results, eval_id)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow run")
    parser.add_argument(
        "--config",
        default="configs/eval_flow.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))
