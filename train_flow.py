import argparse

import mlflow
import torch
from torch.optim import *
from torch.utils.tensorboard import SummaryWriter

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import *
from models.model import *
from utils.utils import load_model, save_diff, save_model
from utils.visualization import Visualization


def train(args, config_parser):
    """
    Main function of the training pipeline for event-based optical flow estimation.
    :param args: arguments of the script
    :param config_parser: YAMLParser object with config data
    """

    mlflow.set_tracking_uri(args.path_mlflow)

    # configs
    config = config_parser.config
    mlflow.set_experiment(config["experiment"])
    run = mlflow.start_run()
    runid = run.to_dictionary()["info"]["run_id"]
    mlflow.log_params(config)
    mlflow.log_param("prev_runid", args.prev_runid)
    config = config_parser.combine_entries(config)
    print("MLflow dir:", mlflow.active_run().info.artifact_uri[:-9])

    # log git diff
    save_diff("train_diff.txt")
    tb_writer = SummaryWriter(log_dir=args.path_mlflow + "mlruns/0/" + runid + "/")

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs
    config["loader"]["device"] = device

    # visualization tool
    if config["vis"]["enabled"]:
        vis = Visualization(config)

    # data loader
    data = H5Loader(config, shuffle=True, path_cache=args.path_cache)
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
    model = eval(config["model"]["name"])(config["model"].copy(), num_bins, key="flow")
    model = model.to(device)
    model, epoch = load_model(args.prev_runid, model, device, curr_run=run, tb_writer=tb_writer)
    model.train()

    # loss functions
    loss_function = eval(config["loss"]["warping"])(config, device)

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer.zero_grad()

    # simulation variables
    train_loss = 0
    best_loss = 1.0e6
    end_train = False
    data.epoch = epoch

    # dataloader loop
    while True:
        for inputs in dataloader:

            if data.new_seq:
                data.new_seq = False
                loss_function.reset()
                model.reset_states()
                optimizer.zero_grad()

            if data.seq_num >= len(data.files):
                tb_writer.add_scalar("loss", train_loss / data.samples, data.epoch)
                mlflow.log_metric("loss", train_loss / data.samples, step=data.epoch)
                with torch.no_grad():
                    if train_loss / data.samples < best_loss:
                        save_model(model)
                        best_loss = train_loss / data.samples

                data.epoch += 1
                data.samples = 0
                train_loss = 0
                data.seq_num = data.seq_num % len(data.files)
                if data.epoch == config["loader"]["n_epochs"]:
                    end_train = True
                    break

            # forward pass (flow in px/input_time)
            x = model(inputs["net_input"].to(device))
            for i in range(len(x["flow"])):
                x["flow"][i] = x["flow"][i] * config["loss"]["flow_scaling"]

            # event-flow association
            loss_function.update(
                x["flow"],
                inputs["event_list"].to(device),
                inputs["event_list_pol_mask"].to(device),
                inputs["d_event_list"].to(device),
                inputs["d_event_list_pol_mask"].to(device),
            )

            # loss computation
            if loss_function.num_passes >= config["data"]["passes_loss"]:
                data.samples += config["loader"]["batch_size"]

                loss = loss_function()
                train_loss += loss.item()
                loss.backward()

                if config["loss"]["clip_grad"] is not None:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])

                optimizer.step()
                optimizer.zero_grad()

                if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                    vis.data["flow"] = x["flow"][-1].clone()

                model.detach_states()
                loss_function.reset()

                with torch.no_grad():
                    if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                        vis.step(inputs)

                if config["vis"]["verbose"]:
                    print(
                        "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}".format(
                            data.epoch,
                            data.seq_num,
                            len(data.files),
                            int(100 * data.seq_num / len(data.files)),
                            train_loss / data.samples,
                        ),
                        end="\r",
                    )

        if end_train:
            break

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_flow.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--path_cache",
        default="",
        help="location of the cache version of the formatted dataset",
    )
    parser.add_argument(
        "--prev_runid",
        default="",
        help="pre-trained model to use as starting point",
    )
    args = parser.parse_args()

    # launch training
    train(args, YAMLParser(args.config))
