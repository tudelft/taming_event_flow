import os

import mlflow
import numpy as np
import pandas as pd
import torch


def load_model(prev_runid, model, device, curr_run=None, tb_writer=None):
    try:
        run = mlflow.get_run(prev_runid)
    except:
        return model, 0

    model_dir = run.info.artifact_uri + "/model/data/model.pth"
    if model_dir[:7] == "file://":
        model_dir = model_dir[7:]

    starting_epoch = 0
    if os.path.isfile(model_dir):
        model_loaded = torch.load(model_dir, map_location=device).state_dict()

        # check for input-dependent layers
        for key in model_loaded.keys():
            if key.split(".")[1] == "pooling" and key.split(".")[-1] in ["weight", "weight_f"]:
                model.encoder_unet.pooling = model.encoder_unet.build_pooling(model_loaded[key].shape).to(device)
                model.encoder_unet.get_axonal_delays()

        new_params = model.state_dict()
        new_params.update(model_loaded)
        model.load_state_dict(new_params)

        loss_file = run.info.artifact_uri[:-9] + "metrics/loss"
        if os.path.isfile(run.info.artifact_uri[:-9] + "metrics/loss"):
            loss = np.genfromtxt(loss_file)
            if curr_run is not None:
                if not os.path.exists(curr_run.info.artifact_uri[:-9] + "metrics/"):
                    os.makedirs(curr_run.info.artifact_uri[:-9] + "metrics/")
                for i in range(loss.shape[0]):
                    mlflow.log_metric("loss", loss[i, 1], step=int(loss[i, 2]))
                    if tb_writer is not None:
                        tb_writer.add_scalar("loss", loss[i, 1], int(loss[i, 2]))
            starting_epoch = int(loss[-1][-1])

        print("Model restored from " + prev_runid + "\n")
    else:
        print("No model found at " + prev_runid + "\n")

    return model, starting_epoch


def create_model_dir(path_results, runid):
    path_results += runid + "/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results


def save_model(model):
    mlflow.pytorch.log_model(model, "model", conda_env={"dependencies": []})


def save_state_dict(runid, state_dict):
    mlflow.start_run(runid)
    mlflow.pytorch.log_state_dict(state_dict, "state_dict")
    mlflow.end_run()


def load_state_dict(runid, dir="state_dict/", filename="state_dict.pth"):
    run = mlflow.get_run(runid)
    model_dir = run.info.artifact_uri + "/" + dir
    if model_dir[:7] == "file://":
        model_dir = model_dir[7:]

    model_dict = None
    if os.path.isfile(model_dir + filename):
        model_dict = torch.load(model_dir + filename, map_location=torch.device("cpu"))
        print("Model restored from " + runid)
    else:
        print("No model found at " + runid)

    return model_dict


def save_csv(data, fname):
    # create file if not there
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":  # to_csv() doesn't work with 'file://'
        path = path[7:]
    if not os.path.isfile(path):
        mlflow.log_text("", fname)
        pd.DataFrame(data).to_csv(path)
    # else append
    else:
        pd.DataFrame(data).to_csv(path, mode="a", header=False)


def save_diff(fname="git_diff.txt"):
    # .txt to allow showing in mlflow
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":
        path = path[7:]
    mlflow.log_text("", fname)
    os.system(f"git diff > {path}")


def binary_search_array(array, x, left=None, right=None, side="left"):
    left = 0 if left is None else left
    right = len(array) - 1 if right is None else right
    mid = left + (right - left) // 2

    if left > right:
        return left if side == "left" else right

    if array[mid] == x:
        return mid

    if x < array[mid]:
        return binary_search_array(array, x, left=left, right=mid - 1, side=side)

    return binary_search_array(array, x, left=mid + 1, right=right, side=side)


def initialize_quant_results(results, filename, metrics):
    if filename not in results.keys():
        results[filename] = {}
    for metric in metrics:
        if metric not in results[filename].keys():
            results[filename][metric] = {}
            results[filename][metric]["metric"] = 0
            results[filename][metric]["it"] = 0

    return results
