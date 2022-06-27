import argparse
import os
import numpy as np


def retrieve_eval(args):
    eval_id = args.eval_id
    if args.eval_id < 0:
        eval_id = 0
        for file in os.listdir(args.path + args.runid + "/"):
            if file == ".DS_Store":
                continue
            tmp = int(file.split(".")[0].split("_")[-1])
            eval_id = tmp + 1 if tmp + 1 > eval_id else eval_id
        eval_id -= 1
    path_from = args.path + args.runid + "/" + "eval_" + str(eval_id) + "/"
    print("Preparing submission for eval_{0}".format(eval_id))

    return path_from


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid")
    parser.add_argument("--path", default="dsec_submissions/")
    parser.add_argument("--eval_id", default=-1, type=int)
    args = parser.parse_args()

    # retrieve last eval run unless specified
    path_from = retrieve_eval(args)

    # retrieve folders in directory
    entry = "/flow_bw/"
    folders = os.listdir(path_from)
    for folder in folders:
        if folder in [".DS_Store", "submission"]:
            continue

        # retrieve files in folder with png extension
        files = os.listdir(path_from + folder + entry)
        indices = []
        for file in files:
            indices.append(int(file.split(".")[0]))
        indices.sort()

        # fixing pred-gt alignment
        flags = np.load(args.path + folder + "_flag.npy")
        flags = np.roll(flags, -1)

        # select gt maps to be submitted
        flow_timestamp = np.genfromtxt(args.path + folder + ".txt", skip_header=1, delimiter=",")
        flow_filenames = flow_timestamp[:, -1]

        selected_indices = []
        for i in range(len(indices)):
            if flags[i] == 1:
                selected_indices.append(indices[i])

        # create new folder
        if not os.path.exists(path_from + "submission/"):
            os.makedirs(path_from + "submission/")
        if not os.path.exists(path_from + "submission/" + folder + "/"):
            os.makedirs(path_from + "submission/" + folder + "/")

        # copy files to new folder with the right name
        for i in range(len(selected_indices)):
            filename = path_from + "submission/" + folder + "/" + str(int(flow_filenames[i])).zfill(6) + ".png"
            os.system("cp " + path_from + folder + entry + str(selected_indices[i]).zfill(9) + ".png " + filename)

        print(folder)
