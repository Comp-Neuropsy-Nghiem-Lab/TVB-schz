import os
import argparse
import importlib
import time
import warnings

import numpy as np

from utils.tools import expand_path, parse_config

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, choices=["ei_tuning", "classification"])
parser.add_argument("--exp", type=str, required=True, choices=["normal", "shuffle_region", "shuffle_subj", "shuffle_graph"])
parser.add_argument("--classifier", type=str, default=None, choices=["SGD", "RF"])
parser.add_argument("--weight_type", type=str, default=None)
parser.add_argument("--parcellation", type=int, default=None)
parser.add_argument("--subject", type=str, default=None)
parser.add_argument("--extra", nargs="*", default=[])


def main():
    start = time.time()
    args = parser.parse_args()
    config = parse_config(os.path.join(os.getcwd(), 'configs', f'{args.task}.yaml'))
    config["exp_name"] = args.exp
    config["weight_type"] = args.weight_type
    config["parcellation"] = args.parcellation
    config["subject"] = args.subject
    
    if args.extra == []:
        pass
    else:
        for item in ",".join(args.extra).split(","):
            key, value = item.split("=", 1)
            config[key] = value

    if args.task == "ei_tuning":
        module = importlib.import_module("tasks.ei_tuning")
        exp_class = getattr(module, config["exp_name"])
        task = exp_class(config)
        task.run()
    elif args.task == "classification":
        if args.classifier is None:
            raise ValueError("--classifier is required for --task classification (choices: SGD, RF)")
        if args.exp == "shuffle_graph":
            raise ValueError("classification does not support --exp shuffle_graph")

        module = importlib.import_module("tasks.classification")
        exp_class = getattr(module, args.classifier)
        config["result_exp"] = args.exp
        task = exp_class(config)
        results = task.run()

        save_dir = expand_path(config["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / f"{args.classifier}_{args.exp}.npy"
        np.save(str(out_file), results)
        print(f"Saved -> {out_file}")

    print(f"Total time taken: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
