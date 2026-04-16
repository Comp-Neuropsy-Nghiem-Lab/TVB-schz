import os
import argparse
import importlib
import time
import warnings

from utils.tools import parse_config

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, choices=["ei_tuning", "classification", "visualization"])
parser.add_argument("--exp", type=str, required=True, choices=["normal", "shuffle_region", "shuffle_subj"])
parser.add_argument("--weight_type", type=str, default=None)
parser.add_argument("--parcellation", type=int, default=None)
parser.add_argument("--subject", type=str, default=None)
parser.add_argument("--sparsity", type=float, default=None)
parser.add_argument("--graph_idx", type=int, default=None)


def main():
    start = time.time()
    args = parser.parse_args()

    config = parse_config(os.path.join(os.getcwd(), 'configs', f'{args.task}.yaml'))
    config["exp_name"] = args.exp
    config["weight_type"] = args.weight_type
    config["parcellation"] = args.parcellation
    config["subject"] = args.subject
    config["sparsity"] = args.sparsity
    config["graph_idx"] = args.graph_idx

    if args.task == "ei_tuning":
        module = importlib.import_module("tasks.ei_tuning")
        exp_class = getattr(module, config["exp_name"])
        task = exp_class(config)
        task.run()
    elif args.task == "classification":
        raise NotImplementedError("analysis is not added yet.")
    elif args.task == "visualization":
        raise NotImplementedError("visualization is not added yet.")

    print(f"Total time taken: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
