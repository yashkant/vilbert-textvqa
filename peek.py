import torch

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="config yaml file")
    arguments = parser.parse_args()
    return arguments

args = parse_args()
checkpoint = torch.load(args.ckpt, map_location="cpu")
for keyword in ["epoch_id", "global_step", "rank", "score", "acc"]:
    for key in checkpoint.keys():
        if keyword in key:
            print(f"{key}: {checkpoint[key]}")

import pdb
pdb.set_trace()


