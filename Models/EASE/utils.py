import os

import argparse
import random

import numpy as np
import torch
from dotenv import load_dotenv


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def args_getter():
    load_dotenv(verbose=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lambda_', type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument('--root_dir', type=str, default=os.environ.get('SM_CHANNEL_ROOT'))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR"))
    parser.add_argument("--np_output_dir", type=str, default=os.environ.get("SM_NP_OUTPUT_DIR"))
    parser.add_argument('--output_name', type=str, default='output_NEASE', help='path to save the final model')

    args = parser.parse_args()

    return args