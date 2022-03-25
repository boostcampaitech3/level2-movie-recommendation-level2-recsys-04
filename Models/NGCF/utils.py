import os
import argparse
import random

from distutils.util import strtobool
from pprint import pprint

import torch
import numpy as np
from dotenv import load_dotenv


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def split_matrix(X, n_splits=100):
    """
    Split a matrix/Tensor into n_folds (for the user embeddings and the R matrices)

    :param X: matrix to be split
    :param n_splits: number of splits
    :return: (List) split matrices
    """
    splits = []
    chunk_size = X.shape[0] // n_splits

    for i in range(n_splits):
        start = i * chunk_size
        end = X.shape[0]if i == n_splits-1 else (i+1) * chunk_size
        splits.append(X[start:end])

    return splits


def args_getter():
    parser = argparse.ArgumentParser()

    load_dotenv(verbose=True)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10, help="Num of epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate (default: 1e-4)")

    parser.add_argument("--layers", nargs="+", type=int, default=[64, 64], help="Layer value (default: 64, 64)")
    parser.add_argument("--emb_dim", type=int, default=64, help="Embedding Dimension (default: 64)")
    parser.add_argument("--reg", type=float, default=1e-5, help="L2 Regularization value (default: 1e-5)")
    parser.add_argument("--msg_dropout", type=float, default=0.1, help="Message Dropout ratio (default: 0.1)")
    parser.add_argument("--node_dropout", type=float, default=0.1, help="Node Dropout ratio (default: 0.1)")
    parser.add_argument("--split_ratio", type=float, default=0.1, help="Train / Test split ration (default: 0.1)")

    parser.add_argument("--log_interval", type=int, default=20, metavar="N", help="Report interval")
    parser.add_argument("--data_dir", type=str, default=os.environ.get('SM_CHANNEL_TRAIN'), help="Data directory")
    parser.add_argument("--wandb", type=lambda x: strtobool(x), default=False, help="WandB usage (default: False)")

    args = parser.parse_args()

    pprint(vars(args))

    return args