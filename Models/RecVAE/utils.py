import os
import random
import argparse
from distutils.util import strtobool
from pprint import pprint

from dotenv import load_dotenv

from scipy import sparse
import pandas as pd
import numpy as np
import torch


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def args_getter():
    parser = argparse.ArgumentParser()

    load_dotenv(verbose=True)

    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val_ratio", type=float, default=0)

    parser.add_argument("--enc_epochs", type=int, default=3)
    parser.add_argument("--dec_epochs", type=int, default=1)

    parser.add_argument("--not_alternating", type=lambda x: strtobool(x), default=False)

    parser.add_argument("--hidden_dim", type=int, default=600)
    parser.add_argument("--latent_dim", type=int, default=400)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.005)

    parser.add_argument("--log_interval", type=int, default=20, metavar="N", help="Report interval")
    parser.add_argument("--save", type=str, default="recvae_e110_n0_ldim400")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data_dir", type=str, default=os.environ.get('SM_CHANNEL_PRE_OUT'),
                        help="VAE Preprocessing file directory")
    parser.add_argument("--root_dir", type=str, default=os.environ.get("SM_CHANNEL_ROOT"),
                        help="Dataset location")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR"))

    args = parser.parse_args()

    pprint(vars(args))

    return args


def load_train_data(csv_file, n_items, n_users, global_indexing=False):
    tp = pd.read_csv(csv_file)

    n_users = n_users if global_indexing else tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['iid']
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items, n_users):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def get_data(args):
    unique_iid = list()
    with open(os.path.join(args.data_dir, 'unique_iid.txt'), 'r') as f:
        for line in f:
            unique_iid.append(line.strip())

    unique_uid = list()
    with open(os.path.join(args.data_dir, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())

    n_items = len(unique_iid)
    n_users = len(unique_uid)

    train_data = load_train_data(os.path.join(args.data_dir, 'train.csv'), n_items, n_users)

    if args.val_ratio == 0:
        data = train_data
        data = data.astype('float32')
    else:
        vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(args.data_dir, 'validation_tr.csv'),
                                                   os.path.join(args.data_dir, 'validation_te.csv'),
                                                   n_items, n_users)
        data = train_data, vad_data_tr, vad_data_te
        data = (x.astype('float32') for x in data)

    return data
