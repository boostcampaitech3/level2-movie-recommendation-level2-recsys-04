import os
import argparse
from distutils.util import strtobool

import pandas as pd
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


def get_count(df, id):
    playcount_groupbyid = df[[id]].groupby(id, as_index=True)
    count = playcount_groupbyid.size()

    return count


def get_args():
    load_dotenv(verbose=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="both")

    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--n_heldout", type=int, default=0, help="n_heldout_user (default: 0)")

    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--latent_dim", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=4096)

    parser.add_argument("--inference", type=lambda x: strtobool(x), default=False)

    parser.add_argument("--log_interval", type=int, default=20, metavar="N", help="Report interval")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument('--root_dir', type=str, default=os.environ.get('SM_CHANNEL_ROOT'))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR"))
    parser.add_argument("--np_output_dir", type=str, default=os.environ.get("SM_NP_OUTPUT_DIR"))
    parser.add_argument('--output_name', type=str, default='output_vasp', help='path to save the final model')
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument('--save', type=str, default='vasp.pt', help='path to save the final model')

    args = parser.parse_args()

    return args


def filter_triplets(df, min_uc=5, min_ic=0):
    if min_ic > 0:
        item_count = get_count(df, 'item')
        df = df[df['item'].isin(item_count.index[item_count >= min_ic])]

    if min_uc > 0:
        user_count = get_count(df, 'user')
        df = df[df['user'].isin(user_count.index[user_count >= min_uc])]

    user_count, item_count = get_count(df, 'user'), get_count(df, 'item')

    return df, user_count, item_count


def numerize(df, profile2id, item2id):
    uid = df['user'].apply(lambda x: profile2id[x])
    iid = df['item'].apply(lambda x: item2id[x])

    return pd.DataFrame(data={'uid': uid, 'iid': iid}, columns=['uid', 'iid'])


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())