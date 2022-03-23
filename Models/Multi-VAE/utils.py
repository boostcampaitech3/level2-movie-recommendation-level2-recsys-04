import os
import argparse
import random
from pprint import pprint

import torch
import numpy as np
import pandas as pd

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
    parser = argparse.ArgumentParser()

    load_dotenv(verbose=True)

    parser.add_argument("--batch_size", type=int, default=500, help="batch size (default: 500)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate (default: 1e-4)")
    parser.add_argument("--heldout", type=int, default=1000, help="n_heldout_user (default: 1000)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="validation ratio (default: 0.2)")

    parser.add_argument("--decay", type=float, default=0.00, help="wright decay coefficient")
    parser.add_argument("--step_size", type=int, default=200000,
                        help="Total number of gradient updates for annealing")
    parser.add_argument("--anneal_cap", type=float, default=0.2, help="Annealing parameter max")

    parser.add_argument("--log_interval", type=int, default=20, metavar="N", help="Report interval")
    parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
    parser.add_argument('--name', type=str, default='model', help='experiment name')
    parser.add_argument('--output_name', type=str, default='output', help='path to save the final model')
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    # SM_CHANNEL_TRAIN = / opt / ml / input / data / train
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"),
                        help="Dataset location")
    parser.add_argument("--root_dir", type=str, default=os.environ.get("SM_CHANNEL_ROOT"),
                        help="Dataset location")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR"))
    # SM_MODEL_DIR =./ model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args = parser.parse_args()

    pprint(vars(args))

    return args


def get_count(df, id):
    playcount_groupbyid = df[[id]].groupby(id, as_index=True)
    count = playcount_groupbyid.size()

    return count


def filter_triplets(df, min_uc=5, min_ic=0):
    if min_ic > 0:
        item_count = get_count(df, 'item')
        df = df[df['item'].isin(item_count.index[item_count >= min_ic])]

    if min_uc > 0:
        user_count = get_count(df, 'user')
        df = df[df['user'].isin(user_count.index[user_count >= min_uc])]

    user_count, item_count = get_count(df, 'user'), get_count(df, 'item')

    return df, user_count, item_count


def split_train_test_proportion(data, prob=0.2):
    group_by_user = data.groupby('user')
    train_list, test_list = list(), list()

    for _, group in group_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            idx[np.random.choice(n_items_u, size=int(prob * n_items_u), replace=False).astype(np.int64)] = True

            train_list.append(group[np.logical_not(idx)])
            test_list.append(group[idx])
        else:
            train_list.append(group)

    data_train = pd.concat(train_list)
    data_test = pd.concat(test_list)

    return data_train, data_test


def numerize(df, profile2id, item2id):
    uid = df['user'].apply(lambda x: profile2id[x])
    iid = df['item'].apply(lambda x: item2id[x])

    return pd.DataFrame(data={'uid': uid, 'iid': iid}, columns=['uid', 'iid'])

def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())
