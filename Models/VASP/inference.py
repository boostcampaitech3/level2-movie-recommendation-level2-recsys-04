import os
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from numba import jit

from tqdm import tqdm
from dotenv import load_dotenv

from scipy.sparse import csr_matrix

from dataset import CustomDataSet
from utils import *
from loss import *


def inference(args):
    with open(os.path.join(args.model_dir, args.model+"_"+args.save), 'rb') as f:
        print(f"Model {args.model+'_'+args.save} Loading...")
        model = torch.load(f)

    path = os.path.join(args.data_dir, 'train_ratings.csv')
    raw_data = pd.read_csv(path, header=0)

    pro_dir = os.path.join(args.data_dir, 'pro_sg')

    print("Load Mapping Files...")
    with open(os.path.join(pro_dir, 'item2id.pkl'), 'rb') as f:
        item2id = pickle.load(f)

    with open(os.path.join(pro_dir, 'user2id.pkl'), 'rb') as f:
        user2id = pickle.load(f)

    unique_iid = list()
    with open(os.path.join(pro_dir, 'unique_iid.txt'), 'r') as f:
        for line in f:
            unique_iid.append(line.strip())

    raw = numerize(raw_data, user2id, item2id)
    n_users = raw['uid'].max() + 1
    # n_items = len(raw_data['item'].unique())
    n_items = len(unique_iid)

    rows, cols = raw['uid'], raw['iid']
    data = csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.float64,
                      shape=(n_users, n_items))
    data_tensor = naive_sparse2tensor(data).to(args.device)

    output, mu, logvar = model(data_tensor)
    output = output.cpu().detach().numpy()
    idx = np.argsort(-output, axis=1)  # matrix that ordered recommendation item per user

    id2user = dict(map(reversed, user2id.items()))
    id2item = dict(map(reversed, item2id.items()))

    seen_df = pd.read_csv(os.path.join(args.root_dir, 'seen_movie.csv'))
    seen_dic = seen_df.set_index('user').to_dict()['seen']
    for key in seen_dic.keys():
        seen_dic[key] = eval(seen_dic[key])

    pred_dic = {}

    for i in tqdm(range(len(idx))):
        decoded = [id2item[x] for x in idx[i]]
        pred_dic[id2user[i]] = decoded

    submission = pd.read_csv(os.path.join(args.root_dir, 'eval', 'sample_submission.csv'))

    print("Recommendation Start")
    # 10 recommend except seen movie
    user_unique = raw_data['user'].unique()
    index = 0
    v, d = 0, 0
    for user in tqdm(user_unique):
        seen_list = np.array(seen_dic[user])
        temp_items = np.array(list(pred_dic[user]))

        temp_items = temp_items[np.isin(temp_items, seen_list) == False]
        top_k_items = temp_items[:10]

        for i in range(10):
            submission.loc[index + i, 'item'] = top_k_items[i]
        index += 10

    print("Recommendation End!")

    if args.model != 'both':
        np.save('/opt/ml/input/data/train/numpy/FLVAE.npy', output)

    submission.to_csv(os.path.join(args.output_dir, f"{args.output_name}_vasp.csv"), index=False)


if __name__ == '__main__':
    args = get_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inference(args)