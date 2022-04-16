import os
import argparse
import pickle

from tqdm import tqdm

import numpy as np
import torch

import pandas as pd
from dotenv import load_dotenv
from scipy.sparse import csr_matrix

from preprocessing import filter_triplets, numerize


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def inference(args, device):
    print("Load Mapping Files...")
    with open(os.path.join(args.data_dir, 'item2id.pkl'), 'rb') as f:
        item2id = pickle.load(f)

    with open(os.path.join(args.data_dir, 'user2id.pkl'), 'rb') as f:
        user2id = pickle.load(f)

    unique_iid = list()
    with open(os.path.join(args.data_dir, 'unique_iid.txt'), 'r') as f:
        for line in f:
            unique_iid.append(line.strip())

    unique_uid = list()
    with open(os.path.join(args.data_dir, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())

    path = os.path.join(args.root_dir, 'train/train.ratings.csv')
    raw_data = pd.read_csv(path)

    raw = numerize(raw_data, user2id, item2id)
    n_users = raw['uid'].max() + 1
    n_items = len(unique_iid)

    rows, cols = raw['uid'], raw['iid']
    data = csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64',
                      shape=(n_users, n_items))
    data_tensor = naive_sparse2tensor(data).to(device)

    with open(f'models/{args.save}.pt', 'rb') as f:
        model = torch.load(f)

    id2user = dict(map(reversed, user2id.items()))
    id2item = dict(map(reversed, item2id.items()))

    model.eval()
    k = 10
    output = model(data_tensor, cal_loss=False).cpu().detach().numpy()
    idx = np.argsort(-output, axis=1)  # user별로 추천할 itemId가 순서대로 담긴 행렬

    print("Now decoding item and user...")

    pred_dic = {}
    for i in tqdm(range(len(idx))):
        decoded = [id2item[x] for x in idx[i]]
        pred_dic[id2user[i]] = decoded

    submission = pd.read_csv(os.path.join(args.root_dir, 'eval', 'sample_submission.csv'))

    # user seen data
    seen_df = pd.read_csv(os.path.join(args.root_dir, 'seen_movie.csv'))
    seen_dic = seen_df.set_index('user').to_dict()['seen']
    for key in seen_dic.keys():
        seen_dic[key] = eval(seen_dic[key])

    # 10 recommend except seen movie
    user_unique = raw_data['user'].unique()
    index = 0
    for user in tqdm(unique_uid):
        temp_items = np.array(list(pred_dic[user]))
        seen_list = np.array(seen_dic[user])
        temp_items = temp_items[np.isin(temp_items, seen_list) == False]
        top_k_items = temp_items[:k]

        for i in range(k):
            submission.loc[index + i, 'item'] = top_k_items[i]
        index += k

    submission.to_csv(os.path.join(args.output_dir, f"{args.output_name}.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    load_dotenv(verbose=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--output_name', type=str, default='output', help='path to save the final model')

    parser.add_argument("--data_dir", type=str, default=os.environ.get('SM_CHANNEL_PRE_OUT'),
                        help="VAE Preprocessing file directory")
    parser.add_argument("--root_dir", type=str, default=os.environ.get("SM_CHANNEL_ROOT"),
                        help="Dataset location")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR"))

    args = parser.parse_args()
