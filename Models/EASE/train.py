import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.sparse import csr_matrix
from tqdm import tqdm

from utils import *
from model import NEASE


if __name__ == '__main__':

    args = args_getter()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed_everything(args.seed)
    # data load
    df = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))

    users = df['user'].unique()
    items = df['item'].unique()
    n_users = len(users)
    n_items = len(items)

    # data change to mapping
    user2id = dict((uid, i) for i, uid in enumerate(users))
    item2id = dict((iid, i) for i, iid in enumerate(items))
    id2user = dict(zip(user2id.values(), user2id.keys()))
    id2item = dict(zip(item2id.values(), item2id.keys()))

    df['uid'] = df['user'].apply(lambda x: user2id[x])
    df['iid'] = df['item'].apply(lambda x: item2id[x])
    values = [1.0] * len(df)

    # making mapping input matrix
    rows = df['uid']
    cols = df['iid']
    raw_matrix = csr_matrix((values, (rows, cols)), dtype='float32').todense()
    raw_matrix = torch.tensor(raw_matrix, device=args.device)

    model = NEASE(n_items, args.device).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    diag = torch.eye(n_items, dtype=torch.bool, device=args.device)

    # Training
    for epoch in range(1, args.epochs+1):
        model.train()

        optimizer.zero_grad()

        drop_idx = torch.FloatTensor(n_users, n_items).uniform_() > args.dropout
        drop_idx = drop_idx.to(args.device)

        input = raw_matrix * drop_idx

        out_matrix = model(input)
        loss = criterion(out_matrix, raw_matrix)
        loss.backward()

        for param in model.parameters():
            param.grad[diag] *= 0.

        optimizer.step()

        if epoch > args.epochs * 0.8:
            optimizer.param_groups[0]['lr'] *= 0.1

        print(f"[{epoch:3}/{args.epochs:3}] Loss : {loss.item():4.4f}")

    seen_df = pd.read_csv(os.path.join(args.root_dir, 'seen_movie.csv'))
    seen_dic = seen_df.set_index('user').to_dict()['seen']
    for key in seen_dic.keys():
        seen_dic[key] = [item2id[i] for i in eval(seen_dic[key])]

    submission = pd.read_csv(os.path.join(args.root_dir, 'eval', 'sample_submission.csv'))

    # Inference
    idx = 0
    outputs = []
    with torch.no_grad():
        model.eval()
        model._set_diag_zero()
        for uid in tqdm(range(n_users)):
            user = id2user[uid]
            data = raw_matrix[uid].to(args.device)

            seen_list = np.array(seen_dic[user])
            pred = model(data).cpu().detach().numpy()

            outputs.append(pred)

            pred = np.argsort(-pred)
            pred = pred[np.isin(pred, seen_list) == False][:10]
            pred = [id2item[iid] for iid in pred]

            for i in range(10):
                submission.loc[idx + i, 'item'] = pred[i]
            idx += 10
    outputs = np.array(outputs)

    np.save(os.path.join(args.np_output_dir, 'NEASE.npy'), outputs)

    submission.to_csv(os.path.join(args.output_dir, f"{args.output_name}.csv"), index=False)
