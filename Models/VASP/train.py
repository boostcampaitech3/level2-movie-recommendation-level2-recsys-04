import argparse
import pickle
import os
from distutils.util import strtobool
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from dotenv import load_dotenv

from tqdm import tqdm

from model import *
from dataset import DataLoader, CustomDataSet
from utils import *
from inference import *
from loss import *


def make_data(raw_data, args):
    raw_data, user_activity, item_activity = filter_triplets(raw_data, 5, 0)

    # Shuffle User Indices
    unique_uid = user_activity.index
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    n_heldout_users = args.n_heldout
    # Split train/valid
    train_users = unique_uid[:(n_users - n_heldout_users)]

    train_plays = raw_data.loc[raw_data['user'].isin(train_users)]
    unique_iid = train_plays['item'].unique()

    item2id = dict((iid, i) for i, iid in enumerate(unique_iid))
    user2id = dict((uid, i) for i, uid in enumerate(unique_uid))

    pro_dir = os.path.join(args.data_dir, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'item2id.pkl'), 'wb') as f:
        pickle.dump(item2id, f)

    with open(os.path.join(pro_dir, 'user2id.pkl'), 'wb') as f:
        pickle.dump(user2id, f)

    with open(os.path.join(pro_dir, 'unique_iid.txt'), 'w') as f:
        for iid in unique_iid:
            f.write(f"{iid}\n")

    train_data = numerize(train_plays, user2id, item2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    print("Make train valid data done!")


def main(args):
    seed_everything(args.seed)

    raw_data = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    print("Make train valid data")
    make_data(raw_data, args)

    # load data
    loader = DataLoader(args.data_dir)
    n_items = loader.load_n_items()
    train_data = loader.load_data('train')

    N = train_data.shape[0]
    idx_list = list(range(N))

    params = {
        "input_dim": n_items,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "device": args.device
    }

    models = {
        "both": VASP(**params),
        "ease": EASE(params["input_dim"], params["device"]),
        "flvae": FLVAE(**params)
    }

    criterions = {
        "both": loss_function_vasp,
        "ease": CosineLoss().to(args.device),
        "flvae": loss_function_vasp,
    }

    model = models[args.model].to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = criterions[args.model]

    print("Training Start!")
    print("-" * 89)

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        model.train()
        train_loss = 0.0

        KLD = 0.
        BCE = 0.

        np.random.shuffle(idx_list)

        for idx, start_idx in enumerate(range(0, N, args.batch_size)):
            end_idx = min(start_idx + args.batch_size, N)
            data = train_data[idx_list[start_idx:end_idx]]
            data = naive_sparse2tensor(data).to(args.device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss, bce, kld = criterion(recon_batch, data, mu, logvar, args.alpha, args.gamma)
            BCE += bce
            KLD += kld


            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_train_loss += loss.item()


            if idx % args.log_interval == 0 and idx > 0:
                print(f"| epoch {epoch:3} | {idx:4d}/{N//args.batch_size:4d} batches | "
                      f"loss {train_loss/args.log_interval:4.2f}")
                train_loss = 0.0
        print(f"| epoch {epoch:3} | loss {total_train_loss/(N//args.batch_size):4.2f} | KLD: {KLD / (N//args.batch_size):4.2f}")
        print("-" * 89)

        if args.model == "both":
            if epoch == args.epochs+1 - 40:
                optimizer.param_groups[0]['lr'] *= 0.2
            elif epoch == args.epochs+1 - 20:
                optimizer.param_groups[0]['lr'] *= 0.1

    with open(os.path.join(args.model_dir, args.model+"_"+args.save), 'wb') as f:
        torch.save(model, f)

    if args.inference:
        inference(args)


if __name__ == '__main__':
    args = get_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pprint(vars(args))

    print(f"Use device : {args.device}")

    main(args)