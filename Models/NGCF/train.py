import os

import torch
import torch.optim as optim
import pandas as pd

from utils import *
from model import NGCF
from dataset import GraphData
from metrics import *


def main(args):

    data_loader = GraphData(path=args.data_dir, batch_size=args.batch_size)
    adj_matrix = data_loader.get_adj_matrix()

    model = NGCF(n_users=data_loader.n_users, n_items=data_loader.n_items,
                 emb_dim=args.emb_dim, layers=args.layers, regularization=args.reg,
                 node_dropout=args.node_dropout, msg_dropout=args.msg_dropout,
                 adj_matrix=adj_matrix, device=args.device)

    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Start Training...")

    # training part
    best_recall = -np.inf
    k = 10
    for epoch in range(1, args.epochs+1):
        model.train()

        n_batch = data_loader.n_train // data_loader.batch_size + 1
        running_loss = 0

        for idx in range(1, n_batch + 1):
            u, i, j = data_loader.sample()
            optimizer.zero_grad()
            loss = model(u, i, j)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if idx % args.log_interval == 0:
                print(f"| epoch {epoch:3} | {idx:4d}/{n_batch:4d} batches | "
                      f"loss {running_loss / args.log_interval:4.2f}")
                running_loss = 0

        # validation part
        with torch.no_grad():
            u_emb = model.user_final_embeddings.detach()
            i_emb = model.item_final_embeddings.detach()

            u_splits = split_matrix(u_emb)
            tr_splits = split_matrix(data_loader.R_train)
            te_splits = split_matrix(data_loader.R_test)

            recall_10, recall_20, recall_50 = [], [], []
            ndcg_k = []

            for u_f, tr_f, te_f in zip(u_splits, tr_splits, te_splits):
                scores = torch.mm(u_f, i_emb.T)

                test_items = torch.from_numpy(te_f.todense()).float().to(args.device)
                non_train_items = torch.from_numpy(1-(tr_f.todense())).float().to(args.device)
                scores = scores * non_train_items

                _, test_indices = torch.topk(scores, dim=1, k=100)
                _, test_indices10 = torch.topk(scores, dim=1, k=10)
                _, test_indices20 = torch.topk(scores, dim=1, k=20)
                _, test_indices50 = torch.topk(scores, dim=1, k=50)

                pred_items = torch.zeros_like(scores).float()
                pred_items.scatter_(dim=1, index=test_indices,
                                    src=torch.ones_like(test_indices).float().to(args.device))

                topk_preds = torch.zeros_like(scores).float()
                topk_preds.scatter_(dim=1, index=test_indices,
                                    src=torch.ones_like(test_indices).float().to(args.device))

                top10_preds = torch.zeros_like(scores).float()
                top10_preds.scatter_(dim=1, index=test_indices10[:, :10], src=torch.ones_like(test_indices10).float())
                top20_preds = torch.zeros_like(scores).float()
                top20_preds.scatter_(dim=1, index=test_indices20[:, :20], src=torch.ones_like(test_indices20).float())
                top50_preds = torch.zeros_like(scores).float()
                top50_preds.scatter_(dim=1, index=test_indices50[:, :50], src=torch.ones_like(test_indices50).float())

                TP10 = (test_items * top10_preds).sum(1)
                TP20 = (test_items * top20_preds).sum(1)
                TP50 = (test_items * top50_preds).sum(1)

                rec10 = TP10/test_indices10.sum(1)
                rec20 = TP20/test_indices20.sum(1)
                rec50 = TP50/test_indices50.sum(1)

                ndcg = NDCG_at_k(pred_items, test_items, test_indices, 100)

                recall_10.append(rec10)
                recall_20.append(rec20)
                recall_50.append(rec50)
                ndcg_k.append(ndcg)

            recall10 = torch.cat(recall_10).mean()
            recall20 = torch.cat(recall_20).mean()
            recall50 = torch.cat(recall_50).mean()
            ndcgk = torch.cat(ndcg_k).mean()

        print_str = f"| end of epoch {epoch:3d} | n100 {ndcgk:5.3f} | " \
                    f"r10 {recall10:5.3f} | r20 {recall20:5.3f} | r50 {recall50:5.3f}"

        if recall10 > best_recall:
            with open(os.path.join(args.model_dir, args.save), 'wb') as f:
                torch.save(model, f)
            best_recall = recall10
            best_n100 = ndcgk
            print_str += " ---> best model save!"

        print("-" * 78)
        print(print_str)
        print("-" * 78)


if __name__ == '__main__':
    args = args_getter()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Use device : {args.device}")

    main(args)