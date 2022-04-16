import torch
import torch.nn as nn

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class EASE:
    def __init__(self, df):
        self.df = df
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self):
        self.users = self.df['user'].unique()
        self.items = self.df['item'].unique()
        n_users = len(self.users)
        n_items = len(self.items)

        user2id = dict((uid, i) for i, uid in enumerate(self.users))
        item2id = dict((iid, i) for i, iid in enumerate(self.items))
        id2user = dict(zip(user2id.values(), user2id.keys()))
        id2item = dict(zip(item2id.values(), item2id.keys()))

        users = np.array([user2id[user] for user in self.df['user']])
        items = np.array([item2id[item] for item in self.df['item']])

        return users, items

    def fit(self, lambda_=0.5):
        users, items = self._get_users_and_items()
        values = np.ones(self.df.shape[0])
        print(users.shape, items.shape, values.shape)

        X = csr_matrix((values, (users, items)))
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)


class NEASE(nn.Module):
    def __init__(self, item_num, device):
        super(NEASE, self).__init__()
        self.encoder = nn.Linear(item_num, item_num, bias=False)

        # constraint diagonal zero
        self.const_eye_zero = torch.ones((item_num, item_num), device=device)
        self.diag = torch.eye(item_num, dtype=torch.bool, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # setting diagonal weight to zero
        self._set_diag_zero()

        output = self.encoder(x)

        return output

    def _set_diag_zero(self):
        self.encoder.weight.data[self.diag] = 0.
