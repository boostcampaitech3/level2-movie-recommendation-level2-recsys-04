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
        users = self.user_enc.fit_transform(self.df['user'])
        items = self.item_enc.fit_transform(self.df['item'])
        self.users = np.unique(users)
        self.items = np.unique(items)

        return users, items

    def fit(self, df, lambda_=0.5):
        users, items = self._get_users_and_items()
        values = np.ones(self.df.shape[0])

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
