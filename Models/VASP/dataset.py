import os

import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix


class CustomDataSet:
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

    def get_matrix(self):
        users, items = self._get_users_and_items()
        values = np.ones(self.df.shape[0])

        data = csr_matrix((values, (users, items)), dtype='float32').todense()

        return torch.tensor(data)


class DataLoader():
    '''
    Load Movielens dataset
    '''

    def __init__(self, path):

        self.pro_dir = os.path.join(path, 'pro_sg')
        assert os.path.exists(self.pro_dir), "Preprocessed files do not exist. Run data.py"

        self.n_items = self.load_n_items()

    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def load_n_items(self):
        unique_iid = list()
        with open(os.path.join(self.pro_dir, 'unique_iid.txt'), 'r') as f:
            for line in f:
                unique_iid.append(line.strip())
        n_items = len(unique_iid)
        return n_items

    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')

        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['iid']
        data = csr_matrix((np.ones_like(rows),
                            (rows, cols)), dtype='float64',
                            shape=(n_users, self.n_items))
        return data

    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['iid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['iid']

        data_tr = csr_matrix((np.ones_like(rows_tr),
                                (rows_tr, cols_tr)), dtype='float64',
                                shape=(end_idx - start_idx + 1, self.n_items))
        data_te = csr_matrix((np.ones_like(rows_te),
                                (rows_te, cols_te)), dtype='float64',
                                shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te
