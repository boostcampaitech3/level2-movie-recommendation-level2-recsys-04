import os

from scipy import sparse
import numpy as np
import pandas as pd


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
        data = sparse.csr_matrix((np.ones_like(rows),
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

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                     (rows_tr, cols_tr)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                     (rows_te, cols_te)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te