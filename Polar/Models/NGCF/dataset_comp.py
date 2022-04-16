import random

import numpy as np
import scipy.sparse as sp


"""
Make dataset to graph datastructure.
Represent the graph by Laplacian Matrix.

In NGCF paper, Laplacian set this:
    L = D^(1/2)AD^(1/2)
    A = | 0     R |
        | R^T   0 |
"""


class GraphData(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + 'test_txt'

        # get user and item counts
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        # get max user_id and item_id
        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    uid = int(line[0])
                    items = [int(i) for i in line[1:]]

                    self.n_users = max(self.n_users, uid)
                    self.n_items = max(self.n_items, max(items))

                    self.n_train += len(items)

        with open(test_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n')
                    try:
                        items = [int(i) for i in line.split(' ')[1:]]
                    except Exception:
                        continue
                    if not items:
                        print("empty test exists")
                        pass
                    else:
                        self.n_items = max(self.n_items, max(items))
                        self.n_test += len(items)

        self.n_items += 1
        self.n_users += 1

        # create interaction/rating matrix R
        print("Creating interaction matrices R_train and R_test")
        self.R_train = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_test = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_items = {}, {}

        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    # if interaction, enter 1
                    for i in train_items:
                        self.R_train[uid, i] = 1.
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    for i in test_items:
                        self.R_test[uid, i] = 1.
                    self.test_items[uid] = test_items

        print("Complete. Interaction matrices R_train, R_test create")

    def create_adj_matrix(self):
        """
        Create the adjacency Matrix and Normalized the matrix

        :return: (scipy.sparse.coo_matrix) adj_matrix
        """
        adj_matrix = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_matrix = adj_matrix.tolil()
        R = self.R_train.tolil()

        adj_matrix[:self.n_users, self.n_users:] = R
        adj_matrix[self.n_users:, :self.n_users] = R.T
        adj_matrix = adj_matrix.todok()

        # normalized adj matrix : D^(1/2)AD^(1/2)
        def normalized_adj_matrix(adj):
            rowsum = np.array(adj.sum(1))                   # same method : get degree matrix
            d_inv = np.power(rowsum -.5).flatten()          # degree inverse matrix
            d_inv[np.isinf(d_inv)] = 0.                     # if 0 or inf -> 0
            d_mat_inv = sp.diags(d_inv)                      # degree matrix making
            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)

            return norm_adj.tocoo()

        print("Normalize the adjacency matrix....")
        ngcf_adj_mat = normalized_adj_matrix(adj_matrix)

        return ngcf_adj_mat

    def get_adj_matrix(self):
        """
        Check save adj matrix.
        If not saved adj matrix, call create_adj_matrix

        :return: (scipy.sparse.coo_matrix) adj_matrix
        """
        try:
            adj_matrix = sp.load_npz(self.path+'/s_adj_matrix.npz')
            print("Find the save file, Now loading the adjacency matrix...")
        except Exception:
            print("Creating adjacency matrix....")
            adj_matrix = self.create_adj_matrix()
            sp.save_npz(self.path+'/s_adj_matrix.npz', adj_matrix)

        return adj_matrix

    def negative_pool(self):
        """
        Create collections of N items that user not interaction

        :return: None
        """
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [random.choice(neg_items) for _ in range(100)]      # Isn't it duplicate items?
            self.neg_pools[u] = pools

    def sample(self):
        """
        Get sample data for mini-batches
        :return: (List, List, List) users, positive items, negative items
        """
        if self.batch_size <= self.n_users:
            users = random.sample(self.exist_users, self.batch_size)
        else:
            users = [random.choice(self.exist_users) for _ in range(self.batch_size)]   # Isn't it duplicate users?

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(0, n_pos_items, 1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)

            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(0, self.n_items, 1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)

            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return random.sample(neg_items, num)

    def get_num_users_items(self):
        return self.n_users, self.n_items
