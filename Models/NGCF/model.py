import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


class NGCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, layers, regularization, node_dropout, msg_dropout, adj_matrix, device):
        super(NGCF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.l_matrix = adj_matrix
        self.l_plus_i_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
        self.regularization = regularization
        self.layers = layers
        self.n_layers = len(self.layers)
        self.node_dropout = node_dropout
        self.msg_dropout = msg_dropout
        self.device = device

        self.weight_dict = self._init_weights()

        # Create torch sparse tensor
        self.L_plus_I = self._convert_sp_matrix_to_sp_tensor(self.l_plus_i_matrix)
        self.L = self._convert_sp_matrix_to_sp_tensor(self.l_matrix)

    def _init_weights(self):
        weight_dict = nn.ParameterDict()

        initializer = torch.nn.init.xavier_uniform_

        weight_dict['user_embedding'] = nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim).to(self.device)))
        weight_dict['item_embedding'] = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim).to(self.device)))

        weight_size_list = [self.emb_dim] + self.layers

        for l in range(self.n_layers):
            weight_dict[f'W_one_{l}'] = nn.Parameter(initializer(torch.empty(weight_size_list[l],
                                                                             weight_size_list[l+1]).to(self.device)))
            weight_dict[f'b_one_{l}'] = nn.Parameter(initializer(torch.empty(1, weight_size_list[l]).to(self.device)))

            weight_dict[f'W_two_{l}'] = nn.Parameter(initializer(torch.empty(weight_size_list[l],
                                                                             weight_size_list[l+1]).to(self.device)))
            weight_dict[f'b_two_{l}'] = nn.Parameter(initializer(torch.empty(1, weight_size_list[l+1]).to(self.device)))

        return weight_dict

    def _convert_sp_matrix_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix

        :param X: (scipy sparse matrix) Adjacency matrix
        :return: (torch.sparse.FloatTensor) Adjacency matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)

        return res

    def _dropout_sparse(self, X):
        """
        Drop individual locations in X

        :param X: (torch.sparse.FloatTensor) adjacency matrix
        :return: (torch.sparse.FloatTensor) dropout matrix
        """
        node_dropout_mask = (self.node_dropout + torch.rand(X._nnz())).floor().bool().to(self.device)
        i = X.coalesce().indices()
        v = X.coalesce()._values()
        i[:, node_dropout_mask] = 0
        v[node_dropout_mask] = 0
        X_dropout = torch.sparse.FloatTensor(i, v, X.shape).to(X.device)

        return X_dropout.mul(1/(1-self.node_dropout))

    def forward(self, u, i, j):
        """
        Forward pass

        :param u: user
        :param i: positive interaction
        :param j: negative interaction
        :return: BPR Loss
        """

        # apply drop out mask
        L_plus_I_hat = self._dropout_sparse(self.L_plus_I) if self.node_dropout > 0 else self.L_plus_I
        L_hat = self._dropout_sparse(self.L) if self.node_dropout > 0 else self.L

        # ego embedding = [(user embeddings);(item_embeddings)] (E)
        ego_embeddings = torch.cat([self.weight_dict['user_embedding'], self.weight_dict['item_embedding']], dim=0)

        # layer embeddings ( E^(l) )
        final_embeddings = [ego_embeddings]

        # forward pass l-th propagation layers
        for l in range(self.n_layers):
            # (L+I)E
            left_L_plus_I_embeddings = torch.sparse.mm(L_plus_I_hat, ego_embeddings)

            # (L+I)EW_1 + b_1
            simple_embeddings = torch.matmul(left_L_plus_I_embeddings, self.weight_dict[f'W_one_{l}']) + self.weight_dict[f'b_one_{l}']

            # LE
            right_LE_embeddings = torch.matmul(L_hat, ego_embeddings)

            # LEE
            interaction_embeddings = torch.mul(right_LE_embeddings, ego_embeddings)

            # LEEW_2 + b_2
            interaction_embeddings = torch.matmul(interaction_embeddings, self.weight_dict[f'W_two_{l}']) + self.weight_dict[f'b_two_{l}']

            # LeakyReLU
            ego_embeddings = F.leaky_relu(simple_embeddings + interaction_embeddings)

            # message dropout
            msg_dropout_mask = nn.Dropout(self.msg_dropout)
            ego_embeddings = msg_dropout_mask(ego_embeddings)

            # L2 normalization
            l2_norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            final_embeddings.append(l2_norm_embeddings)

        final_embeddings = torch.cat(final_embeddings, dim=1)

        # split user embeddings, item embeddings
        user_final_embeddings, item_final_embeddings = final_embeddings.split([self.n_users, self.n_items], dim=0)

        self.user_final_embeddings = nn.Parameter(user_final_embeddings)
        self.item_final_embeddings = nn.Parameter(item_final_embeddings)

        u_emb = user_final_embeddings[u]
        p_emb = item_final_embeddings[i]
        n_emb = item_final_embeddings[j]

        # e_u^Te_i
        y_ui = torch.mul(u_emb, p_emb).sum(1)
        y_uj = torch.mul(u_emb, n_emb).sum(1)

        # loss compute
        bpr_loss = -(torch.log(torch.sigmoid(y_ui - y_uj))).mean()

        if self.regularization > 0:
            l2_norm = (torch.sum(u_emb**2)/2. + torch.sum(p_emb**2)/2. + torch.sum(n_emb**2)/2.) / u_emb.shape[0]
            l2_regrularization = self.regularization * l2_norm
            bpr_loss += l2_regrularization

        return bpr_loss
