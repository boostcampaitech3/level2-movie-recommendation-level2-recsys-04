import torch
import numpy as np


def NDCG_at_k(pred_items, test_items, test_indices, k, args):
    """
    Compute the NDCG@K

    :param pred_items:
    :param test_items:
    :param test_indices:
    :param k: k-th order
    :return: (float) NDCG@K
    """
    r = (test_items * pred_items).gather(1, test_indices)
    f = torch.from_numpy(np.log2(np.arange(2, k+2))).float().to(args.device)

    dcg = (f[:, :k]/f).sum(1)
    dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k]/f).sum(1)
    ndcg = dcg/dcg_max

    ndcg[torch.isnan(ndcg)] = 0

    return ndcg

