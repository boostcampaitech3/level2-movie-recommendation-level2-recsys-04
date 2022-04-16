import os

import pandas as pd
import numpy as np
import torch

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    path = '/opt/ml/input/data/train/numpy/'
    root_dir = '/opt/ml/input/data'
    data_dir = '/opt/ml/input/data/train'
    output_dir = '/opt/ml/input/data/eval/outputs'

    raw_df = pd.read_csv(os.path.join(data_dir, 'train_ratings.csv'))
    seen_df = pd.read_csv(os.path.join(root_dir, 'seen_movie.csv'))

    users = raw_df['user'].unique()

    unique_sid = raw_df['item'].unique()
    unique_uid = raw_df['user'].unique()

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    id2profile = dict(map(reversed, profile2id.items()))
    id2show = dict(map(reversed, show2id.items()))

    seen_dic = seen_df.set_index('user').to_dict()['seen']
    for key in seen_dic.keys():
        seen_dic[key] = eval(seen_dic[key])

    rec_vae = torch.tensor(np.load(path+'seed0_1_3.npy'))
    ease = torch.tensor(np.load(path+'EASE.npy'))

    output = torch.mul(rec_vae, torch.sigmoid(ease)).cpu().detach().numpy()
    # output = torch.add(rec_vae, ease).cpu().detach().numpy()
    submission = pd.read_csv(os.path.join(root_dir, 'eval', 'sample_submission.csv'))

    idx = 0

    idx_out = np.argsort(-output, axis=1)
    pred_dic = dict()

    for i in tqdm(range(len(idx_out))):
        decoded = [id2show[x] for x in idx_out[i]]
        pred_dic[id2profile[i]] = decoded

    print("Inference Start!")
    for user in tqdm(users):
        seen_list = np.array(seen_dic[user])
        tmp_items = np.array(list(pred_dic[user]))
        tmp_items = tmp_items[np.isin(tmp_items, seen_list) == False]
        top_k_items = tmp_items[:10]

        for i in range(10):
            submission.loc[idx + i, 'item'] = top_k_items[i]
        idx += 10

    print("Recommendation End!")

    submission.to_csv(os.path.join(output_dir, f"output_ensemble_rec_nease_mul.csv"), index=False)
