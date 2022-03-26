import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv

from utils import *


def main():
    load_dotenv(verbose=True)

    data_dir = os.environ.get('SM_CHANNEL_TRAIN')
    root_dir = os.environ.get('SM_CHANNEL_ROOT')
    output_dir = os.environ.get('SM_OUTPUT_DIR')

    ratings_df = pd.read_csv(os.path.join(data_dir, 'train_ratings.csv'))
    sample = pd.read_csv(os.path.join(root_dir, 'eval', 'sample_submission.csv'))

    users = ratings_df['user'].unique()
    n_users = len(users)
    items = ratings_df['item'].unique()
    n_items = len(items)

    item2id = dict((item, i) for i, item in enumerate(items))
    user2id = dict((user, i) for i, user in enumerate(users))
    id2user = dict((v, k) for k, v in user2id.items())
    id2item = dict((v, k) for k, v in item2id.items())

    print("Making pivot matrix")
    ratings_df['uid'] = ratings_df['user'].apply(lambda x: user2id[x])
    ratings_df['iid'] = ratings_df['item'].apply(lambda x: item2id[x])
    ratings_matrix = ratings_df.pivot(index='uid', columns='iid', values='time').fillna(0)
    ratings_matrix[ratings_matrix > 0] = 1

    sub_dict = {'user': [], 'item': []}

    for user in tqdm(range(n_users)):
        init_arr = ratings_matrix.loc[user].values

        item_already = np.array(ratings_matrix.loc[user][ratings_matrix.loc[user].values == 1].index)

        sim_item_arr = np.dot(ratings_matrix, init_arr)
        item_score_arr = np.dot(ratings_matrix.T, sim_item_arr)
        cnd_item_list = np.argsort(item_score_arr)[::-1]
        tmp = cnd_item_list[np.isin(cnd_item_list, item_already) == False][:10]
        rec_item = [id2item[i] for i in tmp]

        uid = [id2user[user]] * 10
        sub_dict['user'].extend(uid)
        sub_dict['item'].extend(rec_item)

    submission = pd.DataFrame(sub_dict)

    print(len(sample), len(submission))

    submission.to_csv(os.path.join(output_dir, 'CF_output.csv'), index=False)


if __name__ == '__main__':
    main()
