import os
import argparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from tqdm import tqdm

from model import EASE

if __name__ == '__main__':
    load_dotenv(verbose=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('--lambda_', type=float, default=0.5)

    parser.add_argument('--root_dir', type=str, default=os.environ.get('SM_CHANNEL_ROOT'))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR"))
    parser.add_argument("--np_output_dir", type=str, default=os.environ.get("SM_NP_OUTPUT_DIR"))
    parser.add_argument('--output_name', type=str, default='output', help='path to save the final model')

    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    users = df['user'].unique()
    items = df['item'].unique()
    n_users = len(users)
    n_items = len(items)

    # data change to mapping
    user2id = dict((uid, i) for i, uid in enumerate(users))
    item2id = dict((iid, i) for i, iid in enumerate(items))
    id2user = dict(zip(user2id.values(), user2id.keys()))
    id2item = dict(zip(item2id.values(), item2id.keys()))

    seen_df = pd.read_csv(os.path.join(args.root_dir, 'seen_movie.csv'))
    seen_dic = seen_df.set_index('user').to_dict()['seen']
    for key in seen_dic.keys():
        seen_dic[key] = eval(seen_dic[key])

    model = EASE(df)
    print("Training Start!")
    model.fit(args.lambda_)

    submission = pd.read_csv(os.path.join(args.root_dir, 'eval', 'sample_submission.csv'))

    idx = 0

    print("Inference Start!")
    for user in tqdm(users):
        seen_list = np.array(seen_dic[user])
        uid = user2id[user]
        pred = model.pred[uid]
        pred = np.argsort(-pred)
        pred = np.array([id2item[item] for item in pred])

        tmp_items = pred[np.isin(pred, seen_list) == False]
        top_k_items = tmp_items[:10]

        for i in range(10):
            submission.loc[idx+i, 'item'] = top_k_items[i]
        idx += 10

    print("Recommendation End!")

    np.save(os.path.join(args.np_output_dir, f'EASE_{args.lambda_}.npy'), model.pred)

    submission.to_csv(os.path.join(args.output_dir, f"{args.output_name}_{args.lambda_}.csv"), index=False)
