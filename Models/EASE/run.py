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
    parser.add_argument('--output_name', type=str, default='output', help='path to save the final model')

    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    users = df['user'].unique()

    seen_df = pd.read_csv(os.path.join(args.root_dir, 'seen_movie.csv'))
    seen_dic = seen_df.set_index('user').to_dict()['seen']
    for key in seen_dic.keys():
        seen_dic[key] = eval(seen_dic[key])

    model = EASE(df)
    print("Training Start!")
    model.fit(df, args.lambda_)

    submission = pd.read_csv(os.path.join(args.root_dir, 'eval', 'sample_submission.csv'))

    idx = 0

    print("Inference Start!")
    for user in tqdm(users):
        seen_list = np.array(seen_dic[user])
        user_enc = model.user_enc.transform([user])[0]
        pred = model.pred[user_enc]
        pred = np.argsort(-pred)
        pred = model.item_enc.inverse_transform(pred)

        tmp_items = pred[np.isin(pred, seen_list) == False]
        top_k_items = tmp_items[:10]

        for i in range(10):
            submission.loc[idx+i, 'item'] = top_k_items[i]
        idx += 10

    print("Recommendation End!")

    submission.to_csv(os.path.join(args.output_dir, f"{args.output_name}.csv"), index=False)