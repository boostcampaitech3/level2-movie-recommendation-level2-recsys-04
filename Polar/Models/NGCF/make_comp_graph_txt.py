import os
import random

from tqdm import tqdm

import pandas as pd

from utils import *


def make_user_item(args):
    seed_everything(args.seed)

    path = args.data_dir
    file = os.path.join(path, 'train_ratings.csv')
    log_df = pd.read_csv(file)
    log_user = log_df.groupby('user')['item'].apply(list)

    train_file = os.path.join(path, 'train.txt')
    test_file = os.path.join(path, 'test.txt')

    with open(train_file, 'w')  as tr, open(test_file, 'w') as te:
        train_txt = ""
        test_txt = ""

        for user, items in tqdm(zip(log_user.index, log_user)):
            tr_idx = int(len(items) * (1-args.split_ratio))
            train_txt += f"{str(user)} "
            test_txt += f"{str(user)} "
            random.shuffle(items)
            tr_items = ' '.join(map(str, items[:tr_idx]))
            te_items = ' '.join(map(str, items[tr_idx:]))
            train_txt += (tr_items + "\n")
            test_txt += (te_items + "\n")

        tr.write(train_txt)
        te.write(test_txt)

if __name__ == '__main__':
    args = args_getter()

    make_user_item(args)
