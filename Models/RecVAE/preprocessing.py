import os
import argparse

import pandas as pd
import numpy as np

import pickle

from dotenv import load_dotenv


def get_count(df, id):
    playcount_groupby_id = df[[id]].groupby(id, as_index=True)
    count = playcount_groupby_id.size()
    return count


def filter_triplets(df, min_uc=5, min_ic=0):
    if min_ic > 0:
        item_count = get_count(df, 'item')
        df = df[df['item'].isin(item_count.index[item_count >= min_ic])]

    if min_uc > 0:
        user_count = get_count(df, 'user')
        df = df[df['user'].isin(user_count.index[user_count >= min_uc])]

    user_count, item_count = get_count(df, 'user'), get_count(df, 'item')

    return df, user_count, item_count


def split_train_test_proportion(data, prob=0.2):
    group_by_user = data.groupby('user')
    train_list, test_list = list(), list()

    for _, group in group_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            idx[np.random.choice(n_items_u, size=int(prob * n_items_u), replace=False).astype(np.int64)] = True

            train_list.append(group[np.logical_not(idx)])
            test_list.append(group[idx])
        else:
            train_list.append(group)

    data_train = pd.concat(train_list)
    data_test = pd.concat(test_list)

    return data_train, data_test


def numerize(df, profile2id, item2id):
    uid = df['user'].apply(lambda x: profile2id[x])
    iid = df['item'].apply(lambda x: item2id[x])

    return pd.DataFrame(data={'uid': uid, 'iid': iid}, columns=['uid', 'iid'])


def main(args):
    raw_data = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)

    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

    # Shuffle User Indices
    unique_uid = user_activity.index
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    n_heldout_users = args.heldout_users
    # Split train/valid
    train_users = unique_uid[:(n_users - n_heldout_users)]
    valid_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['user'].isin(train_users)]
    unique_iid = train_plays['item'].unique()

    item2id = dict((iid, i) for i, iid in enumerate(unique_iid))
    user2id = dict((uid, i) for i, uid in enumerate(unique_uid))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'item2id.pkl'), 'wb') as f:
        pickle.dump(item2id, f)

    with open(os.path.join(args.output_dir, 'user2id.pkl'), 'wb') as f:
        pickle.dump(user2id, f)

    with open(os.path.join(args.output_dir, 'unique_iid.txt'), 'w') as f:
        for iid in unique_iid:
            f.write(f"{iid}\n")

    with open(os.path.join(args.output_dir, 'unique_uid.txt'), 'w') as f:
        for uid in unique_uid:
            f.write(f"{uid}\n")

    if args.heldout_users == 0:
        train_data = numerize(train_plays, user2id, item2id)
        train_data.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    else:
        val_plays = raw_data.loc[raw_data['user'].isin(valid_users)]
        val_plays = val_plays.loc[val_plays['item'].isin(unique_iid)]
        val_plays_tr, val_plays_te = split_train_test_proportion(val_plays, args.val_ratio)

        train_data = numerize(train_plays, user2id, item2id)
        train_data.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)

        val_data_tr = numerize(val_plays_tr, user2id, item2id)
        val_data_tr.to_csv(os.path.join(args.output_dir, 'validation_tr.csv'), index=False)

        val_data_te = numerize(val_plays_te, user2id, item2id)
        val_data_te.to_csv(os.path.join(args.output_dir, 'validation_te.csv'), index=False)

    print("Preprocessing Success!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAE type models' Preprocessing Arguments")

    load_dotenv(verbose=True)

    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"),
                        help="Preprocessing data directory")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_CHANNEL_PRE_OUT"),
                        help="Preprocessing output data directory")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="validation ratio (default: 0.2)")
    parser.add_argument("--min_items_per_user", type=int, default=5)
    parser.add_argument("--min_users_per_item", type=int, default=0)
    parser.add_argument("--heldout_users", type=int, default=0)

    args = parser.parse_args()

    main(args)