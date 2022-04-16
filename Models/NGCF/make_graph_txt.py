import os

import pandas as pd


def make_user_item(train=True):
    trains = ['u1.base', 'u2.base', 'u3.base', 'u4.base', 'u5.base']
    tests = ['u1.test', 'u2.test', 'u3.test', 'u4.test', 'u5.test']
    df = pd.DataFrame()
    path = '/opt/ml/input/data/lens/'
    root = '/opt/ml/input/data/lens/ngcf'

    if train:
        for train in trains:
            tmp = pd.read_csv(os.path.join(path, 'ml-100k', train), sep='\t', encoding='latin-1', header=None)
            df = pd.concat([df, tmp], ignore_index=True)
        df.columns=['user_id', 'movie_id', 'rating', 'timestamp']
        users = df['user_id'].unique()

        if 'train.txt' in os.listdir(root): t = 'w'
        else: t = 'w+'

        with open(os.path.join(root, 'train.txt'), t) as f:
            save_txt = ''
            for user in users:
                save_txt += f'{str(user)} '
                items = ' '.join(map(str, df[df['user_id'] == user]['movie_id'].values))
                save_txt += items
                save_txt += '\n'

            f.write(save_txt)
    else:
        for test in tests:
            tmp = pd.read_csv(os.path.join(path, 'ml-100k', test), sep='\t', encoding='latin-1', header=None)
            df = pd.concat([df, tmp], ignore_index=True)

        df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        users = df['user_id'].unique()

        if 'test.txt' in os.listdir(root): t = 'w'
        else: t = 'w+'

        with open(os.path.join(root, 'test.txt'), t) as f:
            save_txt = ''
            for user in users:
                save_txt += f'{str(user)} '
                items = ' '.join(map(str, df[df['user_id'] == user]['movie_id'].values))
                save_txt += items
                save_txt += '\n'

            f.write(save_txt)


if __name__ == '__main__':
    make_user_item(True)
    make_user_item(False)