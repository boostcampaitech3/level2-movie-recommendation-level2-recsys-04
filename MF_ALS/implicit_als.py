from collections import defaultdict
import implicit
from implicit.als import AlternatingLeastSquares
import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
import ast
from ray import tune

def get_count(tp, id):
    # group by 오브젝트 index=True로 해야 id 가 index값으로 들어감
    playcount_groupbyid = tp[[id]].groupby(id, as_index=True)
    # 이걸하면 sql의 count 와 동일해짐
    count = playcount_groupbyid.size() 
    return count

def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]
    
    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item') 
    return tp, usercount, itemcount

def numerize(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


raw_data = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv', header=0)
raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=0)

unique_uid = raw_data['user'].unique()
unique_sid = raw_data['item'].unique()

profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

raw = numerize(raw_data, profile2id, show2id)
n_users = len(unique_uid)
n_items = len(unique_sid)

rows, cols = raw['uid'], raw['sid']
data = sparse.csr_matrix((np.ones_like(rows),
                            (rows, cols)), dtype='float64',
                            shape=(n_users, n_items))

user_item_matrix = data.toarray() 

id2profile= dict(map(reversed,profile2id.items()))
id2show= dict(map(reversed,show2id.items())) 

def train_als(config):
    model = AlternatingLeastSquares(factors=config['factors'], iterations=config['iterations'], num_threads=0)
    model.fit(data)
    k = 10
    users = unique_uid.repeat(k)
    test_df = pd.DataFrame(users, columns=['user'])
    test_df['item']=0

    index = 0
    for user in tqdm(unique_uid):
        uid = profile2id[user]
        recommendations = model.recommend(uid, data[uid])[0]
        recommendations = np.vectorize(id2show.get)(recommendations)
        for i in range(k):
            test_df.loc[index + i, 'item'] = recommendations[i]
        index += k
        


    diff_path = '/opt/ml/input/melon/phil/EDA/diff_movie4.csv'

    user_unique = test_df['user'].unique()

    k_dic = defaultdict(list)
    for user in tqdm(user_unique):
        for item in test_df[test_df['user']==user]['item']:
            k_dic[user].append(item)

    # 유저별 4개 영화가 담긴 데이터 불러오기
    diff_df = pd.read_csv(diff_path)

    # 딕셔너리 형태로 변경
    diff_dic = diff_df.set_index('user').to_dict()['diff']
    for key in tqdm(diff_dic.keys()):
        diff_dic[key] = ast.literal_eval(diff_dic[key])

    # 유저별 gt와 교집합 딕셔너리
    inter_dic={}
    for user in user_unique:
        inter_dic[user] = set(k_dic[user]).intersection(set(diff_dic[user]))

    # 유저별 gt와 같은 개수 딕셔너리
    correct_dic = {}
    for user in inter_dic.keys():
        correct_dic[user] = len(inter_dic[user])

    recall = sum(correct_dic.values())

    tune.report(recall = recall)