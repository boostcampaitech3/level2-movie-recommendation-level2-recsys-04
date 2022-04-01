import torch
from scipy import sparse
import pandas as pd
import numpy as np
from preprocessing import filter_triplets, get_count, numerize
import bottleneck as bn
from tqdm import tqdm
import ast

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def inference(args, device):
    raw_data = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv', header=0)
    raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=0)

    unique_sid = raw_data['item'].unique()
    unique_uid = raw_data['user'].unique()

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    raw = numerize(raw_data, profile2id, show2id)
    n_users = raw['uid'].max() + 1
    n_items = len(unique_sid)

    rows, cols = raw['uid'], raw['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                                (rows, cols)), dtype='float64',
                                shape=(n_users, n_items))

    data_tensor = naive_sparse2tensor(data).to(device)

    with open(f'models/{args.save}.pt', 'rb') as f:
        model = torch.load(f)
    
    model.eval()

    k = 10
    output = model(data_tensor, calculate_loss=False).cpu().detach().numpy()
    idx = bn.argpartition(-output, k, axis=1) # user별로 추천할 itemId가 순서대로 담긴 행렬

    # pred_dic에 user별로 추천 영화 리스트로 넣기
    # idx의 행이 profile2id의 0,1,2... 순서 
    # userId, itemId값 딕셔너리 key, value 순서 0:46936 요렇게 바꿔주기
    id2profile= dict(map(reversed,profile2id.items()))
    id2show= dict(map(reversed,show2id.items())) 
    pred_dic = {}
    for i in tqdm(range(len(idx))):
        decoded = [id2show[x] for x in idx[i]]
        pred_dic[id2profile[i]] = decoded

    # 제출용 빈 데이터프레임 생성
    users = unique_uid.repeat(k)
    test_df = pd.DataFrame(users, columns=['user'])
    test_df['item']=0

    # 유저별로 본 영화 저장한 csv 불러오기
    seen_path = '/opt/ml/input/EDA/seen_movie.csv'
    seen_df = pd.read_csv(seen_path)

    # 딕셔너리 형태로 변경
    seen_dic = seen_df.set_index('user').to_dict()['seen']
    for key in tqdm(seen_dic.keys()):
        seen_dic[key] = ast.literal_eval(seen_dic[key])

    # 유저별로 인기 영화에서 본 영화 빼고 10개씩 추천
    index = 0
    for user in tqdm(unique_uid):
        temp_items = np.array(list(pred_dic[user]))
        seen_list = np.array(seen_dic[user])
        temp_items = temp_items[np.isin(temp_items, seen_list) == False]
        top_k_items = temp_items[:k]
        for i in range(k):
            test_df.loc[index + i, 'item'] = top_k_items[i]
        index += k

    test_df.to_csv(f'/opt/ml/input/submission/{args.save}.csv', index=False)