import torch
from scipy import sparse
import pandas as pd
import numpy as np
from preprocessing import filter_triplets, get_count, numerize
import bottleneck as bn
from tqdm import tqdm
import ast
import pickle

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def inference(args, device):
    data = sparse.load_npz("raw_matrix.npz")
    data_tensor = naive_sparse2tensor(data).to(device)

    with open(f'models/{args.save}.pt', 'rb') as f:
        model = torch.load(f)
    
    model.eval()

    k = 10
    output = model(data_tensor, calculate_loss=False).cpu().detach().numpy()
    output[data.toarray() > 0] = -np.inf

    idx = bn.argpartition(-output, k, axis=1) # user별로 추천할 itemId가 순서대로 담긴 행렬

    with open('id2profile.pkl','rb') as f:
        id2profile = pickle.load(f)
    with open('id2show.pkl','rb') as f:
        id2show = pickle.load(f)
    with open('unique_uid.pkl','rb') as f:
        unique_uid = pickle.load(f)

   # pred_dic에 user별로 추천 영화 리스트로 넣기
    pred_dic = {}
    for i in range(len(idx)):
        decoded = [id2show[x] for x in idx[i][:k]]
        pred_dic[id2profile[i]] = decoded

    # 제출용 빈 데이터프레임 생성
    users = unique_uid.repeat(k)
    test_df = pd.DataFrame(users, columns=['user'])
    test_df['item']=0

    # 유저별로 인기 영화에서 본 영화 빼고 10개씩 추천
    index = 0
    for user in tqdm(unique_uid):
        top_k_items = np.array(list(pred_dic[user]))
        for i in range(k):
            test_df.loc[index + i, 'item'] = top_k_items[i]
        index += k

    test_df.to_csv(f'/opt/ml/input/submission/{args.save}.csv', index=False)