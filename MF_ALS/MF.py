import pandas as pd
import torch
import numpy as np
import os
from typing import Union, Tuple, List
import seaborn as sns
import scipy
import random
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm
import ast
from collections import defaultdict

path = '/opt/ml/input/data/train/train_ratings.csv'
df = pd.read_csv(path)

def sgd(
    P: np.ndarray,
    Q: np.ndarray,
    b: float,
    b_u: np.ndarray,
    b_i: np.ndarray,
    samples: List[Tuple],
    learning_rate: float,
    regularization: float
) -> None:
    """
    MF 모델의 파라미터를 업데이트하는 SGD를 구현하자.
    
    ***********************************************************************************************************
    SGD:
        모든 학습 데이터에 대하여
        1. 현재 주어진 파라미터로 predicted rating을 구함
        2. 실제 rating과 predicted rating의 차이로 error를 구함
        3. 2.에서 구한 error를 통해 유저와 아이템의 bias 업데이트 - 위 수식의 첫 두 줄
        4. 2.에서 구한 error를 통해 유저와 아이템의 잠재 요인 행렬 업데이트 - 위 수식의 아래 두 줄
    ***********************************************************************************************************
    
    :param P: (np.ndarray) 유저의 잠재 요인 행렬. shape: (유저 수, 잠재 요인 수)
    :param Q: (np.ndarray) 아이템의 잠재 요인 행렬. shape: (아이템 수, 잠재 요인 수)
    :param b: (float) 글로벌 bias
    :param b_u: (np.ndarray) 유저별 bias
    :param b_i: (np.ndarray) 아이템별 bias
    :param samples: (List[Tuple]) 학습 데이터 (실제 평가를 내린 데이터만 학습에 사용함)
                    (user_id, item_id, rating) tuple을 element로 하는 list임
    :param learning_rate: (float) 학습률
    :param regularization: (float) l2 정규화 파라미터
    :return: None
    """
    for user_id, item_id, rating in samples:
        if user_id is None:
            continue
        # 1. 현재 주어진 파라미터로 predicted rating을 구함
        predicted_rating = b + b_u[user_id] + b_i[item_id] + P[user_id, :].dot(Q[item_id, :].T)
        
        # 2. 실제 rating과 predicted rating의 차이로 error를 구함
        error = (rating - predicted_rating)
        
        # 3. 2.에서 구한 error를 통해 유저와 아이템의 bias 업데이트
        b_u[user_id] += learning_rate * (error - regularization * b_u[user_id])
        b_i[item_id] += learning_rate * (error - regularization * b_i[item_id])
        
        # 4. 2.에서 구한 error를 통해 유저와 아이템의 잠재 요인 행렬 업데이트
        P[user_id, :] += learning_rate * (error * Q[item_id, :] - regularization * P[user_id,:])
        Q[item_id, :] += learning_rate * (error * P[user_id, :] - regularization * Q[item_id,:])


def get_predicted_full_matrix(
    P: np.ndarray,
    Q: np.ndarray,
    b: float = None,
    b_u: np.ndarray = None,
    b_i: np.ndarray = None
) -> np.ndarray:
    """
    유저와 아이템의 잠재 요인 행렬과 글로벌, 유저, 아이템 bias를 활용하여 예측된 유저-아이템 rating 매트릭스를 구하라.
    
    :param P: (np.ndarray) 유저의 잠재 요인 행렬. shape: (유저 수, 잠재 요인 수)
    :param Q: (np.ndarray) 아이템의 잠재 요인 행렬. shape: (아이템 수, 잠재 요인 수)
    :param b: (float) 글로벌 bias
    :param b_u: (np.ndarray) 유저별 bias
    :param b_i: (np.ndarray) 아이템별 bias
    :return: (np.ndarray) 예측된 유저-아이템 rating 매트릭스. shape: (유저 수, 아이템 수)
    """
    if b is None:
        return P.dot(Q.T)
    else:
        return b + b_u[:, np.newaxis] + b_i[np.newaxis, :] + P.dot(Q.T)


def get_bce(
    R: np.ndarray,
    predicted_R: np.ndarray
) -> float:
    """
    실제 평가를 내린 데이터 + 샘플링한 negative 데이터에 대한 BCE를 계산하라.
    
    ***********************************************************************************************************
    주의) Negative Sampling 어떻게 할까...
    ***********************************************************************************************************
    
    :param R: (np.ndarray) 유저-아이템 rating 매트릭스. shape: (유저 수, 아이템 수)
    :param predicted_R: (np.ndarray) 예측된 유저-아이템 rating 매트릭스. shape: (유저 수, 아이템 수)
    :return: (float) 전체 학습 데이터에 대한 RMSE
    """
    
    user_index, item_index = np.arange(R.shape[0]), np.arange(R.shape[1]) 
    delta = 1e-7
    error = list()
    for user_id, item_id in zip(user_index, item_index):
        bce_error = -np.sum(R[user_id, item_id] * np.log(predicted_R[user_id, item_id] + delta) +
                            (1 - R[user_id, item_id]) * np.log(1 - predicted_R[user_id, item_id] + delta)) 
        error.append(bce_error)
    bce = np.sqrt(np.asarray(error).mean())
    return bce


class MF(object):
    
    def __init__(self, R, K, learning_rate, regularization, epochs, verbose=False):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.verbose = verbose
        self.samples = list()
        
        self.training_process = list()
    
    def train(self):
        
        # 유저, 아이템 잠재 요인 행렬 초기화
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # 글로벌, 유저, 아이템 bias 초기화
        self.b = np.mean(self.R[np.where(self.R != 0)])
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)

        # 학습 데이터 생성
        for i in range(self.num_users):
            for j in range(self.num_items):
                if self.R[i, j] > 0:
                    self.samples.append((i, j, self.R[i, j]))
                elif random.random() < 0.1:
                    self.samples.append((i, j, self.R[i, j]))
        
        for epoch in tqdm(range(1, self.epochs + 1)):
            np.random.shuffle(self.samples)
            sgd(self.P, self.Q, self.b, self.b_u, self.b_i, self.samples, self.learning_rate, self.regularization)
            predicted_R = self.get_predicted_full_matrix()
            predicted_R = np.clip(predicted_R, 0, 1)
            bce = get_bce(self.R, predicted_R)
            self.training_process.append((epoch, bce))
            if self.verbose and (epoch % 10 == 0):
                print("epoch: %d, error = %.4f" % (epoch, bce))
        
        self.training_process = pd.DataFrame(self.training_process, columns = ['epoch', 'bce'])
    
    def get_predicted_full_matrix(self):
        return get_predicted_full_matrix(self.P, self.Q, self.b, self.b_u, self.b_i)


df['rating'] = np.ones_like(df.user.values)

user_item_matrix = df.pivot_table('rating', 'user', 'item').fillna(0)

# 유저-아이템 rating 매트릭스
R = user_item_matrix.to_numpy()

# 잠재 요인 수
K = 30

# learning rate
learning_rate = 0.001

# l2 정규화 파라미터
regularization = 0.2

# 총 epoch 수
epochs = 50

# 학습 과정의 status print 옵션
verbose = True


if __name__ == '__main__':
    mf = MF(R, K, learning_rate, regularization, epochs, verbose)
    mf.train()

    predicted_user_item_matrix = pd.DataFrame(mf.get_predicted_full_matrix(), columns=user_item_matrix.columns, index=user_item_matrix.index)

    # inference 및 제출 파일 만들기

    # 전체 학습 데이터
    rating_path = '/opt/ml/input/data/train/train_ratings.csv'
    train_df = pd.read_csv(rating_path)

    # 제출용 빈 데이터프레임 생성
    user_unique = train_df['user'].unique()
    users = user_unique.repeat(10)
    test_df = pd.DataFrame(users, columns=['user'])
    test_df['item']=0

    # 유저별로 본 영화 저장한 csv 불러오기
    seen_path = '/opt/ml/input/melon/phil/EDA/seen_movie.csv'
    seen_df = pd.read_csv(seen_path)

    # 딕셔너리 형태로 변경
    seen_dic = seen_df.set_index('user').to_dict()['seen']
    for key in tqdm(seen_dic.keys()):
        seen_dic[key] = ast.literal_eval(seen_dic[key])

    # 유저별로 인기 영화에서 본 영화 빼고 10개씩 추천
    index = 0
    for user in tqdm(user_unique):
        temp_items = np.array(list(predicted_user_item_matrix.loc[user].argsort()[::-1])) 
        seen_list = np.array(seen_dic[user])
        temp_items = temp_items[np.isin(temp_items, seen_list) == False]
        top_k_items = temp_items[:10]
        for i in range(10):
            test_df.loc[index + i, 'item'] = top_k_items[i]
        index += 10

    # 제출 df로 유저별 추천 목록이 담긴 딕셔너리 생성
    # 제출 파일 불러오기

    diff_path = '/opt/ml/input/melon/phil/EDA/diff_movie4.csv'

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

    # gt와 같은 개수 총합
    print(sum(correct_dic.values()))

    # test df 저장
    test_df.to_csv(f'/opt/ml/input/melon/mf_epochs30_latent25.csv', index=False)
