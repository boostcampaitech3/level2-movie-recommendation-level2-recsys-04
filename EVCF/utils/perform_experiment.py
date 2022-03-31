from __future__ import print_function

import torch

import math

import time
import pandas as pd
from scipy import sparse
import numpy as np
from tqdm import tqdm
import bottleneck as bn
import ast

device = torch.device("cuda")
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()

    return count

def numerize(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount
# ======================================================================================================================
# def experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, log_dir, model_name='vae'):
def experiment_vae(args, train_loader, model, optimizer, dir, log_dir, model_name='vae'):
    from utils.training import train_vae as train
    from utils.evaluation import evaluate_vae as evaluate

    # 지금 리더보드 상으로 valid, test 안 나누고 통째로 훈련시켜야 좋길래 valid, test 관련 부분 주석처리 해놓았습니다.
    if args.inference == False:
        # SAVING
        torch.save(args, dir + args.model_name + '.config')

        # best_model = model
        best_ndcg = 0.
        e = 0
        last_epoch = 0

        train_loss_history = []
        train_re_history = []
        train_kl_history = []

        # val_loss_history = []
        # val_re_history = []
        # val_kl_history = []

        # val_ndcg_history = []

        time_history = []

        for epoch in range(1, args.epochs + 1):
            time_start = time.time()
            model, train_loss_epoch, train_re_epoch, train_kl_epoch = train(epoch, args, train_loader, model,
                                                                                optimizer)

            # val_loss_epoch, val_re_epoch, val_kl_epoch, val_ndcg_epoch = evaluate(args, model, train_loader, val_loader, epoch, dir, mode='validation')
            time_end = time.time()

            time_elapsed = time_end - time_start

            # appending history
            train_loss_history.append(train_loss_epoch), train_re_history.append(train_re_epoch), train_kl_history.append(
                train_kl_epoch)
            # val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
            #     val_kl_epoch), val_ndcg_history.append(val_ndcg_epoch)
            time_history.append(time_elapsed)

            # printing results
            print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
                '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'.format(epoch, args.epochs, time_elapsed,
                train_loss_epoch, train_re_epoch, train_kl_epoch))
            # print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
            #       '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
            #       'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f}, NDCG: {:.5f})\n'
            #       '--> Early stopping: {}/{} (BEST: {:.5f})\n'.format(
            #     epoch, args.epochs, time_elapsed,
            #     train_loss_epoch, train_re_epoch, train_kl_epoch,
            #     val_loss_epoch, val_re_epoch, val_kl_epoch, val_ndcg_epoch,
            #     e, args.early_stopping_epochs, best_ndcg
            # ))

            # # early-stopping
            # last_epoch = epoch
            # if val_ndcg_epoch > best_ndcg:
            #     e = 0
            #     best_ndcg = val_ndcg_epoch
            #     # best_model = model
            #     print('->model saved<-')
            #     torch.save(model, dir + args.model_name + '.model')
            # else:
            #     e += 1
            #     if epoch < args.warmup:
            #         e = 0
            #     if e > args.early_stopping_epochs:
            #         break

            # # NaN
            # if math.isnan(val_loss_epoch):
            #     break
        torch.save(model, dir + args.model_name + '.model')

    # Infernece
    else: # args.inference == True
        best_model = torch.load(dir + args.model_name + '.model')
        best_model.eval()
        raw_data = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv', header=0)
        unique_sid = pd.unique(raw_data['item'])
        unique_uid = pd.unique(raw_data['user'])
        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
        raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=100)
        raw = numerize(raw_data, profile2id, show2id)
        n_users = raw['uid'].max() + 1
        n_items = len(unique_sid)

        rows, cols = raw['uid'], raw['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                    (rows, cols)), dtype='float64',
                                    shape=(n_users, n_items))

        data_tensor = naive_sparse2tensor(data).to(device)

        output = best_model.reconstruct_x(data_tensor)

        k = 10
        batch_users = output.shape[0]
        output = output.cpu().detach().numpy()
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

        # 전체 학습 데이터
        rating_path = '/opt/ml/input/data/train/train_ratings.csv'
        train_df = pd.read_csv(rating_path)

        # 제출용 빈 데이터프레임 생성
        user_unique = train_df['user'].unique()
        users = user_unique.repeat(10)
        test_df = pd.DataFrame(users, columns=['user'])
        test_df['item']=0

        # 유저별로 본 영화 저장한 csv 불러오기
        seen_path = '/opt/ml/input/EDA/seen_movie.csv'
        seen_df = pd.read_csv(seen_path)

        # 딕셔너리 형태로 변경
        seen_dic = seen_df.set_index('user').to_dict()['seen']
        for key in tqdm(seen_dic.keys()):
            seen_dic[key] = ast.literal_eval(seen_dic[key])

        # # filter 리스트
        # with open('/opt/ml/input/EDA/power_filter.pkl','rb') as f:
        #     power_filter = pickle.load(f)
        # with open('/opt/ml/input/EDA/year_filter.pkl','rb') as f:
        #     year_filter = pickle.load(f)

        # 유저별로 인기 영화에서 본 영화 빼고 10개씩 추천
        index = 0
        for user in tqdm(user_unique):
            temp_items = np.array(list(pred_dic[user]))
            seen_list = np.array(seen_dic[user])
            temp_items = temp_items[np.isin(temp_items, seen_list) == False]
            # temp_items = temp_items[(np.isin(temp_items, seen_list) == False) & ((np.isin(temp_items, power_filter) == True)|(np.isin(temp_items, year_filter) == True))]
            top_k_items = temp_items[:10]
            for i in range(10):
                test_df.loc[index + i, 'item'] = top_k_items[i]
            index += 10

        test_df.to_csv(f'/opt/ml/input/submission/evcf_test.csv', index=False)

    # Final Evaluation
    # test_loss, test_re, test_kl, test_ndcg, \
    # eval_ndcg20, eval_ndcg10, eval_recall50, eval_recall20, \
    # eval_recall10, eval_recall5, eval_recall1 = evaluate(args, best_model, train_loader, test_loader, 9999, dir, mode='test')

    # print("NOTE: " + args.note)
    # print('FINAL EVALUATION ON TEST SET\n'
    #       '- BEST VALIDATION NDCG: {:.5f} ({:} epochs) -\n'
    #       'NDCG@100: {:}  |  Loss: {:.2f}\n'
    #       'NDCG@20: {:}   |  RE: {:.2f}\n'
    #       'NDCG@10: {:}   |  KL: {:.2f}\n'
    #       'Recall@50: {:} |  Recall@5: {:}\n'
    #       'Recall@20: {:} |  Recall@1: {:}\n'
    #       'Recall@10: {:}'.format(
    #     best_ndcg, last_epoch,
    #     test_ndcg, test_loss,
    #     eval_ndcg20, test_re,
    #     eval_ndcg10, test_kl,
    #     eval_recall50, eval_recall5,
    #     eval_recall20, eval_recall1,
    #     eval_recall10
    # ))
    # print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

    # if not args.no_log:
    #     with open(log_dir, 'a') as f:
    #         print(args, file=f)
    #         print("NOTE: " + args.note, file=f)
    #         print('FINAL EVALUATION ON TEST SET\n'
    #               '- BEST VALIDATION NDCG: {:.5f} ({:} epochs) -\n'
    #               'NDCG@100: {:}  |  Loss: {:.2f}\n'
    #               'NDCG@20: {:}   |  RE: {:.2f}\n'
    #               'NDCG@10: {:}   |  KL: {:.2f}\n'
    #               'Recall@50: {:} |  Recall@5: {:}\n'
    #               'Recall@20: {:} |  Recall@1: {:}\n'
    #               'Recall@10: {:}'.format(
    #             best_ndcg, last_epoch,
    #             test_ndcg, test_loss,
    #             eval_ndcg20, test_re,
    #             eval_ndcg10, test_kl,
    #             eval_recall50, eval_recall5,
    #             eval_recall20, eval_recall1,
    #             eval_recall10
    #         ), file=f)
    #         print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

    # # SAVING
    # torch.save(train_loss_history, dir + args.model_name + '.train_loss')
    # torch.save(train_re_history, dir + args.model_name + '.train_re')
    # torch.save(train_kl_history, dir + args.model_name + '.train_kl')
    # torch.save(val_loss_history, dir + args.model_name + '.val_loss')
    # torch.save(val_re_history, dir + args.model_name + '.val_re')
    # torch.save(val_kl_history, dir + args.model_name + '.val_kl')
    # torch.save(val_ndcg_history, dir +args.model_name + '.val_ndcg')
    # torch.save(test_loss, dir + args.model_name + '.test_loss')
    # torch.save(test_re, dir + args.model_name + '.test_re')
    # torch.save(test_kl, dir + args.model_name + '.test_kl')
    # torch.save(test_ndcg, dir +args.model_name + '.test_ndcg')
