# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:22:52 2022

@author: yasub
"""
import sys
import os
import numpy as np
import pandas as pd
import pickle
import dill
import random
import collections
import math
import traceback
import time
import json

import lightgbm as lgb
import optuna
import optuna.integration.lightgbm as lgb_opt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import LabelEncoder
import mysql.connector

from calc_efficient import Set_payment
'''optunaによるチューニングパラメータ：
1. lambda_l1
2. lambda_l2
3. num_leaves
4. feature_fraction
5. bagging_fraction
6. bagging_freq
7. min_child_samples
'''

Rtype = ["shiba","dirt","syogai"]
Track = ["sapporo","hakodate","fukushima","niigata","tokyo","nakayama","tyukyo","kyoto","hanshin","kokura"]
Dtype = ["short","middle","long"]
COLOR = ["r", "g", "b", "c", "m"]

class Constractor():
    def __init__(self, jyo = 1, Nfold = 1, folder = "Datasets_past"):
        self.Nfold = Nfold
        self.folder = folder
        
        if not os.path.exists(f".\\{self.folder}\\0\\"): 
            os.mkdir(f".\\{self.folder}\\0\\")
            
        path = f".\\{self.folder}\\0\\{Track[jyo - 1]}\\"
        if not os.path.exists(path): os.mkdir(path)
            
        self.model_path = path + "\\Model_LightGBM\\"
        self.stan_path = path + "\\Standarize_Result\\"
        #self.prob_path = path + "\\Probability_Result\\"
        self.feature_path = path + "\\Features_LightGBM\\"
        self.valDset_path = path + "\\Valid_DataSet\\"
        
        if not os.path.exists(self.model_path): os.mkdir(self.model_path)
        if not os.path.exists(self.stan_path): os.mkdir(self.stan_path)
        #if not os.path.exists(self.prob_path): os.mkdir(self.prob_path)
        if not os.path.exists(self.feature_path): os.mkdir(self.feature_path)
        if not os.path.exists(self.valDset_path): os.mkdir(self.valDset_path)
        for typ in Rtype:
            if not os.path.exists(f"{self.stan_path}{typ}"): os.mkdir(f"{self.stan_path}{typ}")
            #if not os.path.exists(f"{self.prob_path}{typ}"): os.mkdir(f"{self.prob_path}{typ}")
            
    
        if folder == "Datasets_past":
            from logistic_reglession_ver2 import Past
            SQLclass = Past()
        elif folder == "Datasets_attribute":
            from Race_analysis import Attribution
            SQLclass = Attribution()
        elif folder == "Datasets_basis":
            from basic import Basic
            SQLclass = Basic(jyo)
            
        self.ex_columns =  SQLclass.features# カラムの定義
        self.categorical = SQLclass.categorical
        
        depth_list = [[8,9,6],
                      [8,9,6],
                      [8,9,6],
                      [8,9,6],
                      [9,10,6],
                      [9,10,6],
                      [8,9,6],
                      [8,9,6],
                      [9,10,6],
                      [8,9,6]]
        
        boost_r = [[200, 200, 80],
                   [200, 200, 80],
                   [200, 200, 80],
                   [300, 300, 80],
                   [300, 300, 80],
                   [200, 200, 80],
                   [200, 200, 80],
                   [200, 200, 80],
                   [300, 200, 80],
                   [200, 200, 80]]
        
        self.depth_list = depth_list[jyo - 1]
        self.boost_r = boost_r[jyo - 1]
        
    def TrainTest_Sampling(self, query, X, y1, rid, Nfold):
        
        Wnum = int(len(query) // Nfold) # 交差検証用にデータを分割
        np.random.seed(seed = 100)
        
        Xfold = [[] for i in range(Nfold)]
        yfold = [[] for i in range(Nfold)]
        qfold = [[] for i in range(Nfold)]
        rfold = [[] for i in range(Nfold)]

        rep = np.array([])
        for n in range(Nfold):
            qq = np.in1d(np.arange(len(query)), rep, invert = True)
            qq_in = np.ravel(np.where(qq == True))
            if n == Nfold - 1: index = qq_in
            else: index = np.random.choice(qq_in, size = Wnum, replace = False)
            rep = np.concatenate([rep, index])
            
            qfold[n] = query[index]
            rfold[n] = rid[index]
            
            for q in index:
                if q == len(query) - 1:
                    in1 = np.sum(query[:q])
                    in2 = np.sum(query)
                else:
                    in1 = np.sum(query[:q])
                    in2 = np.sum(query[:q + 1])

                Xfold[n].extend(X[in1 : in2, :])
                yfold[n].extend(y1[in1 : in2])
        
        for n in range(Nfold):
            Xfold[n] = np.array(Xfold[n])
            yfold[n] = np.array(yfold[n])
             
        return Xfold, yfold, qfold, rfold

    def GetQuery_Sampling(self, X, y1, rid, ratio = 0.85, y2 = None, y3 = None, y4 = None, query = None, Nfold = -1):
        
        if self.folder == "Datasets_basis" and y2 is not None: 
            self.ex_columns = X.columns
            X = X.values
        
        if len(self.ex_columns) != len(set(self.ex_columns)):# 重複がないか検索
            counter = dict(collections.Counter(self.ex_columns))
            number = []
            for key, val in counter.items():
                if val > 1:
                    num = np.where(self.ex_columns == key)[0]
                    number.extend(num[1:])
                    
            if len(number):
                self.ex_columns = np.delete(self.ex_columns, number)
                X = np.delete(X, number, 1)
                for num in number:
                    equal = np.where(np.array(self.categorical) == num)[0]
                    large = np.where(np.array(self.categorical) > num)[0]
                    if len(equal): del self.categorical[equal]
                    if len(large):
                        for j in large: self.categorical[j] -= 1
        
        if query is None:
            query = []
            y = np.zeros_like(y1, dtype = y1.dtype)
            cnt = 0
            while True:
                cnt_ = cnt
                if cnt == len(y1): break
                while True:
                    # スコアを重みづけ
                    if y1[cnt]: y[cnt] = 10
                    elif y2[cnt]: y[cnt] = 7
                    elif y3[cnt]: y[cnt] = 3
                    cnt += 1
                    if y3[cnt] == 0: break
                
                while True:
                    if y4[cnt]: y[cnt] = 1
                    cnt += 1
                    if cnt == len(y1): break
                    if y1[cnt]: break
                
                query.append(cnt - cnt_)
        else: y = y1
        
        query = np.array(query, dtype = np.int16)
        if Nfold > 1:
            Xfold, yfold, qfold, rfold = self.TrainTest_Sampling(query, X, y, rid, Nfold)
            X1, y1 = [[] for n in range(Nfold)], [[] for n in range(Nfold)]
            X2, y2, q2, r2 = None, None, None, None
            for n in range(Nfold):
                X1[n] = pd.DataFrame(data = Xfold[n], columns = self.ex_columns)
                #col_float = X1[n].select_dtypes(include = "float")
                #for col in col_float: X1[n][col] = pd.to_numeric(X1[n][col], downcast = "float")
                y1[n] = pd.Series(data = yfold[n], dtype = np.int16)
            q1, r1 = qfold, rfold
            
        else:
            Wnum = int(len(query) * ratio) # テストデータと訓練データに分ける
            np.random.seed(seed = 100)
            train_N = np.random.choice(np.arange(len(query)), size = Wnum, replace = False)
            test_N = np.ravel(np.where(np.in1d(np.arange(len(query)), train_N, invert = True) == True))
            
            Xtrain, ytrain = [], []
            Xtest, ytest = [], []
            train_query, test_query = [], []
            train_rid, test_rid = [], []
            
            for i in train_N:
                if i == len(query) - 1: 
                    Xtrain.extend(X[np.sum(query[:i]) : np.sum(query),:])
                    ytrain.extend(y[np.sum(query[:i]) : np.sum(query)])
                else:
                    Xtrain.extend(X[np.sum(query[:i]) : np.sum(query[:i + 1]),:])
                    ytrain.extend(y[np.sum(query[:i]) : np.sum(query[:i + 1])])
                train_query.append(query[i])
                train_rid.append(rid[i])
                
            for i in test_N:
                if i == len(query) - 1: 
                    Xtest.extend(X[np.sum(query[:i]) : np.sum(query),:])
                    ytest.extend(y[np.sum(query[:i]) : np.sum(query)])
                else:
                    Xtest.extend(X[np.sum(query[:i]) : np.sum(query[:i + 1]),:])
                    ytest.extend(y[np.sum(query[:i]) : np.sum(query[:i + 1])])
                test_query.append(query[i])
                test_rid.append(rid[i])
                
            y1 = pd.Series(data = ytrain, dtype = np.int16)
            q1 = np.array(train_query, dtype = np.int16)
            r1 = np.array(train_rid)
            y2 = pd.Series(data = ytest, dtype = np.int16)
            q2 = np.array(test_query, dtype = np.int16)
            r2 = np.array(test_rid)
            
            try:   
                X1 = pd.DataFrame(data = Xtrain, columns = self.ex_columns)
                X2 = pd.DataFrame(data = Xtest, columns = self.ex_columns)
            except:
                X1 = pd.DataFrame(data = Xtrain)
                X2 = pd.DataFrame(data = Xtest)
                    
                
                '''col_float = X1.select_dtypes(include = "float")
                for col in col_float:
                    X1[col] = pd.to_numeric(X1[col], downcast = "float")
                    if X2 is not None: X2[col] = pd.to_numeric(X2[col], downcast = "float")'''

        return X1, X2, y1, y2, q1, q2, r1, r2
    
    def combine_datasets(self, jyo, num_pid, curs):
        
        YEAR = list(range(2022, 2008, -1))
        if jyo == 8:
            YEAR = np.concatenate([[2023], np.array(list(range(2020, 2011, -1)))])
        Year = [[] for i in range(num_pid)]
        start, end = 0, len(YEAR) // num_pid
        for i in range(num_pid):
            Year[i] = YEAR[start : end]
            start += len(YEAR) // num_pid
            end += len(YEAR) // num_pid
        
        rid = [[] for n in range(3)]
        X = []
        for pid in range(1, 1 + num_pid):
            path = f".\\{self.folder}\\{pid}\\{Track[jyo - 1]}\\"
            if not os.path.exists(path + "dataset.pickle"): continue
        
            with open(path + "dataset.pickle",'rb') as f: 
                x = pickle.load(f)
                for n in range(3):
                    if len(x[n]) == 0: continue
                    x[n] = x[n].loc[:, ~x[n].columns.duplicated()]
                    '''col_float = x[n].select_dtypes(include = "float")
                    for col in col_float:
                        x[n][col] = pd.to_numeric(x[n][col], downcast = "float")'''
            with open(path + "no1.pickle",'rb') as f: ans1 = pickle.load(f)
            with open(path + "no2.pickle",'rb') as f: ans2 = pickle.load(f)
            with open(path + "no3.pickle",'rb') as f: ans3 = pickle.load(f)
            with open(path + "no4.pickle",'rb') as f: ans4 = pickle.load(f)
            if len(X) == 0:
                X = x
                y1 = ans1
                y2 = ans2
                y3 = ans3
                y4 = ans4
            else:
                for n in range(3):
                    if len(x[n]) == 0: continue
                    if self.folder == "Datasets_basis": 
                        X[n] = pd.concat([X[n], x[n]], axis = 0)
                    else:  X[n] = np.vstack([X[n], x[n]])
                    y1[n] = np.vstack([y1[n], ans1[n]])
                    y2[n] = np.vstack([y2[n], ans2[n]])
                    y3[n] = np.vstack([y3[n], ans3[n]])
                    y4[n] = np.vstack([y4[n], ans4[n]])
            for year in Year[pid - 1]:
                curs.execute(f"SHOW TABLES FROM race_result like '{year}{jyo:02d}%'")
                Races = np.ravel(np.array(curs.fetchall()))
                for race in Races:
                    curs.execute(f"SELECT type FROM race_info.`{race}`;")
                    typ = np.ravel(np.array(curs.fetchall()))
                    rid[typ[0]].append(race)
        
        return X, y1, y2, y3, y4, rid

    def Make_ProbModel(self, Xtrain, ytrain, Xvalid, yvalid, rank, typ):  
        params = {'objective':'binary',   # 2値分類
            'boosting_type': 'gbdt',      # GBDTを指定
            'metric' : 'rmse',            # 最適化するロスの定義
            'learning_rate': 0.01,        # 学習率
            'n_estimators': 30000,        # 決定木ノードの最小データ数
            'is_unbalance': 'true',       # 不均衡データなので "true"
            #'min_split_gain': 0.001,      # 分割するときの最小ゲイン
            #'top_rate': 0.4,             # goss用上位誤差を選択割合
            #'other_rate': 0.2,           # goss用下位誤差の選択割合
            #'min_data_per_group': 5,      # EFB用特徴量binningの下限
            'max_depth': 5,               # 深さ
            'num_leaves':21,
            'bagging_fraction':0.8,       # バギング比率
            'bagging_freq': 10,           # バギング頻度
            'max_bin': 100,               # binの数
            'min_data_in_bin': 50,        # binに入る最小データ数
            'subsample_for_bin': 200000}
    
        #lightGBM用にデータセットをインスタンス化
        lgb_train = lgb_opt.Dataset(Xtrain, ytrain)
        lgb_valid = lgb_opt.Dataset(Xvalid, yvalid, reference = lgb_train)
        #optunaでチューニング
        model = lgb_opt.LightGBMTuner(params = params,
                                    train_set = lgb_train,
                                    valid_sets = lgb_valid,
                                    optuna_seed = 100,
                                    early_stopping_rounds = 100,
                                    verbosity = 10)
        model.run()                    
        best_params = model.best_params

        #best parameterで再学習
        model = lgb.LGBMClassifier(**best_params,
                                reg_alpha = best_params["lambda_l1"],
                                reg_lambda = best_params["lambda_l2"],
                                random_state = 100,
                                eval_metric = 'auc',
                                eval_set = [(Xvalid, yvalid)],
                                feature_name = "auto")
        
        model.fit(Xtrain, ytrain)
        
        # モデルを保存
        if not os.path.exists(f"{self.prob_path}{Rtype[typ]}"):
            os.mkdir(f"{self.prob_path}{Rtype[typ]}")
        with open(f'{self.prob_path}{Rtype[typ]}/no{rank + 1}.pickle', 'wb') as f: dill.dump(model, f)
                
    def ReMake2ProbSet(self, model, train_query, Xtrain):
        train_query = np.concatenate([np.array([0], dtype = train_query.dtype), train_query], axis = 0)
        Xt = pd.DataFrame(data = np.full((int(np.sum(train_query)), 19), np.nan))
        
        for len_q in range(len(train_query) - 1):
            in1 = np.sum(train_query[:len_q + 1])
            in2 = np.sum(train_query[:len_q + 2])
            
            val = np.zeros((train_query[len_q + 1]), dtype = np.float32)
            for n in range(len(model)): 
                va = model[n].predict(Xtrain.loc[in1 : in2 - 1, :])
                mean, std = np.mean(va), np.std(va)
                val += np.array(list(map(lambda x, y: x + (y - mean)/std, val, va)))   
            
            val_stan = val / len(model)
            Xt.loc[in1 : in2 - 1, 0] = val_stan
            for kk in range(train_query[len_q + 1]):
                for nn in range(train_query[len_q + 1]):
                    Xt.loc[in1 + kk, 1 + nn] = val_stan[kk] - val_stan[nn]
                Xt.loc[in1 + kk, np.isnan(Xt.loc[in1 + kk, :])] = -1000
                
        return Xt
    
    def PostProcess(self, Xtrain, ytrain, train_query, train_rid, Xtest, ytest, test_query, test_rid, NUM, typ, curs, conn):
        
        f_num = self.Nfold if NUM < 1 else 1
        f_arr = range(1, self.Nfold + 1) if NUM < 1 else [NUM]
        if NUM < 1:# 学習データをfoldから通常に戻す
            for n in range(f_num):
                if n == 0:
                    Xtrain_ = Xtrain[n]
                    ytrain_ = ytrain[n]
                    train_query_ = train_query[n]
                    train_rid_ = train_rid[n]
                else:
                    Xtrain_ = pd.concat([Xtrain_, Xtrain[n]], axis = 0)
                    ytrain_ = pd.concat([ytrain_, ytrain[n]], axis = 0)
                    train_query_ = np.concatenate([train_query_, train_query[n]], axis = 0)
                    train_rid_ = np.concatenate([train_rid_, train_rid[n]], axis = 0)
            Xtrain, ytrain, train_query, train_rid = Xtrain_, ytrain_, train_query_, train_rid_
            Xtrain.reset_index(drop = True, inplace = True)
            ytrain.reset_index(drop = True, inplace = True)
            
        model = [[] for n in range(f_num)]
        for n in range(f_num):
            with open(f"{self.model_path}{f_arr[n]}/{Rtype[typ]}.pickle", "rb") as f:
                model[n] = dill.load(f)
        
        # 標準化し実際の結果との相関を確認 -> 学習データ
        bef = 0
        No = [[] for k in range(5)] # 推奨5頭
        for len_q in range(len(train_query)):
            y = ytrain[bef : bef + train_query[len_q]].values
            
            val = np.zeros_like(y, dtype = np.float32)
            for n in range(f_num): val += model[n].predict(Xtrain[bef : bef + train_query[len_q]])
            val /= f_num
            
            val_mean, val_std = np.mean(val), np.std(val)
            val_stan = np.array(list(map(lambda x: (x - val_mean)/val_std, val)))
                
            index = np.argsort(-val_stan)
            pay, _ = return_payment(train_rid[len_q], curs, conn)
            eff_train = np.zeros((5,2), dtype = np.float32) # 単勝と複勝の回収率
                         
            number = min([5, train_query[len_q]])
            for k in range(number):
                if len(pay):
                    if index[k] == 0:
                        eff_train[k,0] = pay[1]
                        eff_train[k,1] = pay[2]
                    elif index[k] == 1: eff_train[k,1] = pay[3]
                    elif index[k] == 2: eff_train[k,1] = pay[4]
                No[k].append([y[index[k]], val_stan[index[k]], eff_train[k,0], eff_train[k,1]])
        
            if len_q < len(train_query) - 1: bef = np.sum(train_query[:len_q + 1])
            
        # 標準化し実際の結果との相関を確認 -> テストデータ
        bef = 0
        No_test = [[] for k in range(5)] # 推奨5頭
        Xt_prob = []
        yt = [[] for n in range(3)]
        for len_q in range(len(test_query)):
            y = ytest[bef : bef + test_query[len_q]].values
            
            val = np.zeros_like(y, dtype = np.float32)
            for n in range(f_num): val += model[n].predict(Xtest[bef : bef + test_query[len_q]])
            val /= f_num
            
            val_mean = np.mean(val)
            val_std = np.std(val)
            val_stan = np.array(list(map(lambda x: (x - val_mean)/val_std, val)))
            
            index = np.argsort(-val_stan)  
            pay, _ = return_payment(test_rid[len_q], curs, conn)
            eff_test = np.zeros((5,2), dtype = np.float32) # 単勝と複勝の回収率
            
            if NUM < 1:
                xt = Xtest.loc[bef : bef + test_query[len_q],:].reset_index(drop = True)
                Xt = self.ReMake2ProbSet(model, test_query[[len_q]], xt)
                if len(Xt_prob) == 0: Xt_prob = Xt
                else: Xt_prob = pd.concat([Xt_prob, Xt], axis = 0)
    
                for rank in range(3):
                    if rank == 0:  yy = pd.Series(data = np.where(ytest == 10, 1, 0))
                    elif rank == 1: yy = pd.Series(data = np.where(ytest >= 7, 1, 0))
                    else: yy = pd.Series(data = np.where(ytest >= 4, 1, 0))
                    
                    if len(yt[rank]) == 0: yt[rank] = yy
                    else: yt[rank] = pd.concat([yt[rank], yy], axis = 0)
                
            number = min([5, test_query[len_q]])
            for k in range(number):
                if len(pay):
                    if index[k] == 0:
                        eff_test[k,0] = pay[1]
                        eff_test[k,1] = pay[2]
                    elif index[k] == 1: eff_test[k,1] = pay[3]
                    elif index[k] == 2: eff_test[k,1] = pay[4]
                No_test[k].append([y[index[k]], val_stan[index[k]], eff_test[k,0], eff_test[k,1]])
                
            if len_q < len(test_query) - 1: bef = np.sum(test_query[:len_q + 1])
        
        # 確率モデルをテストデータで作成
        if NUM < 1:
            for rank in range(3):
                X1, Xv, y1, yv, q1, q2, r1, r2 = self.GetQuery_Sampling(Xt_prob.values, yt[rank].values, test_rid, ratio = 0.8, query = test_query)
                self.Make_ProbModel(X1, y1, Xv, yv, rank, typ)
            
        fig_f, ax_f = plt.subplots(figsize = (20, 15))
        fig, ax = plt.subplots(figsize = (15,15))
        fig_c, ax_c = plt.subplots(figsize = (10,10))
        
        # 単勝と複勝の回収率
        for n, disc in enumerate(["train", "test"]):
            
            if not os.path.exists(f"{self.stan_path}{Rtype[typ]}/{disc}"):
                os.mkdir(f"{self.stan_path}{Rtype[typ]}/{disc}")
            if not os.path.exists(f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}"):
                os.mkdir(f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}")
            for k in range(5):
                
                if not os.path.exists(f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}/No{k+1}_pred"):
                    os.mkdir(f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}/No{k+1}_pred")
                    
                if disc == "train": No_k = np.array(No[k])
                else: No_k = np.array(No_test[k])
                
                a = No_k[No_k[:,0] == 10, 1:] # 1着
                b = No_k[No_k[:,0] == 7, 1:] # 2着
                c = No_k[No_k[:,0] == 3, 1:] # 3着
                d = No_k[No_k[:,0] <= 1, 1:] # 着外
                
                bins = np.linspace(math.floor(np.min(No_k[:,1])*10)/10, math.ceil(np.max(No_k[:,1])*10)/10, 32)
                bins_ = np.linspace(0, 1, 31)
                hist, hist1, hist2, hist3, label = [], [], [], [], []
                if len(a): 
                    hist.append(a[:,0])
                    # hist1.append(a[:,3])
                    # hist2.append(a[:,4])
                    # hist3.append(a[:,5])
                    label.append("1着")
                if len(b): 
                    hist.append(b[:,0])
                    # hist1.append(b[:,3])
                    # hist2.append(b[:,4])
                    # hist3.append(b[:,5])
                    label.append("2着")
                if len(c): 
                    hist.append(c[:,0])
                    # hist1.append(c[:,3])
                    # hist2.append(c[:,4])
                    # hist3.append(c[:,5])
                    label.append("3着")
                if len(d): 
                    hist.append(d[:,0])
                    # hist1.append(d[:,3])
                    # hist2.append(d[:,4])
                    # hist3.append(d[:,5])
                    label.append("着外")
                    
                for kk in range(4):
                    if kk == 0: fig_c, ax_c = self.Plot_figure([hist, bins, label], [fig_c, ax_c], f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}/No{k+1}_pred/{kk+1}_correlation.png", mode = 2)
                    # elif kk == 1: fig_c, ax_c = self.Plot_figure([hist1, bins_, label], [fig_c, ax_c], f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}/No{k+1}_pred/{kk+1}_correlation.png", mode = 2)
                    # elif kk == 2: fig_c, ax_c = self.Plot_figure([hist2, bins_, label], [fig_c, ax_c], f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}/No{k+1}_pred/{kk+1}_correlation.png", mode = 2)
                    # elif kk == 3: fig_c, ax_c = self.Plot_figure([hist3, bins_, label], [fig_c, ax_c], f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}/No{k+1}_pred/{kk+1}_correlation.png", mode = 2)
                
                wins = np.array([140, 150, 200, 300, 400, 500, 700, 1000, 2000, 3000, 5000, 10000, 10010])# 単勝のbin
                place = np.array([110, 130, 150, 170, 200, 300, 400, 500, 700, 1000, 1010])
                wins_cnt = np.zeros((len(bins), len(wins)), dtype = np.int32)
                place_cnt = np.zeros((len(bins), len(place)), dtype = np.int32)
                wins_cnt_ = np.zeros((len(bins_), len(wins), 3), dtype = np.int32)
                place_cnt_ = np.zeros((len(bins_), len(place), 3), dtype = np.int32)
                
                for kk in range(len(bins)):
                    
                    if kk == 0: ind_bin = np.ravel(np.where(No_k[:,1] < bins[1]))
                    elif kk == len(bins) - 1: ind_bin = np.ravel(np.where(No_k[:,1] > bins[-2]))
                    else: ind_bin = np.ravel(np.where(np.all(np.array(list(zip(No_k[:,1] >= bins[kk - 1], No_k[:,1] < bins[kk]))),axis=1) == True))
                    
                    if len(ind_bin):
                        
                        No_k_bins = No_k[ind_bin[No_k[ind_bin, 0] == 10], :]
                        if len(No_k_bins):
                            for kj in range(len(wins)):
                                if kj == 0: ind_win = np.ravel(np.where(No_k_bins[:,2] <= wins[kj]))
                                elif kj == len(wins) - 1: ind_win = np.ravel(np.where(No_k_bins[:,2] >= wins[kj]))
                                else: 
                                    ind_win = np.ravel(np.where(np.all(np.array(list(zip(No_k_bins[:,2] > wins[kj - 1], No_k_bins[:,2] <= wins[kj]))), axis = 1) == True))
                                wins_cnt[kk, kj] = len(ind_win)
                                
                        No_k_bins = No_k[ind_bin[No_k[ind_bin, 0] >= 3], :]
                        if len(No_k_bins):
                            for kj in range(len(place)):
                                if kj == 0: ind_place = np.ravel(np.where(No_k_bins[:,3] <= place[kj]))
                                elif kj == len(place) - 1: ind_place = np.ravel(np.where(No_k_bins[:,3] >= place[kj]))
                                else: 
                                    ind_place = np.ravel(np.where(np.all(np.array(list(zip(No_k_bins[:,3] > place[kj - 1], No_k_bins[:,3] <= place[kj]))), axis = 1) == True))
                                place_cnt[kk, kj] = len(ind_place)
                                
                '''for rank in range(3):
                    for kk in range(len(bins_)):
                        if kk == 0: continue
                        elif kk == 1: ind_bin = np.ravel(np.where(No_k[:,4 + rank] < bins_[kk + 1]))
                        elif kk == len(bins_) - 1: ind_bin = np.ravel(np.where(No_k[:,4 + rank] > bins_[-2]))
                        else: ind_bin = np.ravel(np.where(np.all(np.array(list(zip(No_k[:,4 + rank] >= bins_[kk - 1], No_k[:,4 + rank] < bins_[kk]))),axis=1) == True))
                        
                        if len(ind_bin):
                            
                            No_k_bins = No_k[ind_bin[No_k[ind_bin, 0] == 10], :]
                            if len(No_k_bins):
                                for kj in range(len(wins)):
                                    if kj == 0: continue
                                    if kj == 1: ind_win = np.ravel(np.where(No_k_bins[:,2] <= wins[kj]))
                                    elif kj == len(wins) - 1: ind_win = np.ravel(np.where(No_k_bins[:,2] >= wins[kj]))
                                    else: 
                                        ind_win = np.ravel(np.where(np.all(np.array(list(zip(No_k_bins[:,2] > wins[kj - 1], No_k_bins[:,2] <= wins[kj]))), axis = 1) == True))
                                    wins_cnt_[kk, kj, rank] = len(ind_win)
                                    
                            No_k_bins = No_k[ind_bin[No_k[ind_bin, 0] >= 3], :]
                            if len(No_k_bins):
                                for kj in range(len(place)):
                                    if kj == 0: continue
                                    if kj == 1: ind_place = np.ravel(np.where(No_k_bins[:,3] <= place[kj]))
                                    elif kj == len(place) - 1: ind_place = np.ravel(np.where(No_k_bins[:,3] >= place[kj]))
                                    else: 
                                        ind_place = np.ravel(np.where(np.all(np.array(list(zip(No_k_bins[:,3] > place[kj - 1], No_k_bins[:,3] <= place[kj]))), axis = 1) == True))
                                    place_cnt_[kk, kj, rank] = len(ind_place)'''
                
                for rank in range(4):
                    if rank == 0:
                        xx, yy = np.meshgrid(wins, bins)
                        fig, ax = self.Plot_figure([xx, yy, wins_cnt], [fig, ax], f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}/No{k+1}_pred/{rank + 1}_wins.png", mode = 1)
                    #else:
                        #xx, yy = np.meshgrid(wins, bins_)
                        #fig, ax = self.Plot_figure([xx, yy, wins_cnt_[:,:,rank - 1]], [fig, ax],  f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}/No{k+1}_pred/{rank + 1}_wins.png", mode = 1)
                
                
                for rank in range(4):
                    if rank == 0:
                        xx, yy = np.meshgrid(place, bins)
                        fig, ax = self.Plot_figure([xx, yy, place_cnt], [fig, ax], f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}/No{k+1}_pred/{rank + 1}_place.png", mode = 1)
                    #else:
                       # xx, yy = np.meshgrid(place, bins_)
                        #fig, ax = self.Plot_figure([xx, yy, place_cnt_[:,:,rank - 1]], [fig, ax], f"{self.stan_path}{Rtype[typ]}/{disc}/{NUM}/No{k+1}_pred/{rank + 1}_place.png", mode = 1)
    
        #特徴量の重要度
        if NUM > 0:
            if not os.path.exists(f"{self.feature_path}{NUM}"): os.mkdir(f"{self.feature_path}{NUM}")
            importance = pd.DataFrame(model[0].feature_importances_ / float(model[0].feature_importances_.sum()),
                                      index = self.ex_columns, columns=["importance"])
            imp = importance.sort_values('importance', ascending = False)
            fig_f, ax_f = self.Plot_figure(imp, [fig_f, ax_f],  f"{self.feature_path}{NUM}/Feature-importance_{Rtype[typ]}", mode = 0)
            
        fig_f.clf()
        fig.clf()
        fig_c.clf()
        plt.close("all")
      
    def Plot_figure(self, args, figs, fig_path, mode = 0): 
        
        fig, ax = figs
        if mode == 0:
            imp = args
            
            ax.barh(imp.index[:75], imp['importance'].values[:75], align='center')
            fig.tight_layout()
            fig.savefig(fig_path + "_Upper75.png")

            ax.cla()
            time.sleep(3)
            
            ax.barh(imp.index[-75:], imp['importance'].values[-75:], align='center')
            fig.tight_layout()
            fig.savefig(fig_path + "_Lower75.png")
            
        elif mode == 1:
            xx, yy, cnt = args

            for n in range(2):
                mappable = ax.pcolormesh(xx, yy, cnt[1:,1:], cmap = "jet", norm = Normalize(vmin = 0, vmax = np.max(cnt)//5 * 5))
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", "5%", pad = 0.05)
                pp = fig.colorbar(mappable, cax = cax)
                if n == 0: pp.remove()
            
            pp.set_ticks(np.linspace(0, int(np.amax(cnt)/10+1)*10,6))
            fig.tight_layout()
            fig.savefig(fig_path)
            
        elif mode == 2:
            hist, bins, label = args
            
            ax.hist(hist, bins, label = label)
            ax.legend(loc = "upper left")
            #ax.set_xticks(np.linspace(0,5*np.max(np.array(hist))//5, 6))
            #ax.set_yticks(np.linspace(0,5*np.max(np.array(bins))//5, 6))
            fig.savefig(fig_path)
          
        #ax.set_axis_off()
        ax.cla()
        #fig.clf()
        time.sleep(3)
        
        return fig, ax
            
    def Run_query(self, jyo, num_pid, trial = None, test = False): # ランキング学習
        
        if trial is not None:
            params = {'objective':'lambdarank', # ランキング学習
                      'boosting_type': 'gbdt',      # GBDTを指定
                      'metric' : 'ndcg',            # 最適化するロスの定義
                      'learning_rate': 0.01,        # 学習率
                      'n_estimators': 10000,        # 決定木ノードの最小データ数
                      'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                      'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                      'num_leaves': trial.suggest_int('num_leaves', 64, 256),
                      'max_depth':trial.suggest_int('num_leaves', 3, 10),
                      'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                      'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                      'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                      'max_bin': 1000,              # binの数
                      'min_data_in_bin': 1,         # binに入る最小データ数
                      'random_state': 100,
                      'min_data_in_leaf': 100,
                      'eval_at': (1,2,3)
                      }
        else:
            params = {'objective':'lambdarank', # ランキング学習
                      'boosting_type': 'gbdt',      # GBDTを指定
                      'metric' : 'ndcg',            # 最適化するロスの定義
                      'learning_rate': 0.1,       # 学習率
                      'n_estimators': 10000,        # 決定木ノードの最小データ数
                      #'max_depth': 9,               # 深さ
                      #'num_leaves':350,
                      'bagging_fraction':0.8,       # バギング比率
                      'bagging_freq': 50,           # バギング頻度
                      'feature_fraction':0.4,
                      'min_data_in_bin': 10,        # binに入る最小データ数
                      'random_state': 100,
                      'min_data_in_leaf': 100,      # binの数
                      'eval_at': (1,2,3)
                      }
        
        
        # MYSQL
        conn = mysql.connector.connect(
            host = 'localhost',
            port = '3306',
            user = 'root',
            password = 'american1016#',
            database = 'race_info'
        )
        #カーソル呼び出し
        curs = conn.cursor()
        
        X, y1, y2, y3, y4, rid = self.combine_datasets(jyo, num_pid, curs)
        boost_r = self.boost_r
        
        le = LabelEncoder() # カテゴリカル変数をラベル -> 登録
        label = [[[] for j in range(4)] for i in range(len(self.categorical))]
        for i, column in enumerate(self.ex_columns[self.categorical]):
            label[i][0] = column
            for j in range(1,4):
                if len(X[j-1]) == 0: continue
                Xorigin = X[j-1][column]
                le = le.fit(X[j-1][column])
                X[j-1][column] = le.transform(X[j-1][column])
                index = X[j-1][column].duplicated() == False
                label[i][j] = np.concatenate([Xorigin[index], X[j-1][column][index]]).reshape(2,-1).T
        
        with open(f".\\{self.folder}\\0\\{Track[jyo - 1]}\\label_encoder.pickle", mode = "wb") as f: pickle.dump(label, f)
        with open(f".\\{self.folder}\\0\\{Track[jyo - 1]}\\dataset.pickle", mode = "wb") as f: pickle.dump(X, f)
        with open(f".\\{self.folder}\\0\\{Track[jyo - 1]}\\no1.pickle", mode = "wb") as f: pickle.dump(y1, f)
        with open(f".\\{self.folder}\\0\\{Track[jyo - 1]}\\no2.pickle", mode = "wb") as f: pickle.dump(y2, f)
        with open(f".\\{self.folder}\\0\\{Track[jyo - 1]}\\no3.pickle", mode = "wb") as f: pickle.dump(y3, f)
        with open(f".\\{self.folder}\\0\\{Track[jyo - 1]}\\no4.pickle", mode = "wb") as f: pickle.dump(y4, f)
        
        fig, ax = plt.subplots(figsize = (20, 15))
        for typ in range(3): 
            params["max_depth"] = self.depth_list[typ]
            params["num_leaves"] = int(2 ** params["max_depth"] * 0.7)
            
            path = f"./{self.folder}/0/{Track[jyo - 1]}/Model_LightGBM/params_{Rtype[typ]}.json"
            if os.path.exists(path):
                with open(path, "r") as f: params = json.load(f)
            else:
                with open(path, "w") as f: json.dump(params, f, indent = 4)
                
            if len(X[typ]) == 0: continue
            #Xtrain, Xtest, ytrain, ytest, train_query, test_query, train_rid, test_rid = self.GetQuery_Sampling(X[typ], y1[typ][:,0], rid[typ], ratio = 0.9, y2 = y2[typ][:,0], y3 = y3[typ][:,0], y4 = y4[typ][:,0])
            Xtrain, Xtest, ytrain, ytest, train_query, test_query, train_rid, test_rid = self.GetQuery_Sampling(X[typ], y1[typ][:,0], rid[typ], ratio = 0.8, y2 = y2[typ][:,0], y3 = y3[typ][:,0], y4 = y4[typ][:,0])
            
            if test:
                Xtest.reset_index(drop = True, inplace = True)
                ytest.reset_index(drop = True, inplace = True)
                # 検証用データを保存
                with open(f"{self.valDset_path}Xtest__{Rtype[typ]}.pickle", "wb") as f: dill.dump(Xtest, f)
                with open(f"{self.valDset_path}ytest__{Rtype[typ]}.pickle", "wb") as f: dill.dump(ytest, f)
                with open(f"{self.valDset_path}query__{Rtype[typ]}.pickle", "wb") as f: dill.dump(test_query, f)
                with open(f"{self.valDset_path}race_id__{Rtype[typ]}.pickle", "wb") as f: dill.dump(test_rid, f)    
            
            if self.Nfold > 1:
                Xfold, _, yfold, _, qfold, _, rfold, _ = self.GetQuery_Sampling(Xtrain.values, ytrain.values, train_rid, ratio = 0.8, query = train_query, Nfold = self.Nfold)
             
                for n in range(self.Nfold):
                    print(f"query construct @ {n+1} / {self.Nfold}")
                    if self.Nfold > 1:
                        train_rid = []
                        for k in range(self.Nfold):
                            if n == k:
                                Xvalid = Xfold[k]
                                yvalid = yfold[k]
                                valid_query = qfold[k]
                            else:
                                if len(train_rid) == 0:
                                    Xtrain = Xfold[k]
                                    ytrain = yfold[k]
                                    train_query = qfold[k]
                                    train_rid = rfold[k]
                                else:
                                    Xtrain = pd.concat([Xtrain, Xfold[k]], axis = 0)
                                    ytrain = pd.concat([ytrain, yfold[k]], axis = 0)
                                    train_query = np.concatenate([train_query, qfold[k]])
                                    train_rid = np.concatenate([train_rid, rfold[k]])
            else:
                if test:
                    Xtrain, Xvalid, ytrain, yvalid, train_query, valid_query, train_rid, _ = self.GetQuery_Sampling(Xtrain.values, ytrain.values, train_rid, query = train_query)
                else:
                    Xvalid, yvalid, valid_query, valid_rid = Xtest, ytest, test_query, test_rid

            Xtrain.reset_index(drop = True, inplace = True)
            Xvalid.reset_index(drop = True, inplace = True)
            ytrain.reset_index(drop = True, inplace = True)
            yvalid.reset_index(drop = True, inplace = True)
            
            train_Dsets = lgb.Dataset(Xtrain, label = ytrain, group = train_query, categorical_feature = self.categorical)
            valid_Dsets = lgb.Dataset(Xvalid, label = yvalid, group = valid_query, reference = train_Dsets, categorical_feature = self.categorical)
            
            '''if os.path.exists(f"{self.model_path}/{n+1}/{Rtype[typ]}.pickle"):
                with open(f"{self.model_path}/{n+1}/{Rtype[typ]}.pickle", "rb") as f:
                    model = dill.load(f)
            else:
            model.fit(Xtrain, ytrain,
                   group = train_query,
                   categorical_feature = self.categorical,
                   eval_metric = 'ndcg',
                   eval_set = [(Xvalid, yvalid)],
                   eval_group = [valid_query],
                   eval_at = (1,2,3),
                   #callbacks = [lgb.early_stopping(stopping_rounds = 100, first_metric_only = True, verbose = 10)],
                   verbose = 10)'''

            model = lgb_opt.train(params, 
                                  train_Dsets, 
                                  valid_sets = [valid_Dsets], 
                                  num_boost_round = boost_r[typ],
                                  callbacks = [lgb.early_stopping(stopping_rounds = 80)]
                                  )
            
            best_params = model.params
            
            model = lgb.train(best_params, 
                            train_Dsets, 
                            valid_sets = [valid_Dsets], 
                            num_boost_round = boost_r[typ],
                            callbacks = [lgb.early_stopping(stopping_rounds = 80)]
                            )
            if self.Nfold == 1:
                with open(f"{self.model_path}{Rtype[typ]}.pickle", "wb") as f: 
                    dill.dump(model, f)

            else:
                if not os.path.exists(f"{self.model_path}/{n+1}"): os.mkdir(f"{self.model_path}/{n+1}")
                # モデルを保存   
                with open(f"{self.model_path}/{n+1}/{Rtype[typ]}.pickle", "wb") as f: 
                    dill.dump(model, f)

                if not os.path.exists(f"{self.feature_path}{n+1}"): os.mkdir(f"{self.feature_path}{n+1}")
            
            # 特徴量の重要度
            importance = pd.DataFrame(model.feature_importance(iteration = model.best_iteration) / float(model.feature_importance(iteration = model.best_iteration).sum()),
                                      index = self.ex_columns, columns=["importance"])
            imp = importance.sort_values('importance', ascending = False)
        
            ax.barh(imp.index[:75], imp['importance'].values[:75], align='center')
            fig.tight_layout()
            
            path = f"{self.feature_path}Feature-importance_{Rtype[typ]}_Upper75.png" if self.Nfold == 1 else f"{self.feature_path}{n+1}/Feature-importance_{Rtype[typ]}_Upper75.png"
            fig.savefig(path)

            ax.cla()
            time.sleep(3)
            
            ax.barh(imp.index[-75:], imp['importance'].values[-75:], align='center')
            fig.tight_layout()
            path = f"{self.feature_path}Feature-importance_{Rtype[typ]}_Lower75.png" if self.Nfold == 1 else f"{self.feature_path}{n+1}/Feature-importance_{Rtype[typ]}_Lower75.png"
            fig.savefig(path)
            
            ax.cla()
            time.sleep(3)
            
        curs.close
        conn.close
        
def return_payment(rid, curs, conn):
    for k in range(2):
        try:
             curs.execute(f"SELECT race_rank, pay_win, pay_place1, pay_place2, pay_place3, pay_waku_ren, pay_uma_ren,\
                          pay_wide12, pay_wide13, pay_wide23, pay_uma_tan, pay_trio, pay_tierce FROM race_info.`{rid}`;")
             pay = np.ravel(np.array(curs.fetchall()))
         
             index = np.ravel(np.where(pay < 0))
             if np.any(np.in1d(index, [0,4,5], invert = True)):# 同着の場合は-1が入っているため、上のものだけを入れる
                 Set_payment(f"https://db.netkeiba.com/race/{rid}", curs, conn)
             else: break
        except: 
             pay = []
             break
 
    res = None
    '''curs.execute(f"SELECT horse_num FROM race_result.`{rid}` WHERE ranking > 0 ORDER BY ranking;")
    res = np.ravel(np.array(curs.fetchall()))'''
    return pay, res
 
if __name__ == "__main__":
    
    plt.rcParams['font.family'] = 'MS Gothic'
    plt.rcParams["font.size"] = 15
    for jyo in [2,3,5,9]:
        constractor = Constractor(jyo = jyo, Nfold = 1, folder = "Datasets_basis")
        constractor.Run_query(jyo, num_pid = 7)
        print(f"{Track[jyo-1]} output done...")