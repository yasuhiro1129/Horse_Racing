# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 22:01:01 2022

@author: yasub
"""
import numpy as np
import mysql.connector
import sys
import os
import traceback
from multiprocessing import Process, freeze_support
from _global import Globals
#各競馬場毎のタイム指数
'''
4C時点での上位5頭の平均所要時間の平均差
入線時上位3頭+αの上りタイムの平均差
'''

col=["sample={:.0f}","no1={:.3f}","no2={:.3f}","no3={:.3f}","Ave_odds={:.1f}","no3_odds={:.1f}",\
     "ave_4c={:.2f}","no3_4c={:.2f}","rap4c={:.3f}","rap3f={:.3f}"]
    
def Labeling(grade, option, tp = 0):
    
    if option != "name":
        try: grade = int(grade)
        except: 
            if option == "career":
                return 0 #未出走
            else: 
                return np.nan
    
    if option == "age":
        if grade > 7: grade = 6
        else: grade = int(grade) - 2
    elif option == "weight":
        if grade < 400: grade = 0
        elif grade > 400 + 10*13: grade = 14
        else: grade = (grade - 400)//10 + 1
    elif option == "change":
        if grade < -21: grade = 0
        elif grade >= 21: grade = 15
        else: grade = (grade + 21)//3 + 1
    elif option == "carry":
        if grade <= 52: grade = 0
        elif grade > 58: grade = 8#7
        else: grade = int(grade - 51)
    elif option == "name":
        element = ["新馬","未勝利","1勝","2勝","3勝"]
        if grade in element: grade = 1
        else: grade = 0
    elif option == "month":
        if grade in [4,5,6]: grade = [-1,1]
        elif grade in [7,8,9]: grade = [-1,-1]
        elif grade in [10,11,12]: grade = [1,-1]
        else: grade = [1,1]
    elif option == "intev":
        if grade >= 24: grade = 0
        else: 
            grade = grade//2
            if grade <= 0: grade = 0
    elif option == "waku":
        pass
    elif option == "foward":
        if grade < 0 or grade > 1: grade = np.nan
        if grade <= 1/3: grade = 0
        elif grade <= 2/3: grade = 1
        else: grade = 2
    elif option == "career":
        if grade <= 10: grade = grade
        elif grade <= 15: grade = 11
        elif grade <= 20: grade = 12
        elif grade <= 30: grade = 13
        else: grade = 14
    elif tp < 2:
        if option == "grade":
            if grade == 0: grade = 0
            elif grade <= 2:  grade = 1
            else: grade = 2
        elif option == "rank":
            if grade == 0: grade = 0
            elif grade == 1:  grade = 1
            elif grade <= 4:  grade = 2
            else: grade = 3
        elif option == "distance":
            if grade <= 1400: grade = 0
            elif grade < 2000: grade = 1
            elif grade <= 2400: grade = 2
            else: grade = 3
    else:
        if option == "grade":
            if grade <= 1: grade = 0
            else: grade = 1
        elif option == "rank":
            if grade <= 4:  grade = 0
            else: grade = 1
        elif option == "distance":
            if grade < 3500: grade = 0
            else: grade = 1        
    return grade


class Results(Globals):
    def __init__(self, pid = 0):
        super().__init__(pid = pid)
        # ジョッキー
        sql=""
        for n in range(2):
            for k in range(1,11): sql += self.Rtype[n]+"_sample_{:02d}+".format(k)
              
        self.jtbl, _ = self.Return_Person_table("jockey")
        self.ttbl, self.ttblc = self.Return_Person_table("trainer")
        self.otbl, self.otblc = self.Return_Person_table("owner")
        self.btbl, self.btblc = self.Return_Person_table("breeder")
        self.lfa, _ = self.Return_Person_table("father")
        self.lmf, _ = self.Return_Person_table("mfather")
    
    #データベース"waku"：枠,ジョッキー,父,母父,性別,競馬場,距離,馬場(オーナー/調教師/生産者は枠順に影響を与えない)
    def correlation_waku(self, Year, pid):
        
        self.curs.execute("SELECT jockey_id,jockey_label FROM jockey_int;")
        jockey = np.array(self.curs.fetchall())
        
        line = self.log_loading("waku_", pid, Year[0])
        array = np.zeros((10), float)
        flag = np.sum(np.array(line)[1:])
        for year in Year:
            if line[0] < year: continue
            for jyo in range(line[1], 10):
                try:
                    self.curs.execute("SHOW TABLES FROM race_result LIKE '{}{:02d}%';".format(year,jyo+1))
                    tables = self.curs.fetchall()
                    for j in range(line[2], len(tables)):
                        try:
                            tbl = tables[j][0]
                            self.curs.execute("SELECT type,distance,turf_condition,dirt_condition,isout FROM race_info.`{}`".format(tbl))
                            lab = self.curs.fetchall()
                            self.curs.execute("SELECT horse_sex,winning_odds,fin_corner,horse_id,jockey_id,waku,\
                                         total_time,fin3f,ranking FROM race_result.`{}` Order By Ranking;".format(tbl))
                        
                            res = np.array(self.curs.fetchall())
                            res = res[np.where(res[:,-1].astype(int)>0)[0],:-1]
                           
                            typ = lab[0][0]
                            dis = Labeling(lab[0][1], "distance", typ)
                            cond = max(lab[0][2], lab[0][3])
                            if cond > 2: cond = 2
                           
                            for k in range(line[3], len(res)):
                                if flag: 
                                    flag = 0
                                    continue
                                try:
                                   self.curs.execute("SELECT fid,mfid FROM horse_prof.`{}`".format(int(res[k,3])))
                                   hid = self.curs.fetchall()
                                   
                                   waku = res[k,5]
                                   jid = np.where(jockey[:,0]==int(res[k,4]))[0]
                                   if len(jid): jid = jockey[jid[0],1]
                                   else:jid = -1
                                   
                                   fid = np.where(self.lfa[:,0] == str(hid[0][0]))[0]
                                   if len(fid): fid = self.lfa[fid[0],1]
                                   else: fid = -1
                                   mfid = np.where(self.lmf[:,0] == str(hid[0][1]))[0]
                                   if len(mfid):mfid = self.lmf[mfid[0],1]
                                   else: mfid = -1
                                       
                                   sex = int(res[k,0])
                                   X = 100 * (float(res[k,6]) - float(res[k,7])) / float(lab[0][1] - 600)
                                   Y = res[k,7] / 6
                                   
                                   if typ:
                                        try:
                                            x = X - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                                                    & (self.tendency[jyo][typ][:,1] == cond))[0],3][0]
                                            y = Y - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                                                    & (self.tendency[jyo][typ][:,1] == cond))[0],4][0]
                                        except:
                                            x = X - self.tendency[jyo][typ][np.where(self.tendency[jyo][typ][:,0] == lab[0][1])[0],3][0]
                                            y = Y - self.tendency[jyo][typ][np.where(self.tendency[jyo][typ][:,0] == lab[0][1])[0],4][0]
                                   else:
                                        try:
                                            x = X - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                                & (self.tendency[jyo][typ][:,1] == cond)
                                                                & (self.tendency[jyo][typ][:,2] == lab[0][4]))[0],4][0]
                                            y = Y - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                                & (self.tendency[jyo][typ][:,1] == cond)
                                                                & (self.tendency[jyo][typ][:,2] == lab[0][4]))[0],5][0]
                                        except:
                                            x = X - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                                & (self.tendency[jyo][typ][:,2] == lab[0][4]))[0],4][0]
                                            y = Y - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                                & (self.tendency[jyo][typ][:,2] == lab[0][4]))[0],5][0]
                                   
                                   self.curs.execute("SELECT sample,no1,no2,no3,Ave_odds,no3_odds,ave_4c,no3_4c,rap4c,rap3f FROM waku WHERE \
                                                waku={} AND jockey_label={} AND father_label={} AND mfather_label={} AND type={} AND\
                                                sex={} AND track={} AND distance={} AND cond={};".format(waku,jid,fid,mfid,typ,sex,jyo+1,dis,cond))
                                   ret = self.curs.fetchall()
                                   
                                   if len(ret):
                                       
                                       for m in range(10):
                                           if m in [5,7]: array[m] = ret[0][m] * array[3]
                                           elif m == 0: array[m] = ret[0][m]
                                           else:array[m] = ret[0][m]*array[0]
                                           
                                       array[0] += 1
                                       array[4] = ret[0][4]*ret[0][0] + float(res[k,1])
                                       array[6] = ret[0][6]*ret[0][0] + (float(res[k,2])-1)/(len(res)-1)
                                       array[8] = ret[0][8]*ret[0][0] + x
                                       array[9] = ret[0][9]*ret[0][0] + y
                                       if k == 0: array[1] = ret[0][1]*ret[0][0] + 1
                                       if k <= 1: array[2] = ret[0][2]*ret[0][0] + 1
                                       if k <= 2:
                                           array[3] = ret[0][3] * ret[0][0] + 1
                                           array[5] = ret[0][5] * ret[0][3] * ret[0][0] + float(res[k,1])
                                           array[7] = ret[0][7] * ret[0][3] * ret[0][0] + (float(res[k,2]) - 1) / (len(res) - 1)
                                           
                                       for m in range(1,10):
                                           if m in [5,7]:
                                               if array[3] > 0: array[m]/=array[3]*array[0]
                                           else: array[m] /= array[0]                                
                                       
                                       for m in range(10):
                                           sql = "UPDATE waku SET " + col[m].format(array[m]) + " WHERE waku={} AND jockey_label={} AND father_label={} AND mfather_label={}\
                                               AND type={} AND sex={} AND track={} AND distance={} AND cond={};".format(waku,jid,fid,mfid,typ,sex,jyo+1,dis,cond)
                                           self.curs.execute(sql)
                                           self.conn.commit()
                                   else:
                                       
                                       for m in range(8): array[m] = 0
                                           
                                       array[0] = 1
                                       array[4] = float(res[k,1])
                                       array[6] = float(res[k,2]) / len(res)
                                       if k == 0: array[1] = 1
                                       if k <= 1: array[2] = 1
                                       if k <= 2:
                                           array[3] = 1
                                           array[5] = float(res[k,1])
                                           array[7] = float(res[k,2]) / len(res)
                                       sql="INSERT INTO waku (waku,jockey_label,father_label,mfather_label,type,sex,track,distance,cond,sample,\
                                           no1,no2,no3,Ave_odds,no3_odds,ave_4c,no3_4c,rap4c,rap3f) VALUES({},{},{},{},{},{},{},{},{},1,{:.3f},\
                                           {:.3f},{:.3f},{:.1f},{:.1f},{:.2f},{:.2f},{:.3f},{:.3f});".format(waku,jid,fid,mfid,typ,sex,jyo+1,\
                                           dis,cond,array[1],array[2],array[3],array[4],array[5],array[6],array[7],x,y)
                                       self.curs.execute(sql)
                                       self.conn.commit()
                                       
                                   self.log_writing(f".\\waku_sql_pid{pid}.txt", [year, jyo, j, k])
                                except mysql.connector.Error as err:
                                    self.Write_ErrorLog(f".\\waku_sql_pid{pid}.txt", err)
                                    k -= 1
                                    continue
                                except Exception as err: 
                                    self.Write_ErrorLog(f".\\waku_sql_pid{pid}.txt", err)
                                
                            line[3] = 0
                        except mysql.connector.Error as err:
                            self.Write_ErrorLog(f".\\waku_sql_pid{pid}.txt", err)
                            j -= 1
                            continue
                        except Exception as err:
                            self.Write_ErrorLog(f".\\waku_sql_pid{pid}.txt", err)
                    line[2] = 0
                except mysql.connector.Error as err:
                    self.Write_ErrorLog(f".\\waku_sql_pid{pid}.txt", err)
                    jyo -= 1
                    continue
                except Exception as err: self.Write_ErrorLog(f".\\waku_sql_pid{pid}.txt", err)
            line[1] = 0
                 
    
    #テーブル："horse_ability": ジョッキー,調教師,生産者,オーナー,父,母父,競馬場,馬場,距離における4C、3Fラップを出す
    def horse_performance(self, Year, pid):

        line = self.log_loading("horse-perf_", pid, Year[0])
        array = np.zeros((10), float)
        flag = np.sum(np.array(line)[1:])
        for year in Year:
            if line[0] < year: continue
            for jyo in range(line[1], 10):
                try:
                    self.curs.execute("SHOW TABLES FROM race_result LIKE '{}{:02d}%';".format(year,jyo+1))
                    tables = self.curs.fetchall()
                    for i in range(line[2], len(tables)):
                        try:
                            tbl = tables[i][0]
                            self.curs.execute("SELECT type,turf_condition,dirt_condition,distance,isout FROM race_info.`{}`".format(tbl))
                            lab = self.curs.fetchall()
                            
                            typ = lab[0][0]
                            cond = max(lab[0][1],lab[0][2])
                            if cond > 2: cond = 2
                            dis = Labeling(lab[0][3],"distance",typ)
                            
                            self.curs.execute("SELECT horse_sex,total_time,fin3f,jockey_id,trainer_id,owner_id,horse_id,winning_odds,\
                                         fin_corner,ranking FROM race_result.`{}` Order by ranking;".format(tbl))
                            res = np.array(self.curs.fetchall())
                            res = res[np.where(res[:,-1].astype(int) > 0)[0], :-1]
                            for j in range(line[3], len(res)):
                                if flag: 
                                    flag = 0
                                    continue
                                try:
                                    #各出走馬IDから属性を取得
                                    try:
                                        self.curs.execute("SELECT fid,mfid,breeder_id FROM horse_prof.`{:d}`".format(int(res[j,6])))
                                    except Exception:
                                        self.curs.execute("SELECT fid,mfid,breeder_id FROM horse_prof.`{}`".format(res[j,6]))
                                        
                                    hid = self.curs.fetchall()
                                    
                                    #ジョッキーラベルの取得
                                    jid = np.where(self.jtbl[:,0] == int(res[j,3]))[0]
                                    if len(jid): jid = self.jtbl[jid[0],1]
                                    else: jid = -1
                                    #調教師ラベルの取得
                                    try:
                                        tid = np.where(self.ttbl[:,0] == int(res[j,4]))[0]
                                        if len(tid): tid = self.ttbl[tid[0],1]
                                        else: tid = -1
                                    except Exception:
                                        if len(self.ttblc):
                                            tid = np.where(self.ttblc[:,0] == res[j,4])[0]
                                            if len(tid): tid = int(self.ttblc[tid[0],1])
                                            else: tid = -1
                                        else: tid = -1
                                    #オーナーラベルの取得
                                    try:
                                        oid = np.where(self.otbl[:,0] == int(res[j,5]))[0]
                                        if len(oid): oid = self.otbl[oid[0],1]
                                        else: oid = -1
                                    except Exception:
                                        if len(self.otblc):
                                            oid = np.where(self.otblc[:,0] == res[j,5])[0]
                                            if len(oid): oid = int(self.otblc[oid[0],1])
                                            else: oid = -1
                                        else: oid = -1
                                    #生産者ラベルの取得
                                    try:
                                        bid = np.where(self.btbl[:,0] == int(hid[0][2]))[0]
                                        if len(bid): bid = self.btbl[bid[0],1]
                                        else: bid = -1
                                    except Exception:
                                        if len(self.btblc):
                                            bid = np.where(self.btblc[:,0] == hid[0][2])[0]
                                            if len(bid): bid = int(self.btblc[bid[0],1])
                                            else: bid = -1
                                        else: bid = -1
                                        
                                    fid = np.where(self.lfa[:,0] == str(hid[0][0]))[0]
                                    if len(fid): fid = self.lfa[fid[0],1]
                                    else: fid = -1
                                    mfid = np.where(self.lmf[:,0] == str(hid[0][1]))[0]
                                    if len(mfid): mfid = self.lmf[mfid[0],1]
                                    else: mfid = -1
                                        
                                    sex = int(res[j,0])
                                    
                                    X = 100 * (float(res[j,1]) - float(res[j,2]))/float(lab[0][3] - 600)
                                    Y = float(res[j,2])/6
                                    if typ:
                                        try:
                                            x = X - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][3])
                                                                                    & (self.tendency[jyo][typ][:,1] == cond))[0],3][0]
                                            y = Y - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][3])
                                                                                    & (self.tendency[jyo][typ][:,1] == cond))[0],4][0]
                                        except:
                                            x = X - self.tendency[jyo][typ][np.where(self.tendency[jyo][typ][:,0] == lab[0][3])[0],3][0]
                                            y = Y - self.tendency[jyo][typ][np.where(self.tendency[jyo][typ][:,0] == lab[0][3])[0],4][0]
                                    else:
                                        try:
                                            x = X - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][3])
                                                                & (self.tendency[jyo][typ][:,1] == cond)
                                                                & (self.tendency[jyo][typ][:,2] == lab[0][4]))[0],4][0]
                                            y = Y - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][3])
                                                                & (self.tendency[jyo][typ][:,1] == cond)
                                                                & (self.tendency[jyo][typ][:,2] == lab[0][4]))[0],5][0]
                                        except:
                                            x = X - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][3])
                                                                & (self.tendency[jyo][typ][:,2] == lab[0][4]))[0],4][0]
                                            y = Y - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][3])
                                                                & (self.tendency[jyo][typ][:,2] == lab[0][4]))[0],5][0]
                                    
                                    self.curs.execute("SELECT sample,no1,no2,no3,Ave_odds,no3_odds,ave_4c,no3_4c,rap4c,rap3f FROM horse_performance\
                                                 WHERE jockey_label={} AND trainer_label={} AND breeder_label={} AND owner_label={} AND\
                                                 father_label={} AND mfather_label={} AND type={} AND sex={} AND track={} AND distance={}\
                                                 AND cond={};".format(jid,tid,bid,oid,fid,mfid,typ,sex,jyo+1,dis,cond))
                                    ret = self.curs.fetchall()
                                    
                                    if len(ret):
                                        
                                        for k in range(10):
                                            if k in [5,7]: array[k] = ret[0][k]*array[3]
                                            elif k==0: array[k] = ret[0][k]
                                            else: array[k] = ret[0][k]*array[0]
                                            
                                        array[0] += 1
                                        array[4] = ret[0][4]*ret[0][0]+float(res[j,7])
                                        array[6] = ret[0][6]*ret[0][0]+(float(res[j,8]) - 1)/(len(res) - 1)
                                        array[8] = ret[0][8]*ret[0][0]+x
                                        array[9] = ret[0][9]*ret[0][0]+y
                                        if j == 0: array[1] = ret[0][1]*ret[0][0]+1
                                        if j <= 1: array[2] = ret[0][2]*ret[0][0]+1
                                        if j <= 2:
                                            array[3] = ret[0][3]*ret[0][0]+1
                                            array[5] = ret[0][5]*ret[0][3]*ret[0][0]+float(res[j,7])
                                            array[7] = ret[0][7]*ret[0][3]*ret[0][0]+(float(res[j,8])-1)/(len(res)-1)
                                            
                                        for k in range(1,10):
                                            if k in [5,7]:
                                                if array[3] > 0: array[k] /= array[3]*array[0]
                                            else: array[k] /= array[0]                                
                                        
                                        for k in range(10):
                                            sql = "UPDATE horse_performance SET " + col[k].format(array[k])+" WHERE jockey_label={} AND\
                                                trainer_label={} AND breeder_label={} AND owner_label={} AND father_label={} AND mfather_label={} AND type={}\
                                                AND sex={} AND track={} AND distance={} AND cond={};".format(jid,tid,bid,oid,fid,mfid,typ,sex,jyo+1,dis,cond)
                                            self.curs.execute(sql)
                                            self.conn.commit()
                                    else:
                                        
                                        for k in range(10): array[k] = 0
                                            
                                        array[0] = 1
                                        array[4]=float(res[j,7])
                                        array[6]=(float(res[j,8])-1)/(len(res)-1)
                                        if j == 0: array[1] = 1
                                        if j <= 1: array[2] = 1
                                        if j <= 2:
                                            array[3] = 1
                                            array[5] = float(res[j,7])
                                            array[7] = (float(res[j,8])-1)/(len(res)-1)
                                        sql = "INSERT INTO horse_performance (jockey_label,trainer_label,breeder_label,owner_label,father_label,\
                                            mfather_label,type,sex,track,distance,cond,sample,no1,no2,no3,Ave_odds,no3_odds,ave_4c,no3_4c,rap4c,rap3f)\
                                            VALUES({},{},{},{},{},{},{},{},{},{},{},1,{:.3f},{:.3f},{:.3f},{:.1f},{:.1f},{:.2f},{:.2f},{:.3f},{:.3f});".format\
                                            (jid,tid,bid,oid,fid,mfid,typ,sex, jyo + 1, dis,cond,array[1],array[2],array[3],array[4],array[5],array[6],array[7],x,y)
                                        self.curs.execute(sql)
                                        self.conn.commit()
                                        
                                    self.log_writing(f".\\horse-perf_sql_pid{pid}.txt", [year, jyo, i, j])
                                except mysql.connector.Error as err:
                                    self.Write_ErrorLog(f".\\horse-perf_sql_pid{pid}.txt", err)
                                    j -= 1
                                    continue
                                except Exception as err: self.Write_ErrorLog(f".\\horse-perf_sql_pid{pid}.txt", err)
                            line[3] = 0
                        except mysql.connector.Error as err:
                            self.Write_ErrorLog(f".\\horse-perf_sql_pid{pid}.txt", err)
                            i -= 1
                            continue
                        except Exception as err: self.Write_ErrorLog(f".\\horse-perf_sql_pid{pid}.txt", err)
                    line[2] = 0
                except mysql.connector.Error as err:
                    self.Write_ErrorLog(f".\\horse-perf_sql_pid{pid}.txt", err)
                    jyo -= 1
                    continue
                except Exception as err: self.Write_ErrorLog(f".\\horse-perf_sql_pid{pid}.txt", err)
            line[1] = 0
    
        
        
    #テーブル："weight": 馬体重、体重変化、年齢、間隔、調教師、父親、母父親
    #テーブル："carry": 斤量、ジョッキー、馬体重、減量騎手、位置取り、父親、母父親
    #テーブル："career": 歴戦数、年齢、生まれ月、レース年齢,季節、父親、母父親
    def horse_property(self, Year):
    
        col01=[s for s in col if "odds" not in s]        
          
        line = self.log_loading("horse-prop_", self.pid, Year[0])        
        flag = np.sum(np.array(line)[1:])
        for year in Year:
             if line[0] < year: continue
             for jyo in range(line[1], 10):
                 try:
                     self.curs.execute("SHOW TABLES FROM race_result LIKE '{}{:02d}%';".format(year,jyo+1))
                     tables = self.curs.fetchall()
                 except mysql.connector.Error as err:
                     self.Write_ErrorLog(f".\\horse-prop_sql_pid{self.pid}.txt", err)
                     jyo -= 1
                     continue
                 except Exception as err: self.Write_ErrorLog(f".\\horse-prop_sql_pid{self.pid}.txt", err)
                 
                 for j in range(line[2], len(tables)):
                     try:
                         tbl = tables[j][0]
                         self.curs.execute("SELECT type,distance,turf_condition,dirt_condition,\
                                           race_weight,race_sex,race_age,day_id,month,race_class,\
                                           isout FROM race_info.`{}`".format(tbl))
                         lab = self.curs.fetchall()
                         self.curs.execute("SELECT race_name FROM race_info.`{}`".format(tbl))
                         rname = self.curs.fetchall()
                         self.curs.execute("SELECT horse_sex,horse_age,horse_weight,weight_change,weight_to_carry,\
                                    fin_corner,horse_id,jockey_id,trainer_id,total_time,fin3f,ranking FROM\
                                    race_result.`{}` Order By Ranking;".format(tbl))
                   
                         res = np.array(self.curs.fetchall())
                         res = res[np.where(res[:,-1].astype(int) > 0)[0],:-1]                     
                      
                         typ = lab[0][0]
                         dis = Labeling(lab[0][1], "distance", typ)
                         Rage = lab[0][6]
                         season = Labeling(lab[0][8], "month", typ)   
                         grd = Labeling(lab[0][9], "grade", typ)   
                         cond = max(lab[0][2],lab[0][3])
                         if cond > 2: cond = 2
                                
                         #減量騎手を見抜く
                         special = Labeling(rname,"name",typ)#非特別レース
                         if special and not lab[0][4]:#非ハンデ戦
                             if Rage == 3:#2歳戦
                                 isdec = np.array(list(map(lambda x : 1 if x < 54 else 0, res[:,4])))
                             else:
                                 isdec=np.zeros((len(res)),int)
                                 for k in range(len(res)):
                                     if res[k,0]:#牡馬、セン馬
                                         base = 56 if year < 2022 else 57
                                         isdec[k] = int(res[k,4] - base)
                                     else:#牝馬
                                         base = 54 if year < 2022 else 55
                                         isdec[k] = int(res[k,4] - base)
                         else: 
                             isdec = np.zeros((len(res)), int)
                             
                     except mysql.connector.Error as err:
                         self.Write_ErrorLog(f".\\horse-prop_sql_pid{self.pid}.txt", err)
                         j -= 1
                         continue
                     except Exception as err: self.Write_ErrorLog(f".\\horse-prop_sql_pid{self.pid}.txt", err)

                     for k in range(line[3], len(res)):
                         if flag: 
                             flag= 0
                             continue
                         try:
                             self.curs.execute("SELECT fid,mfid,month,day FROM horse_prof.`{}`".format(int(res[k,6])))
                             hid = self.curs.fetchall()

                             #間隔とキャリア
                             self.curs.execute("SELECT intev,total_num,Fin_corner,day_id FROM horse_result.`{}` WHERE day_id<={} Order By Day_ID DESC;".format(int(res[k,6]),lab[0][7]))
                             hres = np.array(self.curs.fetchall())
                             # intevがNullのとき
                             try:
                                np.sum(hres[:,0])
                             except:
                                for kk in range(len(hres)):
                                    if kk == len(hres) - 1: hres[kk,0] = -1
                                    else: 
                                        day1 = str(hres[kk,3])
                                        day2 = str(hres[kk + 1,3])
                                        YY = (int(day1[:4]) - int(day2[:4]))*365
                                        MM = (int(day1[4:6]) - int(day2[4:6]))*30
                                        DD = (int(day1[6:]) - int(day2[6:]))
                                        intev = int((YY + MM + DD)/7 +0.5)
                                        if intev > 127: intev = 127
                                        hres[kk,0] = intev
                                        print(f"intev: {hres[kk,0]}")

                                    self.curs.execute(f"UPDATE horse_result.`{res[k,6]:.0f}` SET intev={hres[kk,0]} WHERE day_id={hres[kk,3]:.0f};")
                                    self.conn.commit()
                             hres = hres[1:,:-1]
                             if len(hres):
                                 career = len(hres)
                                 intev = Labeling(hres[0][0],"intev",typ)
                                 tar = np.where(hres[:,2]>0)[0]
                                 tarlen = min(5,career)
                                 foward = np.array(list(map(lambda x, y : (x - 1) / (y - 1), hres[tar[:tarlen], 2],
                                                         hres[tar[:tarlen], 1] ))).sum() / tarlen
                                 if foward < 1 / 3: foward = 0
                                 elif foward < 2 / 3: foward = 1
                                 else: foward = 2
                             else:
                                 career = 0
                                 intev = -1
                                 foward = -1
                                 
                             #ジョッキーラベルの取得
                             jid = np.where(self.jtbl[:,0] == int(res[k,7]))[0]
                             if len(jid): jid = self.jtbl[jid[0],1]
                             else: jid = -1
                             
                             #調教師ラベルの取得
                             try:
                                 tid = np.where(self.ttbl[:,0] == int(res[k,8]))[0]
                                 if len(tid): tid = self.ttbl[tid[0],1]
                                 else: tid = -1
                             except Exception:
                                 if len(self.ttblc):
                                     tid = np.where(self.ttblc[:,0]==res[k,8])[0]
                                     if len(tid): tid=int(self.ttblc[tid[0],1])
                                     else: tid = -1
                                 else: tid = -1
                                 
                             #父馬ラベルの取得
                             fid = np.where(self.lfa[:,0] == str(hid[0][0]))[0]
                             if len(fid): fid = self.lfa[fid[0],1]
                             else: fid = -1
                             
                             #母父馬ラベルの取得
                             mfid = np.where(self.lmf[:,0] == str(hid[0][1]))[0]
                             if len(mfid): mfid = self.lmf[mfid[0],1]
                             else: mfid = -1
          
                             sex = int(res[k,0])
                             birth = 2 * (int(hid[0][2]) - 1) + int((float(hid[0][3]) - 1)/30 + 0.5)
                             age = Labeling(res[k,1],"age", typ)
                             weight = Labeling(res[k,2],"weight", typ)
                             change = Labeling(res[k,3],"change", typ)
                             carry = Labeling(res[k,4],"carry", typ)
          
                             X = 100 * (float(res[k,9]) - float(res[k,10]))/float(lab[0][1] - 600)
                             Y = float(res[k,10]) / 6
          

                             if typ:
                                 try:
                                     x = X - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                                              & (self.tendency[jyo][typ][:,1] == cond))[0],3][0]
                                     y = Y - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                        & (self.tendency[jyo][typ][:,1] == cond))[0],4][0]
                                 except:
                                     x = X - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1]))[0],3][0]
                                     y = Y - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1]))[0],4][0]
                                 
                             else:
                                 try:
                                     x = X - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                            & (self.tendency[jyo][typ][:,1] == cond)
                                                            & (self.tendency[jyo][typ][:,2] == lab[0][10]))[0],4][0]
                                     y = Y - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                            & (self.tendency[jyo][typ][:,1] == cond)
                                                            & (self.tendency[jyo][typ][:,2] == lab[0][10]))[0],5][0]
                                 except:
                                     x = X - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                            & (self.tendency[jyo][typ][:,2] == lab[0][10]))[0],4][0]
                                     y = Y - self.tendency[jyo][typ][np.where((self.tendency[jyo][typ][:,0] == lab[0][1])
                                                            & (self.tendency[jyo][typ][:,2] == lab[0][10]))[0],5][0]
                             #"weight"テーブルへの追加
                             if (self.pid - 1) % 3 == 0:
                                 dic = dict()
                                 dic = {"weight":weight,"delta":change,"age":age,"intev":intev,
                                      "trainer_label":tid,"father_label":fid,"mfather_label":mfid,
                                      "type":typ,"sex":sex,"distance":dis,"track":jyo+1,"cond":cond}
                                 self.register("weight", dic, x, y, j, k, col01, year, jyo, res)

                             #"carry"テーブルへの追加
                             elif (self.pid - 1) % 3 == 1:
                                 dic = dict()
                                 dic = {"carry":carry,"jockey_label":jid,"foward":foward,"weight":weight,
                                        "decreasing":isdec[k],"father_label":fid,"mfather_label":mfid,
                                        "type":typ,"sex":sex,"distance":dis,"track":jyo+1,"cond":cond}
                                 self.register("carry", dic, x, y, j, k, col01, year, jyo, res)

                             #"career"テーブルへの追加
                             elif (self.pid - 1) % 3 == 2:
                                 dic = dict()
                                 dic = {"career":career,"birth":birth,"age":age,"race_age":Rage,"season1":season[0],
                                        "season2":season[1],"father_label":fid,"mfather_label":mfid,
                                        "type":typ,"sex":sex,"distance":dis,"track":jyo+1,"cond":cond}
                                 self.register("career", dic, x, y, j, k, col01, year, jyo, res)
                                    
                         except mysql.connector.Error as err:
                                 self.Write_ErrorLog(f".\\horse-prop_sql_pid{self.pid}.txt", err)
                                 k -= 1
                                 continue
                         except Exception as err: self.Write_ErrorLog(f".\\horse-prop_sql_pid{self.pid}.txt", err)

                     line[3] = 0
                 line[2] = 0
             line[1] = 0
                     
    def register(self, disc, dic, x, y, j, k, col01, year, jyo, res):
         
        for loop in range(2):
            array = np.zeros((10),float)
            base = ""
            sql = f"SELECT sample,no1,no2,no3,ave_4c,no3_4c,rap4c,rap3f FROM {disc} WHERE"
            for key, val in dic.items(): base += f" {key}={val} AND"
            sql += base[:-4] + ";"
             
            try:
                self.curs.execute(sql)
                ret = self.curs.fetchall()
                                    
                if len(ret):
                                                
                    for m in range(8):
                        if m == 5: array[m] = ret[0][m] * array[3]
                        elif m == 0: array[m] = ret[0][m]
                        else: array[m] = ret[0][m] * array[0]
                                                        
                    array[0] += 1
                    array[4] += (float(res[k,5])-1)/(len(res)-1)
                    array[6] += x
                    array[7] += y
                    if k == 0: array[1] += 1
                    if k <= 1: array[2] += 1
                    if k <= 2:
                        array[3] += 1
                        array[5] += (float(res[k,5]) - 1) / (len(res) - 1)
                    
                    for m in range(1,8):
                        if m == 5:
                            if array[3] > 0: array[m] /= array[3] * array[0]
                            else: array[m] /= array[0]
                        else: array[m] /= array[0]
                    
                    sql = f"UPDATE {disc} SET "
                    for m in range(8): sql += col01[m].format(array[m]) + ","
                    sql = sql[:-1] + " WHERE" + base[:-4] + ";" 
                    self.curs.execute(sql)
                    self.conn.commit()
                else:
                    
                    keys = f"INSERT INTO {disc} ("
                    vals = "sample,no1,no2,no3,ave_4c,no3_4c,rap4c,rap3f) VALUES("
                    for key, val in dic.items():
                        keys += key + ","
                        vals += str(val) + ","
                                                    
                    array[0] = 1
                    array[4] = (float(res[k,5])-1)/(len(res)-1)
                    array[6], array[7] = x, y
                    if k == 0: array[1] = 1
                    if k <= 1: array[2] = 1
                    if k <= 2:
                        array[3] = 1
                        array[5] = (float(res[k,5])-1)/(len(res)-1)
                            
                    for m in range(8):
                        if m == 0: vals += f"{array[m]:.0f},"
                        elif m <= 3 or m >= 6: vals += f"{array[m]:.3f},"
                        else: vals += f"{array[m]:.2f},"
                                       
                    self.curs.execute(keys + vals[:-1] + ");")
                    self.conn.commit()
            except mysql.connector.Error as err:
                self.Write_ErrorLog(f".\\horse-prop_sql_pid{self.pid}.txt", err)
                continue
            except Exception as err:
                self.Write_ErrorLog(f".\\horse-prop_sql_pid{self.pid}.txt", err) 
            break
        self.log_writing(f".\\horse-prop_sql_pid{self.pid}.txt", [year, jyo, j, k])
            
        
def wrapper(year, pid, yyear):
    SQLclass = Results(pid = pid)
    SQLclass.correlation_waku(year, pid)
    #if None in year: SQLclass.horse_performance(yyear, pid) 
    #else: SQLclass.horse_property(year)
    

if __name__=="__main__":
    
    multi = True
    YEAR = range(2022, 2007, -1)
    
    conn = mysql.connector.connect(
        host = 'localhost',
        port = '3306',
        user = 'root',
        password = 'american1016#',
        database = 'corresponding'
    )
    curs = conn.cursor(buffered = True)
    
    #テーブルの作成
    #競馬場×馬場×距離
    '''curs.execute("DROP TABLE IF EXISTS waku;")
    conn.commit()
    curs.execute("CREATE TABLE waku (waku TINYINT, jockey_label SMALLINT,father_label SMALLINT,mfather_label SMALLINT,\
                 type TINYINT,sex TINYINT,track TINYINT,distance TINYINT,cond TINYINT,sample SMALLINT,no1 FLOAT,no2 FLOAT,\
                 no3 FLOAT,Ave_odds FLOAT, no3_odds FLOAT,ave_4c FLOAT,no3_4c FLOAT,rap4c FLOAT,rap3f FLOAT);")
    conn.commit()
    
    curs.execute("DROP TABLE IF EXISTS horse_performance;")
    conn.commit()
    curs.execute("CREATE TABLE horse_performance (jockey_label SMALLINT,trainer_label SMALLINT,breeder_label SMALLINT,\
                 owner_label SMALLINT,father_label SMALLINT,mfather_label SMALLINT,type TINYINT,sex TINYINT,\
                 track TINYINT,distance TINYINT,cond TINYINT,sample SMALLINT,no1 FLOAT,no2 FLOAT,no3 FLOAT,\
                 Ave_odds FLOAT,no3_odds FLOAT,ave_4c FLOAT,no3_4c FLOAT,rap4c FLOAT,rap3f FLOAT);")
    conn.commit()
    
    curs.execute("DROP TABLE IF EXISTS weight;")
    conn.commit()
    curs.execute("DROP TABLE IF EXISTS carry;")
    conn.commit()
    curs.execute("DROP TABLE IF EXISTS career;")
    conn.commit()
    curs.execute("CREATE TABLE weight (weight TINYINT,delta TINYINT,age TINYINT,intev TINYINT,trainer_label SMALLINT,\
                father_label SMALLINT,mfather_label SMALLINT,type TINYINT,sex TINYINT,track TINYINT,distance TINYINT,\
                cond TINYINT,sample SMALLINT,no1 FLOAT,no2 FLOAT,no3 FLOAT,ave_4c FLOAT,no3_4c FLOAT,rap4c FLOAT,rap3f FLOAT);")
    curs.execute("CREATE TABLE carry (carry TINYINT,jockey_label SMALLINT,weight TINYINT,decreasing TINYINT,foward TINYINT,\
                father_label SMALLINT,mfather_label SMALLINT,type TINYINT,sex TINYINT,track TINYINT,distance TINYINT,cond TINYINT,\
                sample SMALLINT,no1 FLOAT,no2 FLOAT,no3 FLOAT,ave_4c FLOAT,no3_4c FLOAT,rap4c FLOAT,rap3f FLOAT);")
    curs.execute("CREATE TABLE career (career TINYINT,birth TINYINT,race_age TINYINT,age TINYINT,season1 FLOAT,season2 FLOAT,\
                father_label SMALLINT,mfather_label SMALLINT,type TINYINT,sex TINYINT,track TINYINT,distance TINYINT,cond TINYINT,\
                sample SMALLINT,no1 FLOAT,no2 FLOAT,no3 FLOAT,ave_4c FLOAT,no3_4c FLOAT,rap4c FLOAT,rap3f FLOAT);")
    conn.commit()'''
    
    curs.close
    conn.close
    
    if multi:
        process_num = 5
        year = [[] for i in range(process_num)]
        counter = 0
        while counter < len(YEAR):
            for i in range(process_num):
                try: year[i].append(YEAR[counter])
                except: break
                counter += 1
            
        result = []
        res = [[] for i in range(process_num)]
        freeze_support()
        for i in range(process_num):

            res[i] = Process(target = wrapper, kwargs = {"year":year[i], "pid":i+1, "yyear":None})
            res[i].start()
            
        [res[i].join() for i in range(process_num)]
    else:
        wrapper([2022], pid = 1, yyear = [2022])