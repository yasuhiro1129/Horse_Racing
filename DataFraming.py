# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from selenium import webdriver
import traceback
from datetime import datetime as dt
from bs4 import BeautifulSoup
import re
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import InvalidSessionIdException
from selenium.webdriver.chrome.options import Options
from time import sleep
import mysql.connector
import sys
    

def SetInitial():
    options =Options()
    #options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    browser = webdriver.Chrome('chromedriver.exe', options = options)
    YEAR = 2008
    #MYSQL
    conn = mysql.connector.connect(
        host = 'localhost',
        port = '3306',
        user = 'root',
        password = 'american1016#',
        database = 'race_info'
    )
    #カーソル呼び出し
    curs = conn.cursor()
    return browser, YEAR, conn, curs

def Getsource_fromPage(driver,page):
    while 1:
        try:
            driver.get(page)
            driver.implicitly_wait(40)  # 見つからないときは、60秒まで待つ
            page_source = driver.page_source
            soup=BeautifulSoup(page_source,'lxml')
            break
        except TimeoutException as ex:
            driver=WriteErrorLog(driver,page,ex)
        except Exception as e:
            #ページを拾えない時は毎回ログ出力
            driver=WriteErrorLog(driver,page,e)
    
    return driver,soup

def WriteErrorLog(driver,page,e):
    '''tdate=dt.now()
    filepath=os.getcwd()+'\\'+tdate.strftime('%Y-%m-%d %H-%M-%S')+'.txt'
    with open(filepath,'w',newline='',encoding='utf_8_sig') as fp:
        fp.write(page+'\n\n')
        #fp.write('Driver Session: '+driver.session_id)
        fp.write("{}\n\n".format(e))
        fp.write(traceback.format_exc())'''
    
    #ブラウザがクラッシュしているのでそもそも閉じれない
    #driver.close()
    options=Options()
    #options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    while 1:
        sleep(5)
        try:
            driver.quit()
            driver=webdriver.Chrome('chromedriver.exe',options=options)  
            break
        except InvalidSessionIdException:
            driver=webdriver.Chrome('chromedriver.exe',options=options)
            pass
        except Exception:
            driver=webdriver.Chrome('chromedriver.exe',options=options)
            pass
    
    '''with open(filepath,'a',newline='',encoding='utf_8_sig') as fp:
        fp.write('\n\nNew Driver Session: '+driver.session_id)'''
        
    return driver
        
def ExecuteSQL(statement,curs,conn):
    try:
        curs.execute(statement)
        conn.commit()
    #SQL文法ミス:後で追加できるようにログを吐いておく
    except mysql.connector.ProgrammingError as err:
        if "Unknown column" in str(err):
            col=re.findall("column \'(.*)\' in","{}".format(err))[0]
            tbl=re.findall('`(.*)`',statement)[0]
            try:
                int(tbl)
                tbl="`"+tbl+"`"
            except:
                pass
            
            try:
                curs.execute("ALTER TABLE {} ADD ({} tinyint);".format(tbl,col))
                curs.execute(statement)
                conn.commit()
            except mysql.connector.Error:
                pass
            
            funChangeColumn(statement,"Day_ID",curs,conn,1)
        else:
            tdate=dt.now()
            filepath=os.getcwd()+'\\SQL_Error_'+tdate.strftime('%Y-%m-%d %H-%M-%S')+'.txt'
            with open(filepath,'w',newline='',encoding='utf_8_sig') as fp:
                fp.write('Statement: '+statement+'\n\n')
                fp.write("{}\n\n".format(err))
                fp.write(traceback.format_exc())
    #SQL中でのエラー
    except mysql.connector.Error as err:
        if "Data too long" in "{}".format(err):#varchar型のカラムの長さが足りない
            col=re.findall("column \'(.*)\'","{}".format(err))[0]
            funChangeColumn(statement,col,curs,conn,0)
        elif "Data truncated" in "{}".format(err):#int型からvarchar()型に変更
            col=re.findall("column \'(.*)\'","{}".format(err))[0]
            funChangeColumn(statement,col,curs,conn,0)
        elif "Incorrect integer value" in "{}".format(err):#int型からvarchar()型に変更
            col=re.findall("column \'(.*)\'","{}".format(err))[0]
            funChangeColumn(statement,col,curs,conn,0)
        elif "Duplicate" in "{}".format(err):
            pass
        else:
            tdate=dt.now()
            filepath=os.getcwd()+'\\SQL_Error_'+tdate.strftime('%Y-%m-%d %H-%M-%S')+'.txt'
            with open(filepath,'w',newline='',encoding='utf_8_sig') as fp:
                fp.write('Statement: '+statement+'\n\n')
                fp.write("{}\n\n".format(err))
                fp.write(traceback.format_exc())

def funChangeColumn(statement, key, curs, conn, option):
    tbl = re.findall('`(.*)`',statement)[0]
    try:
        int(tbl)
        tbl = "`"+tbl+"`"
    except: pass
    
    cnt = 0
    for i in range(statement.find(key)):
        if statement[i] == ',':
            cnt = cnt + 1
    num = statement.find('VALUES')
    cnt00, val, flag = 0, 0, 1
    for i in range(num,len(statement)):
        if statement[i] == ',' and flag == 1:
            cnt00 = cnt00 + 1
        if statement[i] == '"' or statement[i] == "'":
            flag *= -1
        if cnt00 == cnt:
            flag = 1
            for j in range(1,100):
                if statement[i + j] == '"' or statement[i + j] == "'":
                    flag *= -1
                if statement[i + j] == ',' and flag == 1:
                    val = j - 1
                    break
                elif statement[i + j] == ')':
                    val = j - 1
                    break
        if val: break
        
    if option:#カラム"intev"を追加するために一旦削除
        curs.execute("DELETE FROM {} WHERE {}={}".format(tbl,key,statement[i + 1:i + val + 1]))
        conn.commit()
    else:
        sql = 'ALTER TABLE {} MODIFY COLUMN {} VARCHAR({});'.format(tbl, key, val)
        curs.execute(sql)
        conn.commit()
        
    try:
        curs.execute(statement)
        conn.commit()
    except mysql.connector.Error as err:
        if "Data too long" in str(err):#varchar型のカラムの長さが足りない
            col = re.findall("column \'(.*)\'","{}".format(err))[0]
            funChangeColumn(statement, col, curs, conn, 0)
        elif "Data truncated" in str(err):#int型からvarchar()型に変更
            col = re.findall("column \'(.*)\'","{}".format(err))[0]
            funChangeColumn(statement, col, curs, conn, 0)
        elif "Incorrect integer value" in str(err):#int型からvarchar()型に変更
            col = re.findall("column \'(.*)\'","{}".format(err))[0]
            funChangeColumn(statement, col, curs, conn, 0)
        elif "Unknown column" in str(err):
            col = re.findall("column \'(.*)\' in","{}".format(err))[0]
            curs.execute("ALTER TABLE {} ADD ({} tinyint);".format(tbl, col))
            curs.execute(statement)
            conn.commit()
            funChangeColumn(statement, "Day_ID", curs, conn, 1)
        elif "Duplicate" in str(err): pass
        else:
            tdate = dt.now()
            filepath = os.getcwd()+'\\SQL_Error_' + tdate.strftime('%Y-%m-%d %H-%M-%S') + '.txt'
            with open(filepath,'w',newline = '',encoding = 'utf_8_sig') as fp:
                fp.write('Statement: ' + statement + '\n\n')
                fp.write("{}\n\n".format(err))
                fp.write(traceback.format_exc())
    
def SetSQLstatemaent(sql, Race, typ, curs, conn):
    Race_col = Race.columns
    len00 = len(Race_col)
    if typ < 5:
        '''dfの中身
        columns=['Race','Jockey','Trainer','Horse','Owner','Breeder','f','m','ff','fm','mf','mm']\
        ,index=['ID','Name']
        '''
        df = DefNameLength(Race, 0)
        if len(df.columns) == 1: return
        Nlist = ['Race','Jockey','Trainer','Horse','Owner','Breeder','f','m','ff','fm','mf','mm']
    for i in range(len00):
        if typ == 5:#RPY
            if 'prize' in Race_col[i]: sql += ' {} INT'.format(Race_col[i])
            elif '_rate' in Race_col[i]: sql += ' {} FLOAT'.format(Race_col[i])
            elif 'others' in Race_col[i]: sql += ' {} MEDIUMINT'.format(Race_col[i])
            elif 'num' in Race_col[i]: sql += ' {} MEDIUMINT'.format(Race_col[i])
            else: sql += ' {} SMALLINT'.format(Race_col[i])
                
            if i == 0: sql += ' PRIMARY KEY,'
            elif i < len(Race_col) - 1: sql += ','
            else: sql += ');'
                
        else:
            if 'ID' in Race_col[i]:
                if 'Day' in Race_col[i]: sql += ' {} INT'.format(Race_col[i])
                else:
                    for k in range(len(Nlist)):
                        if k <= 5:
                            if Nlist[k] in Race_col[i]:
                                sql += ' {} {}'.format(Race_col[i], df.iat[0,k])
                                break
                        else:
                            if Nlist[k] + "ID" == Race_col[i]:
                                sql += ' {} {}'.format(Race_col[i], df.iat[0,k])
                                break
            elif 'Name' in Race_col[i]:
                for k in range(len(Nlist)):
                    if k <= 5:
                        if Nlist[k] in Race_col[i]:
                            sql += ' {} {}'.format(Race_col[i], df.iat[1,k])
                            break
                    else:
                        if Nlist[k] == Race_col[i]:
                            sql += ' {} {}'.format(Race_col[i], df.iat[1,k])
                            break
            elif 'year' in Race_col[i]: sql += ' {} SMALLINT'.format(Race_col[i])
            elif 'time' in Race_col[i]: sql += ' {} FLOAT'.format(Race_col[i])
            elif 'Odds' in Race_col[i]: sql += ' {} FLOAT'.format(Race_col[i])
            elif 'prize' in Race_col[i]: sql += ' {} MEDIUMINT'.format(Race_col[i])
            elif 'pace' in Race_col[i]: sql += ' {} FLOAT'.format(Race_col[i])
            elif 'Ave_pos' in Race_col[i]: sql += ' {} FLOAT'.format(Race_col[i])
            elif 'Horse_Weight' in Race_col[i]: sql += ' {} SMALLINT'.format(Race_col[i])
            elif 'distance' in Race_col[i]: sql += ' {} SMALLINT'.format(Race_col[i])
            elif 'weight_to_carry' in Race_col[i]: sql += ' {} FLOAT'.format(Race_col[i])
            elif 'pace' in Race_col[i]: sql += ' {} FLOAT'.format(Race_col[i])
            elif 'Ave_pos' in Race_col[i]: sql += ' {} FLOAT'.format(Race_col[i])
            elif 'Diff' in Race_col[i]: sql += ' {} FLOAT'.format(Race_col[i])
            elif len(Race_col[i]) <= 2: sql += ' {} VARCHAR(20)'.format(Race_col[i])
            elif 'pri' in  Race_col[i]:#獲得賞金/売値
                sql += ' {} INT'.format(Race_col[i])
            elif '3F_Rank' in Race_col[i]: sql += ' {} TINYINT'.format(Race_col[i])
            elif '3F' in Race_col[i]: sql += ' {} FLOAT'.format(Race_col[i])
            elif 'hometown' in Race_col[i]: sql += ' {} VARCHAR(20)'.format(Race_col[i])
            elif 'height' in Race_col[i]: sql += ' {} SMALLINT'.format(Race_col[i])
            elif 'deview' in Race_col[i]: sql += ' {} SMALLINT'.format(Race_col[i])
            elif 'pay_' in Race_col[i]: sql += ' {} INT'.format(Race_col[i])
            elif 'fav_' in Race_col[i]: sql += ' {} SMALLINT'.format(Race_col[i])
            elif 'page' in Race_col[i]: sql += ' {} SMALLINT default 0'.format(Race_col[i])
            elif '_win' in Race_col[i]: sql += ' {} INT'.format(Race_col[i])
            elif '_1st' in Race_col[i]: sql += ' {} INT'.format(Race_col[i])
            elif 'first' in Race_col[i]: sql += ' {} INT'.format(Race_col[i])
            elif 'Birthday' in Race_col[i]: sql += ' {} INT'.format(Race_col[i])
            elif '_win_num' in Race_col[i]: sql += ' {} SMALLINT'.format(Race_col[i])
            elif 'color'in Race_col[i]:
                if Race.iloc[0,i] == '-1': sql += ' {} TINYINT'.format(Race_col[i])
                else: sql += ' {} VARCHAR({:d})'.format(Race_col[i], len(Race.iloc[0,i]))
            elif 'pattern' in Race_col[i]:
                if Race.iloc[0,i] == '-1':
                    sql = sql + ' {} TINYINT'.format(Race_col[i])
                else:
                    sql = sql + ' {} VARCHAR({:d})'.format(Race_col[i], len(Race.iloc[0,i]))
            elif 'sleeve' in Race_col[i]:
                if Race.iloc[0,i] == '-1':
                    sql = sql + ' {} TINYINT'.format(Race_col[i])
                else:
                    sql = sql + ' {} VARCHAR({:d})'.format(Race_col[i], len(Race.iloc[0,i]))
            elif 'base' in Race_col[i]:
                sql = sql + ' {} VARCHAR({:d})'.format(Race_col[i], len(Race.iloc[0,i]))
            else: sql = sql + ' {} TINYINT'.format(Race_col[i])
                
            
            if typ == 1 and 'Horse_ID' in Race_col[i]: sql += ' PRIMARY KEY,'
            elif typ == 2 and 'Race_ID' in Race_col[i]: sql += ' PRIMARY KEY,'
            elif typ == 3 and i == 0: sql += ' PRIMARY KEY,'
            elif typ == 4 and i == len(Race_col) - 1: sql += ' ,PRIMARY KEY (Race_ID,Horse_ID));'
            elif typ < 4 and i == len(Race_col) - 1: sql += ');'
            else: sql += ','
        
    ExecuteSQL(sql, curs, conn)   
    
def Insert2SQL(ID, Race, curs, conn, option):
    try:
        if float(Race.iat[0,0]) == -1: return 0
    except: pass
    
    try:
        int(ID)
        ID = "`" + str(ID) + "`"
    except: pass
    
    df = DefNameLength(Race, 1)
    if len(df.columns) == 1:
        return 1
    Nlist = ['Race_ID','Jockey_ID','Trainer_ID','Horse_ID','Owner_ID','Breeder_ID','fID','mID','ffID','fmID','mfID','mmID']
    length = len(Race)
    if length == 1:
        for j in range(len(Race.columns)):
            value=str(Race.iat[0,j])
            if value[1:-1].find("'") >= 0: value = "'" + value[1:-1].replace("'","\\'") + "'"
            if j == 0:
                col = 'INSERT INTO {}({},'.format(ID,Race.columns[j])
                val = 'VALUES ({},'.format(value)
                if 'ID' in Race.columns[j]:
                    tk = np.where(np.array(Nlist) == Race.columns[j])[0]
                    if len(tk) > 0:
                        if "VARCHAR" in df.iat[0, tk[0]]: val = "VALUES ('{}',".format(value)
            elif j == len(Race.columns) - 1:
                col = col + '{}) '.format(Race.columns[j])
                if 'ID' in Race.columns[j]:
                    if 'Day' in Race.columns[j]: val = val + '{});'.format(value)
                    else:
                        tk = np.where(np.array(Nlist) == Race.columns[j])[0][0]
                        if "VARCHAR" in df.iat[0,tk]: val = val + "'{}');".format(value)
                        else: val = val + '{});'.format(value)
                else: val = val + '{});'.format(value)
            else:
                col = col + '{},'.format(Race.columns[j])
                if 'ID' in Race.columns[j]:
                    tk = np.where(np.array(Nlist) == Race.columns[j])[0]
                    if len(tk) > 0:
                        if "VARCHAR" in df.iat[0, tk[0]]: val = val+"'{}',".format(value)
                        else: val = val + '{},'.format(value)
                    else: val = val + '{},'.format(value)
                else: val = val + '{},'.format(value)
        ExecuteSQL(col + val, curs, conn)

    #Resultデータベースの場合
    else:    
    #データの挿入:Race_info
        for i in range(len(Race)):
            try:
                if int(Race.iat[i,0]) == -1: return 0
                elif int(Race.iat[i,0]) == -10: continue
            except: pass
            if option:
             #生産者/オーナー/調教師の場合は地方馬の比率が高いと思われるので重賞以外はデータが入っていない
                 if int(Race.iat[i,4]) < 1 or int(Race.iat[i,4]) > 10:
                    if int(Race.iat[i,9]) <= 0: continue
            for j in range(len(Race.columns)):
                value = str(Race.iat[i,j])
                if value[1:-1].find("'") > 0:
                    value = "'"+value[1:-1].replace("'","\\'") + "'"
                if j == 0:
                    col = 'INSERT INTO {}({},'.format(ID,Race.columns[j])
                    val = 'VALUES ({},'.format(value)
                    if 'ID' in Race.columns[j]:
                        tk = np.where(np.array(Nlist) == Race.columns[j])[0]
                        if len(tk) > 0:
                            if "VARCHAR" in df.iat[0, tk[0]]: val = "VALUES ('{}',".format(value)

                elif j == len(Race.columns) - 1:
                    col = col + '{}) '.format(Race.columns[j])
                    if 'ID' in Race.columns[j]:
                        if 'Day' in Race.columns[j]:
                            val = val + '{});'.format(value)
                        else:
                            tk = np.where(np.array(Nlist) == Race.columns[j])[0][0]
                            if "VARCHAR" in df.iat[0, tk]: val = val+ "'{}');".format(value)
                            else: val = val + '{});'.format(value)

                    else: val = val+'{});'.format(value)
                else:
                    col = col + '{},'.format(Race.columns[j])
                    if 'ID' in Race.columns[j]:
                        tk = np.where(np.array(Nlist) == Race.columns[j])[0]
                        if len(tk)>0:
                            if "VARCHAR" in df.iat[0, tk[0]]: val = val + "'{}',".format(value)
                            else: val = val + '{},'.format(value)   
                        else: val = val + '{},'.format(value)
                    else: val = val + '{},'.format(value)
            ExecuteSQL(col + val, curs, conn)
    
    return 1

def SavePage2SQL(cnt, key, conn, curs, seg):
    sql = 'USE {}_List'.format(seg)
    curs.execute(sql)
    conn.commit()
    try:
        int(key)
        sql = 'UPDATE `{}` SET page = {:d} WHERE {}_ID = {}'.format(key, cnt, seg, key)
    except:
        sql = 'UPDATE `{}` SET page = {:d} WHERE {}_ID = "{}"'.format(key, cnt, seg, key)
    ExecuteSQL(sql, curs, conn)
    
    sql = 'USE {}_Result'.format(seg)
    curs.execute(sql)
    conn.commit()
    

def Get_RaceData(browser, YEAR, conn, curs, page, year, jyo, kai, day, R):
    ID = str(year) + "{:02d}{:02d}{:02d}{:02d}".format(jyo, kai, day, R)    
    browser,soup = Getsource_fromPage(browser, page + ID)
    
    #レース情報を格納するデータフレームを生成
    RACE = pd.DataFrame(np.full((1,51),-1), dtype = 'object', 
                        columns = ['Race_ID','year','month','day','track','type',\
                                    'distance','rotation','Horse_Num','kai','date',\
                                    'round','Race_Name','Race_rank','Race_class',\
                                    'Race_age','Race_sex','Race_weight','weather',\
                                    'Turf_condition','Dirt_condition','Init3F','Fin3F',\
                                    'Diff3F','total_prize','pay_win','fav_win',\
                                    'pay_place1','pay_place2','pay_place3','fav_place1',\
                                    'fav_place2','fav_place3','pay_waku_ren',\
                                    'fav_waku_ren_fav','pay_uma_ren','fav_uma_ren',\
                                    'pay_wide12','pay_wide13','pay_wide23','fav_wide12',\
                                    'fav_wide13','fav_wide23','pay_uma_tan','fav_uma_tan',
                                    'pay_trio','fav_tri','pay_tierce','fav_tierce','Day_ID','IsOut'])
    #ページにレース内容が書かれているかの判断に使う
    RACE.iat[0,0] = '-1'
    Result = pd.DataFrame(data = [-1])
    CHART = soup.find_all('table' ,attrs = {'class':'race_table_01 nk_tb_common'})
    if len(CHART) > 0:
        RACE.iat[0,0] = ID
        RACE.iat[0,4] = str(jyo)
        RACE.iloc[0,9:12] = [str(kai), str(day), str(R)]
        title = soup.find('div', attrs = {'class':'data_intro'})
        condition = title.find('dd')
        info = title.find('p',attrs = {'class':'smalltxt'}).text.split(' ')        
        RaceName = condition.h1.text
        
        RaceRank = re.findall('\((.*)\)$', RaceName)
        if len(RaceRank) == 0:
            RaceRank = '0'
            name = re.findall('回(.*)', RaceName)
            if len(name) > 0: name00 = name[0]
            else: name00 = RaceName
        else:
            if 'OP' in RaceRank: RaceRank = '1'
            elif 'L' in RaceRank: RaceRank = '2'
            elif 'G3' in RaceRank: RaceRank = '3'
            elif 'G2' in RaceRank: RaceRank = '4'
            elif 'G1' in RaceRank:  RaceRank = '5'
            else: RaceRank = '-1'
            name = re.findall('回(.*)\(',RaceName)
            if len(name) > 0: name00 = name[0]
            else: name00 = re.findall('(.*)\(',RaceName)[0]
                  
        RACE.iloc[0, 12:14] = ["'" + name00 + "'", RaceRank]
        Head = condition.find('span').text.split('/')
        OUT = 0
        if '外' in Head[0]: OUT = 1
        RACE.iat[0,50] = OUT
        
        if '障' in Head[0]: Type = 2
        elif '芝' in Head[0]: Type = 0
        elif 'ダ' in Head[0]: Type = 1        
        if '右' in Head[0]: Rot = 0
        elif '左' in Head[0]: Rot = 1
        else: Rot = 2
        RACE.iloc[0,5:8] = [Type, re.findall('(\d*)m', Head[0][2:])[0], Rot]
        
        Weather = re.findall(':(.*)$', Head[1].replace('\xa0',''))[0]
        if '晴' in Weather: Weather = 0
        elif '曇' in Weather: Weather = 1
        elif '小雨' in Weather: Weather = 2
        elif '雨' in Weather: Weather = 3
        else:  Weather = 4
        RACE.iat[0,18] = Weather
        
        Scond00 = re.findall('芝 :(.*)', Head[2])
        Dcond00 = re.findall('ダート :(.*)', Head[2])   
        Scond = -1
        for co in Scond00:
            for kk in range(len(co)):
                coo = co[kk]
                try:
                    if ord(coo) > 10000:
                        if '不' in coo: Scond = 3
                        elif '良' in coo: Scond = 0
                        elif '稍' in coo: Scond = 1
                        elif '重' in coo: Scond = 2
                        else:  Scond = 4
                        break
                except:
                    pass
            
            
        Dcond = -1
        for co in Dcond00:
            for kk in range(len(co)):
                coo = co[kk]
                try:
                    if ord(coo) > 10000:
                        if '不' in coo: Dcond = 3
                        elif '良' in coo: Dcond = 0
                        elif '稍' in coo: Dcond = 1
                        elif '重' in coo: Dcond = 2
                        else:  Dcond=4
                        break
                except:
                    pass
        
        RACE.iloc[0,19:21] = [Scond,Dcond]
        dd=list(filter(None, re.findall('(\d*)', info[0])))
        RACE.iloc[0,1:4] = dd
        RACE.iat[0,49] = "{}{:02d}{:02d}".format(dd[0],int(dd[1]),int(dd[2]))
        
        if '3歳以上' in info[2]: Age = 0
        elif '4' in info[2]: Age = 1
        elif '3' in info[2]: Age = 2
        else: Age = 3
        RACE.iat[0,15] = Age
        
        if '新馬' in info[2]: RaceCLASS = 0
        elif '未勝利' in info[2]: RaceCLASS = 1
        elif '1勝' in info[2]: RaceCLASS = 2
        elif '2勝' in info[2]: RaceCLASS = 3
        elif '3勝' in info[2]: RaceCLASS = 4
        elif '500万' in info[2]: RaceCLASS = 2
        elif '1000万' in info[2]: RaceCLASS = 3
        elif '1600万' in info[2]: RaceCLASS = 4
        else: RaceCLASS = 5
        RACE.iat[0,14] = RaceCLASS
        
        if '牝' in info[2]: RACE.iat[0,16] = 0
        else: RACE.iat[0,16] = 1
            
        if 'ハンデ' in info[2]: RACE.iat[0,17] = 2
        elif '別定' in info[2]: RACE.iat[0,17] = 1
        else: RACE.iat[0,17] = 0
        
        #レース結果を取得
        Res =CHART[0].find_all('tr')
        #競走除外馬は未出走とみなす
        Num = 0
        for i in range(1,len(Res)):
            if '取' in Res[i].find('td',attrs={'class':'txt_r'}).text: continue
            if '除' in Res[i].find('td',attrs = {'class':'txt_r'}).text: continue
            Num = Num + 1
        RACE.iat[0,8] = Num
        
        
        box = soup.find('div',attrs = {'class':'result_info box_left'})
        #払い戻し結果
        pay = box.find('dl',attrs = {'class':'pay_block'})
        payBlocks = pay.find_all('table',attrs = {'class':'pay_table_01'})
        RACE.iloc[0,25:49] = np.tile('-1', (1,24))
        for payBlock in payBlocks:
            conts = payBlock.find_all('tr')
            for con in conts:
                try:
                    hea = con.find('th').text#券種
                    ref = con.find_all('td',attrs={'class':'txt_r'})[0].get_text('-').split('-')#配当
                    fav = con.find_all('td',attrs={'class':'txt_r'})[1].get_text('-').split('-')#人気
                    ref = list(map(lambda x : x.replace(',',''), ref)) 
                    fav = list(map(lambda x : x.replace(',',''), fav)) 
                    if '単勝' in hea: RACE.iloc[0,25:27] = [ref[0], fav[0]]
                    elif '複勝' in hea: 
                        if len(ref) == 2: 
                            ref = np.concatenate([ref[:2], [-1]])
                            fav = np.concatenate([fav[:2], [-1]])
                        else: 
                            ref = ref[:3]
                            fav = fav[:3]
                        RACE.iloc[0,27:33] = np.concatenate([ref, fav])
                    elif '枠連' in hea: RACE.iloc[0,33:35] = [ref[0], fav[0]]
                    elif '馬連' in hea: RACE.iloc[0,35:37] = [ref[0], fav[0]]
                    elif 'ワイド' in hea: 
                        if len(ref) == 2: 
                            ref = np.concatenate([ref[:2], [-1]])
                            fav = np.concatenate([fav[:2], [-1]])
                        else: 
                            ref = ref[:3]
                            fav = fav[:3]
                        RACE.iloc[0,37:43] = np.concatenate([ref, fav])
                    elif '馬単' in hea: RACE.iloc[0,43:45] = [ref[0], fav[0]]
                    elif '三連複' in hea: RACE.iloc[0,45:47] = [ref[0], fav[0]]
                    elif '三連単' in hea: RACE.iloc[0,47:49] = [ref[0], fav[0]]
                except:
                    pass
        #ラップタイムの取得
        discript=box.find_all('tbody')[-1].find('th').text
        if discript=='ラップ':
            Rap=soup.find('div',attrs={'class':'result_info box_left'})\
                .find_all('tbody')[-1].find('td').text.replace(' ','').split('-')
            Rap=[float(i) for i in Rap]
            #非根幹距離は始めの100mをスキップ
            if int(RACE.iat[0,6]) % 200:
                Init3F=sum(Rap[1:4])
            else:
                Init3F=sum(Rap[:3])
            Fin3F=sum(Rap[-3:])
            Diff3F=Init3F-Fin3F
        else:
            Init3F=-1
            Fin3F=-1
            Diff3F=-1
        
        RACE.iloc[0,21:24]=["{:.1f}".format(Init3F),"{:.1f}".format(Fin3F),"{:.1f}".format(Diff3F)]
        #"Result"データベースにレースIDでテーブルを作成
        Result=pd.DataFrame(np.zeros((Num,26)),dtype='object',columns=['Ranking','Waku','Horse_Num','Horse_Name','Horse_ID','Horse_SEX',\
                                                                       'Horse_AGE','Weight_To_Carry','Jockey_Name','Jockey_ID','Total_time',\
                                                                       'Diff_time','Init_corner','Fin_corner','Ave_pos','Fin3F','Fin3F_Rank',\
                                                                       'Winning_Odds','Favorite_Rank','Horse_Weight','Weight_Change',\
                                                                       'Belonging','Trainer_Name','Trainer_ID','Owner_Name','Owner_ID'])        
        try:
            #レース結果を取得
            prize=0
            pflag=1
            for i in range(1,Num+1):
                result=Res[i].find_all('td')
                for j in range(len(result)):
                    if j in [8,9,15,16,17]:
                        continue
                    elif j==20:
                        if pflag:
                            try:
                                prize=prize+float(result[j].text.replace(',',''))
                            except:
                                pflag=0
                                RACE.iat[0,24]="{:.0f}".format(prize+0.5)
                        continue
                    elif j<=2:#着順、枠、馬番
                        Result.iat[i-1,j]=result[j].text
                        if j==0:
                            dd=list(filter(None,re.findall('(\d*)',result[j].text)))
                            if len(dd)>0:
                                Result.iat[i-1,0]=dd[0]
                            elif '中' in result[j].text:
                                Result.iat[i-1,0]='-10'
                            else:
                                Result.iat[i-1,0]='-1'
                    elif j==3:#馬名、馬リンク
                        Result.iat[i-1,3]="'"+list(filter(None,re.findall('[\u30A0-\u30FF]+',result[j].text)))[0]+"'"
                        Result.iat[i-1,4]=Extract_from_slash(result[j].find('a').get('href'))
                    elif j==4:#性齢
                        if '牝' in result[j].text:
                            SEX=0
                        elif '牡' in result[j].text:
                            SEX=1
                        else:
                            SEX=2
                        Result.iat[i-1,5]=str(SEX)
                        Result.iat[i-1,6]=re.findall('[(\d*)]',result[j].text)[0]
                    elif j==5:#斤量
                        Result.iat[i-1,7]=result[j].text
                    elif j==6:#ジョッキー、ジョッキーリンク
                        Result.iat[i-1,8]="'"+result[j].text.replace('\n','')+"'"
                        Result.iat[i-1,9]=Extract_from_slash(result[j].find('a').get('href'))
                    elif j==7:#タイム、タイム差
                        index=result[j].text.find(':')
                        if index<0:
                            try:
                                Time=float(result[j].text)
                            except:
                                Time=-1
                        else:
                            Time=float(result[j].text[:index])*60+float(result[j].text[index+1:])
                        if i==1:
                            TopTime=Time
                        if Time<0:
                            Result.iloc[i-1,10:12]='-1'
                        else:
                            Result.iat[i-1,10]="{:.1f}".format(Time)
                            Result.iat[i-1,11]="{:.1f}".format(Time-TopTime)            
                    elif j==10:#コーナー通過順位
                        try:
                            cc=result[j].text.split('-')
                            cc=[str(int(k)) for k in cc]
                            if i==1:
                                Cnum=len(cc)
                            if len(cc)==Cnum:
                                Pass=[float(k) for k in cc]
                                Result.iat[i-1,12]="{:.0f}".format(Pass[0])
                                Result.iat[i-1,13]="{:.0f}".format(Pass[-1])
                                Result.iat[i-1,14]="{:.1f}".format(sum(Pass)/len(Pass))
                            else:
                                Result.iat[i-1,12]="{:.0f}".format(Pass[0])
                                Result.iloc[i-1,13:15]='-1'
                        except:
                            Result.iloc[i-1,12:15]='-1'
                    elif j==11:#上り
                        try:
                            Result.iat[i-1,15]="{:.1f}".format(float(result[j].text))
                            if len(re.findall('\d',result[j].get('class')[0])):
                                Result.iat[i-1,16]=re.findall('\d',result[j].get('class')[0])[0]
                            else:
                                Result.iat[i-1,16]='4'
                        except Exception:
                            Result.iloc[i-1,15:17]='-1'                                   
                    elif j==12:#単勝オッズ
                        try:
                            Result.iat[i-1,17]="{:.1f}".format(float(result[j].text))
                        except Exception:
                            Result.iat[i-1,17]='-1'
                    elif j==13:#単勝人気
                        Result.iat[i-1,18]=result[j].text
                        if len(result[j].text)==0:
                            Result.iat[i-1,18]='-1'
                    elif j==14:#馬体重
                        cc=re.findall('^\d*',result[j].text)
                        if len(cc)>0:
                            Result.iat[i-1,19]=cc[0]
                        else:
                            Result.iat[i-1,19]='-1'
                        cc=re.findall('\((.*)\)',result[j].text)
                        if len(cc)>0:
                            Result.iat[i-1,20]=cc[0]
                        else:
                            Result.iat[i-1,20]='-1'
                    elif j==18:#調教師
                        if re.findall('\[(.*)\]',result[j].text)[0]=='西':
                            Result.iat[i-1,21]='0'
                        elif re.findall('\[(.*)\]',result[j].text)[0]=='東':
                            Result.iat[i-1,21]='1'
                        elif re.findall('\[(.*)\]',result[j].text)[0]=='外':
                            Result.iat[i-1,21]='2'
                        else:
                            Result.iat[i-1,21]='-1'
                        Result.iat[i-1,22]="'"+re.findall('\]\n(.*)\n$',result[j].text)[0]+"'"
                        Result.iat[i-1,23]=Extract_from_slash(result[j].find('a').get('href'))
                    elif j == 19:#オーナー、オーナーリンク
                        Result.iat[i-1,24] = "'"+result[j].text.replace('\n','')+"'"
                        Result.iat[i-1,25] = Extract_from_slash(result[j].find('a').get('href'))
                
                for jj in [4,9,23,25]:
                    try:
                        int(Result.iat[i-1,jj])
                        IDd = "`"+str(Result.iat[i-1,jj])+"`"
                    except:
                        IDd = str(Result.iat[i-1,jj])
                        
                    if len(Result.iat[i-1,jj]) == 0: continue
                    
                    link = r"https://db.netkeiba.com/"
                    if jj == 4:#馬
                        link += "horse/{}/".format(Result.iat[i-1,jj])
                        sql = "SELECT 1 FROM horse_prof.{} LIMIT 1;".format(IDd)
                    '''elif jj == 9:#ジョッキー
                        link += "jockey/result/recent/{}/".format(Result.iat[i-1,jj])
                        sql = "SELECT 1 FROM jockey_list.{} LIMIT 1;".format(IDd)
                    elif jj == 23:#調教師
                        link += "trainer/result/recent/{}/".format(Result.iat[i-1,jj])
                        sql = "SELECT 1 FROM trainer_list.{} LIMIT 1;".format(IDd)
                    else:#オーナー
                        link += "owner/result/recent/{}/".format(Result.iat[i-1,jj])
                        sql = "SELECT 1 FROM owner_list.{} LIMIT 1;".format(IDd)'''
                    try:
                        curs.execute(sql)#初登場の場合はここで例外が発生
                        conn.rollback()
                         #情報の更新
                        if jj == 4:#馬
                            sql = "SELECT intev FROM horse_result.{} WHERE Race_ID={};".format(IDd, ID)
                        '''elif jj == 9:#ジョッキー
                            sql="SELECT Day_ID FROM jockey_result.{} WHERE Race_ID={};".format(IDd, ID)
                        elif jj == 23:#調教師
                            sql="SELECT Day_ID FROM trainer_result.{} WHERE Race_ID={} AND Horse_ID={};".format(IDd, ID, Result.iat[i-1,4])
                        else:#オーナー
                            sql = "SELECT Day_ID FROM owner_result.{} WHERE Race_ID={} AND Horse_ID={};".format(IDd, ID, Result.iat[i-1,4])'''
                        curs.execute(sql)
                        
                        day_id = curs.fetchall()
                        if len(day_id) == 0:

                            if jj == 4:
                                '''RESULT = realloc(RACE, Result.iloc[i-1,:], 0)
                                Did = str(day_id)
                                RESULT['intev'] = (int(RACE.iat[0,1]) - int(Did[:4]))*365 + (int(RACE.iat[0,2]) - int(Did[4:6]))*30 + int(RACE.iat[0,3]) - int(Did[6:])
                                RESULT['intev'] = int(RESULT['intev']/7 - 0.5)
                                if RESULT.at[0,'intev'] > 127: RESULT.at[0,'intev'] = 127'''
                                browser = SetHorse2SQL(browser, YEAR, conn, curs, link)
                                
                            '''if jj == 9:#ジョッキー
                                RESULT = realloc(RACE, Result.iloc[i-1, :], 1)
                            else:#オーナー
                                RESULT = realloc(RACE, Result.iloc[i-1, :], 2)
                            
                            if jj == 9: curs.execute("USE Jockey_Result;")
                            elif jj == 23: curs.execute("USE Trainer_Result;")
                            else: curs.execute("USE Owner_Result;")
                            
                            conn.commit()
                            if jj == 4:
                                RESULT = realloc(RACE, Result.iloc[i-1,:], 2)
                                curs.execute("SELECT Breeder_ID FROM Horse_prof.{};".format(IDd))
                                Bid = curs.fetchall()
                                Bid = Bid[0][0]
                                
                                curs.execute("USE Breeder_Result;")
                                conn.commit()
                                
                                #SQLに入っているIDは整数型のため頭の"0"が抜け落ちる
                                try:
                                    int(Bid)
                                    curs.execute("SELECT 1 FROM `{}`".format(Bid))
                                    conn.rollback()
                                except mysql.connector.Error:
                                    for bi in range(6):
                                        Bid = "0"+str(Bid)
                                        try:
                                            curs.execute("SELECT 1 FROM `{}` LIMIT 1;".format(Bid))
                                            conn.rollback()
                                            break
                                        except mysql.connector.Error:
                                            continue
                                except:
                                    pass
                                    
                                ret = Insert2SQL(Bid,RESULT,curs,conn,0)
                            else:
                                ret = Insert2SQL(IDd,RESULT,curs,conn,0)'''
                            
                    except mysql.connector.Error:#MYSQLにないテーブルだと例外が発生する
                        if jj == 4:#馬
                            browser = SetHorse2SQL(browser, YEAR, conn, curs, link)
                        '''elif jj==9:#ジョッキー
                            browser=SetJockey2SQL(browser,YEAR,conn,curs,link)
                        elif jj==23:#調教師
                            browser=SetTrainer2SQL(browser,YEAR,conn,curs,link)
                        elif jj==25:#オーナー
                            browser=SetOwner2SQL(browser,YEAR,conn,curs,link,'Owner')'''
                    except Exception:#MYSQLにないテーブルだと例外が発生する
                        if jj == 4:#馬
                            browser = SetHorse2SQL(browser, YEAR, conn, curs, link)
                        '''elif jj==9:#ジョッキー
                            browser=SetJockey2SQL(browser,YEAR,conn,curs,link)
                        elif jj==23:#調教師
                            browser=SetTrainer2SQL(browser,YEAR,conn,curs,link)
                        elif jj==25:#オーナー
                            browser=SetOwner2SQL(browser,YEAR,conn,curs,link,'Owner')'''
                   
        except Exception as e:
            #エラーもしくは途中で止めた場合はエラー出力して終了させる
            tdate=dt.now()
            filepath=os.getcwd()+'\\'+tdate.strftime('%Y-%m-%d %H-%M-%S')+'.txt'
            with open(filepath,'w',newline='',encoding='utf_8_sig') as fp:
                fp.write('PAGE: '+page+str(year)+"{:02d}{:02d}{:02d}{:02d}".format(jyo,kai,day,R) +'\n'+str(jj)+'\n')
                print('INDEX:{:d},{:d}\nContents:{}\n\n\n'.format(i,j,result[j].text))
                fp.write("{}\n\n".format(e))
                fp.write(traceback.format_exc())
            
            curs.close
            conn.close
            sys.exit()
                        
    return RACE,Result,browser

def realloc(RACE,Result,flag):
    
    RESULT=pd.DataFrame(np.full((1,34),-1),dtype='object',columns=['year',\
          'month','day','kai','track','date','weather','round','Race_Name','Race_rank',\
          'Race_ID','total_num','waku','horse_num','winning_Odds','favorite_rank','result',\
          'weight_to_carry','type','distance','cond','Total_time','Diff_time',\
          'Init_corner','Fin_corner','Ave_pos','pace1st','pace2nd','Diff_pace',\
          'Fin3F_time','Fin3F_Rank','Horse_Weight','Weight_Change','Day_ID'])
    
    for col in RESULT.columns:
        tfl=1
        for i in range(len(Result.index)):
            if col.lower() in Result.index[i].lower():
                RESULT[col]=Result[Result.index[i]]
                tfl=0                
        if tfl:
            for i in range(len(RACE.columns)):
                if col.lower() == RACE.columns[i].lower():
                    RESULT[col]=RACE[RACE.columns[i]]
    
    RESULT['total_num']=RACE['Horse_Num']
    RESULT['result']=Result['Ranking']
    RESULT['pace1st']=RACE['Init3F']
    RESULT['pace2nd']=RACE['Fin3F']
    RESULT['Diff_pace']=RACE['Diff3F']
    RESULT['Fin3F_time']=Result['Fin3F']
    RESULT['cond']=str(max(int(RACE['Turf_condition']),int(RACE['Dirt_condition'])))
    if int(RESULT['Race_rank'])==0:
        RESULT['Race_rank']=RACE['Race_class']
        
    if flag==0:#horse_result
        RESULT['Jockey_Name']=Result['Jockey_Name']
        RESULT['Jockey_ID']=Result['Jockey_ID']
        RESULT.rename(columns={'Fin3F_time':'Fin3F'},inplace=True)
    
    elif flag==1:#jockey_result
        RESULT['Horse_Name']=Result['Horse_Name']
        RESULT['Horse_ID']=Result['Horse_ID']
        
    else:#trainer他
        RESULT['Jockey_Name']=Result['Jockey_Name']
        RESULT['Jockey_ID']=Result['Jockey_ID']
        RESULT['Horse_Name']=Result['Horse_Name']
        RESULT['Horse_ID']=Result['Horse_ID']
            
    return RESULT

def GetPastResult(browser,link,YEAR,flag,tflag):
    browser,soup=Getsource_fromPage(browser,link)
    #flag
    # 0: jockey 1: trainer 2: owner/breeder   
    Jid='-1'
    RPY=pd.DataFrame(np.full((1,1),-1))
    persona=pd.DataFrame(np.full((1,1),-1))
    Result=pd.DataFrame(np.full((1,1),-1))
    CHART=soup.find_all('div',attrs={'id':'contents_liquid'})
    if len(CHART)==0:
        return Jid,RPY,persona,Result,browser
 
    Jid=Extract_from_slash(link)
    Heads=CHART[0].find_all('div',attrs={'class':'db_head_regist fc'})
    if len(Heads)>0:
        Heads=Heads[0].find_all('li')
        for he in Heads:
            url=he.find_all('a')
            if len(url)>0:
                url=he.find('a').get('href').find("https")
                if url<0:
                    url=r"https://db.netkeiba.com"+he.find('a').get('href')
                else:
                    url=he.find('a').get('href')
                if tflag:
                    if "プロフィール" in he.text:
                        persona,browser=GetJockeyProfile(browser,url,flag)
                        persona.iat[0,0]=Jid
                    elif "年度別成績" in he.text:#年度別成績
                        RPY,browser=GetYearlyJockey(browser,url)
                        break

    title=CHART[0].find_all('div',attrs={'class':'db_head_name fc'})
    if len(title)>0:
        if tflag and not flag:#初回かつジョッキーの場合
            try:
                info=title[0].find('p',attrs={'class':'txt_01'}).text
                if '地方' in info:#地方所属ジョッキーは無視
                    return '-1',RPY,persona,Result,browser
            except:
                pass    
    Tables=CHART[0].find_all('table',attrs={'class':'nk_tb_common race_table_01'})
    if len(Tables)==0:
        return Jid,RPY,persona,Result,browser
    
    Tables=Tables[0]
    RES=Tables.find('tbody').find_all('tr')
    dLen=len(RES)
    if dLen>0:   
        if flag:#調教師/オーナー/生産者の時はジョッキーのデータ分だけ多い
            Result=pd.DataFrame(np.full((dLen,38),-1),dtype='object',columns=['year','month','day','kai','track','date','weather','round','Race_Name',\
                                                                            'Race_rank','Race_ID','total_num','waku','horse_num','winning_Odds',\
                                                                            'favorite_rank','result','Horse_Name','Horse_ID','Jockey_Name',\
                                                                            'Jockey_ID','weight_to_carry','type','distance','cond','Total_time','Diff_time',\
                                                                            'Init_corner','Fin_corner','Ave_pos','pace1st','pace2nd','Diff_pace',\
                                                                            'Fin3F_time','Fin3F_Rank','Horse_Weight','Weight_Change','Day_ID'])
        else:
            Result=pd.DataFrame(np.full((dLen,36),-1),dtype='object',columns=['year','month','day','kai','track','date','weather','round','Race_Name',\
                                                                            'Race_rank','Race_ID','total_num','waku','horse_num','winning_Odds',\
                                                                            'favorite_rank','result','Horse_Name','Horse_ID','weight_to_carry',\
                                                                            'type','distance','cond','Total_time','Diff_time','Init_corner','Fin_corner',\
                                                                            'Ave_pos','pace1st','pace2nd','Diff_pace','Fin3F_time','Fin3F_Rank',\
                                                                            'Horse_Weight','Weight_Change','Day_ID'])
        #レース履歴を取得
        for i in range(dLen):
            parts=RES[i].find_all('td')
            for j in range(len(parts)):
                if flag:
                    #調教師/生産者/オーナーの場合は地方馬も含めて検索するので重賞以外はデータベース化しない
                    if j==5:
                        if int(Result.iat[i,4])<1 or int(Result.iat[i,4])>10:
                            if int(Result.iat[i,9])<3 or int(Result.iat[i,9])>5:
                                Result.iat[i,0]='-10'
                                break
                        continue
                    elif j in [3,23,24]:
                        continue
                else:
                    if j in [3,5,22,23]:
                        continue
                try:
                    if j==0:#日付
                        date=parts[j].text.split('/')
                        if int(date[0])<YEAR:
                            Result.iat[i,0]='-1'
                            return Jid,RPY,persona,Result,browser
                        
                        Result.iloc[i,:3]=date
                        Result.iat[i,-1]="{}{:02d}{:02d}".format(Result.iat[i,0],int(Result.iat[i,1]),int(Result.iat[i,2]))
                    elif j==1:#開催場所
                        dd=re.findall('\/sum\/(\d*)\/',parts[j].find('a').get('href'))[0]
                        if len(dd)>0:
                            Result.iat[i,4]=str(int(dd))
                            if len(re.findall('^(\d*)',parts[j].text)[0])>0:
                                Result.iat[i,3]=re.findall('^(\d*)',parts[j].text)[0]
                                Result.iat[i,5]=re.findall('[^(\d*)](\d*)$',parts[j].text)[0]
                            else:
                                Result.iat[i,3]='-1'
                                Result.iat[i,5]='-1'
                        else:
                            Result.iloc[i,3:6]='-1'
                    elif j==2:
                        if '晴' in parts[j].text:
                            Result.iat[i,6]='0'
                        elif '曇' in parts[j].text:
                            Result.iat[i,6]='1'
                        elif '小雨' in parts[j].text:
                            Result.iat[i,6]='2'
                        elif '雨' in parts[j].text:
                            Result.iat[i,6]='3'
                        elif '小雪' in parts[j].text:
                            Result.iat[i,6]='4'
                        elif '雪' in parts[j].text:
                            Result.iat[i,6]='5'
                        else:
                            try:
                                num=ord(parts[j].text)
                                if num<100:
                                    Result.iat[i,6]='-1'
                                else:
                                    Result.iat[i,6]='6'
                            except:
                                if len(parts[j].text)==0:
                                    Result.iat[i,6]='-1'
                                else:
                                    Result.iat[i,6]='6'
                            Result.iat[i,6]='6'
                    elif j==4:#レース情報
                        data=re.findall('\((.*)\)$',parts[j].text)
                        Result.iat[i,10]=Extract_from_slash(parts[j].find('a').get('href'))
                        if len(Result.iat[i,10])==0:
                            Result.iat[i,10]='-1'
                        Result.iat[i,7]=Result.iat[i,10][-2:]
                        if len(data)>0:
                            Result.iat[i,8]="'"+re.findall('^(.*)\(',parts[j].text)[0]+"'"
                            if 'OP' in data[0]:
                                Result.iat[i,9]='1'
                            elif 'L' in data[0]:
                                Result.iat[i,9]='2'
                            elif 'G3' in data[0]:
                                Result.iat[i,9]='3'
                            elif 'G2' in data[0]:
                                Result.iat[i,9]='4'
                            elif 'G1' in data[0]:
                                Result.iat[i,9]='5'
                            elif '1勝' in data[0]:
                                Result.iat[i,9]='10'
                            elif '2勝' in data[0]:
                                Result.iat[i,9]='11'
                            elif '3勝' in data[0]:
                                Result.iat[i,9]='12'
                            elif '500' in data[0]:
                                Result.iat[i,9]='10'
                            elif '1000' in data[0]:
                                Result.iat[i,9]='11'
                            elif '1600' in data[0]:
                                Result.iat[i,9]='12'
                            else:
                                Result.iat[i,9]='-1'
                        else:
                            Result.iat[i,8]="'"+parts[j].text+"'"
                            if '1勝' in parts[j].text:
                                Result.iat[i,9]='20'
                            elif '2勝' in parts[j].text:
                                Result.iat[i,9]='21'
                            elif '3勝' in parts[j].text:
                                Result.iat[i,9]='22'
                            elif '500' in parts[j].text:
                                Result.iat[i,9]='20'
                            elif '1000' in parts[j].text:
                                Result.iat[i,9]='21'
                            elif '1600' in parts[j].text:
                                Result.iat[i,9]='22'
                            elif '未勝利' in parts[j].text:
                                Result.iat[i,9]='23'
                            elif '新馬' in parts[j].text:
                                Result.iat[i,9]='0'
                            else:
                                Result.iat[i,9]='-2'
                    elif j<=11:
                        num=float(parts[j].text)
                        if j==9:
                            Result.iat[i,j+5]="{:.1f}".format(num)
                        else:
                            Result.iat[i,j+5]="{:.0f}".format(num)
                    elif j==12:#馬名
                        Result.iat[i,18]=Extract_from_slash(parts[j].find('a').get('href'))
                        Result.iat[i,17]="'"+list(filter(None,re.findall('[\u30A0-\u30FF]+',parts[j].text)))[0]+"'"
                    if flag:#ジョッキー以外
                        if j==13:#ジョッキー
                            Result.iat[i,19]="'"+parts[j].text.replace('\n','')+"'"
                            Result.iat[i,20]=Extract_from_slash(parts[j].find('a').get('href'))
                            if len(Result.iat[i,20])==0:
                                Result.iat[i,0]='-10'
                                break
                        elif j==14:#斤量
                            Result.iat[i,21]=parts[j].text
                            if len(list(filter(None,re.findall('(\d*)',parts[j].text))))==0:
                                Result.iat[i,21]='-1'
                        elif j==15:#芝/ダート　距離
                            Type=re.findall('[^(\d*)]',parts[j].text)[0]
                            if '障' in Type:
                                Result.iat[i,22]='2'
                            elif '芝' in Type:
                                Result.iat[i,22]='0'
                            elif 'ダ' in Type:
                                Result.iat[i,22]='1'
                            Result.iat[i,23]=re.findall('[^(\d*)](\d*)$',parts[j].text)[0]
                        elif j==16:#馬場コンディション
                            if '稍' in parts[j].text:
                                Result.iat[i,24]='1'
                            elif '重' in parts[j].text:
                                Result.iat[i,24]='2'
                            elif '不' in parts[j].text:
                                Result.iat[i,24]='3'
                            else:
                                Result.iat[i,24]='0'
                        elif j==17:#タイム
                            d0=parts[j].text.find(':')
                            if d0>0:
                                Result.iat[i,25]="{:.1f}".format(float(parts[j].text[:d0])*60+float(parts[j].text[d0+1:]))
                            else:
                                num=float(parts[j].text)
                                Result.iat[i,25]="{:.1f}".format(num)
                        elif j==18:#着差
                            Result.iat[i,26]="{:.1f}".format(float(parts[j].text))                               
                        elif j==19:#コーナー通過
                            data=parts[j].text.split('-')
                            data=[str(int(k)) for k in data]
                            Result.iat[i,27]=data[0]
                            Result.iat[i,28]=data[-1]
                            Result.iat[i,29]="{:.1f}".format(float(sum([int(k) for k in data])/len(data)))
                        elif j==20:#ペース
                            data=parts[j].text.split('-')
                            data=[str(float(k)) for k in data]
                            Result.iat[i,30]=data[0]
                            if int(Result.iat[i,16])==-10:
                                Result.iloc[i,31:33]='-1'
                            else:
                                Result.iat[i,31]=data[-1]
                                Result.iat[i,32]="{:.1f}".format(float(data[0])-float(data[-1]))
                        elif j==21:#上り
                            Result.iat[i,33]="{:.1f}".format(float(parts[j].text))
                            data=re.findall('\d',parts[j].get('class')[0])
                            if len(data)>0:
                                Result.iat[i,34]=data[0]
                            else:
                                Result.iat[i,34]='4'
                        elif j==22:
                            Result.iat[i,35]=str(int(re.findall('^(\d*)',parts[j].text)[0]))
                            Result.iat[i,36]=str(int(re.findall('(\d*)\)$',parts[j].text)[0]))
                    else:#ジョッキー
                        if j==13:#斤量
                            Result.iat[i,19]=parts[j].text
                            if len(list(filter(None,re.findall('(\d*)',parts[j].text))))==0:
                                Result.iat[i,19]='-1'
                        elif j==14:#芝/ダート　距離
                            Type=re.findall('[^(\d*)]',parts[j].text)[0]
                            if '障' in Type:
                                Result.iat[i,20]='2'
                            elif '芝' in Type:
                                Result.iat[i,20]='0'
                            elif 'ダ' in Type:
                                Result.iat[i,20]='1'
                            Result.iat[i,21]=re.findall('[^(\d*)](\d*)$',parts[j].text)[0]
                        elif j==15:
                            if '稍' in parts[j].text:
                                Result.iat[i,22]='1'
                            elif '重' in parts[j].text:
                                Result.iat[i,22]='2'
                            elif '不' in parts[j].text:
                                Result.iat[i,22]='3'
                            else:
                                Result.iat[i,22]='0'
                        elif j==16:#タイム
                            d0=parts[j].text.find(':')
                            if d0>0:
                                Result.iat[i,23]="{:.1f}".format(float(parts[j].text[:d0])*60+float(parts[j].text[d0+1:]))
                            else:
                                Result.iat[i,23]="{:.1f}".format(float(parts[j].text))
                        elif j==17:#着差
                            Result.iat[i,24]="{:.1f}".format(float(parts[j].text))
                        elif j==18:#コーナー通過順位
                            data=parts[j].text.split('-')
                            data=[str(int(k)) for k in data]
                            Result.iat[i,25]=data[0]
                            Result.iat[i,26]=data[-1]
                            Result.iat[i,27]="{:.1f}".format(float(sum([int(k) for k in data])/len(data)))
                        elif j==19:#ペース
                            data=parts[j].text.split('-')
                            data=[str(float(k)) for k in data]
                            Result.iat[i,28]=data[0]
                            if int(Result.iat[i,16])==-10:
                                Result.iat[i,29:31]='-1'
                            else:
                                Result.iat[i,29]=data[-1]
                                Result.iat[i,30]="{:.1f}".format(float(data[0])-float(data[-1]))
                        elif j==20:#上り
                            Result.iat[i,31]="{:.1f}".format(float(parts[j].text))
                            data=re.findall('\d',parts[j].get('class')[0])
                            if len(data)>0:
                                Result.iat[i,32]=data[0]
                            else:
                                Result.iat[i,32]='4'
                        elif j==21:
                            Result.iat[i,33]=str(int(re.findall('^(\d*)',parts[j].text)[0]))
                            Result.iat[i,34]=str(int(re.findall('(\d*)\)$',parts[j].text)[0]))
                except:
                    if j==1:#開催が空のとき、海外など
                        Result.iat[i,4]='-1'
                    elif j<=11:
                        Result.iat[i,j+5]='-1'
                        if j==11:
                            if '中' in parts[j].text:
                                Result.iat[i,j+5]='-10'
                            else:
                                dd=list(filter(None,re.findall('(\d*)',parts[j].text)))
                                if len(dd)>0:
                                    Result.iat[i,j+5]=dd[0]
                    elif j==12:
                        Result.iat[i,17]="'"+parts[j].text.replace('\n','')+"'"
                    if flag:
                        if j==17:
                            Result.iat[i,25]='-1'
                        elif j==18:
                            Result.iat[i,26]='100'
                        elif j==19:
                            Result.iloc[i,27:30]='-1'
                        elif j==20:
                            Result.iloc[i,30:33]='-1'
                        elif j==21:
                            Result.iloc[i,33:35]='-1'
                        elif j==22:
                            Result.iloc[i,35:37]='-1'
                    else:
                        if j==16:
                            Result.iat[i,23]='-1'
                        elif j==17:
                            Result.iat[i,24]='100'
                        elif j==18:
                            Result.iloc[i,25:28]='-1'
                        elif j==19:
                            Result.iloc[i,28:31]='-1'
                        elif j==20:
                            Result.iloc[i,31:33]='-1'
                        elif j==21:
                             Result.iloc[i,33:35]='-1'
                             
    return Jid,RPY,persona,Result,browser
                    
def GetYearlyJockey(browser,link):
    browser,soup=Getsource_fromPage(browser,link)
    
    Table=soup.find_all('table',attrs={'class':'nk_tb_common race_table_01'})
    if len(Table)==0:
        yearlyData=pd.DataFrame(np.full((1,1),-1),dtype='object',columns=['year'])
        return yearlyData,browser
    
    Table=Table[0].find_all('tr')
    if len(Table)>2:
        Dlen=len(Table)-2
        yearlyData=pd.DataFrame(np.full((Dlen,20),-1),dtype='object',columns=['year','ranking','no1','no2','no3','others','graded_num',\
                                                                            'graded_winning','open_num' ,'open_winning','normal_num',\
                                                                            'normal_winning','turf_num','turf_winning','dirt_num','dirt_winning',\
                                                                            'winning_rate','second_rate','third_rate','total_prize'])        
        for j in range(2,Dlen+2):
            contents=Table[j].find_all('td')
            for k in range(20):
               if j==2 and k<=1:
                   yearlyData.iat[j-2,k]='0'
               else:
                   try:
                       data=float(contents[k].text.replace(',',''))
                       if k in range(16,19):
                           yearlyData.iat[j-2,k]="{:.1f}".format(data*100)
                       else:
                           yearlyData.iat[j-2,k]="{:.0f}".format(data)
                   except Exception:
                       yearlyData.iat[j-2,k]='-1'
    else:
       yearlyData = pd.DataFrame(np.full((1,1),-1), dtype='object', columns=['year'])
                   
    return yearlyData, browser

def GetJockeyProfile(browser, link, flag):
    browser, soup = Getsource_fromPage(browser, link)
    
    if flag == 1:#調教師
        persona = pd.DataFrame(np.full((1,15), '-1'), dtype = 'object',
                               columns=['Trainer_ID','Trainer_Name','Birthday','EW','hometown','deview',\
                                        'G1_win_num','Grade_win_num' ,'first','first_win','Grade_1st',\
                                        'Glade_1st_win','G1_1st','G1_1st_win','page'])
    elif flag == 2:#オーナー
        persona = pd.DataFrame(np.full((1,12),'-1'), dtype='object',
                               columns=['Owner_ID','Owner_Name','color','pattern','sleeve','G1_win_num',\
                                        'Grade_win_num','Glade_1st','Glade_1st_win','G1_1st','G1_1st_win','page'])
    elif flag == 3:#生産者
        persona = pd.DataFrame(np.full((1,10),'-1'), dtype = 'object', 
                               columns = ['Breeder_ID','Breeder_Name','base','G1_win_num',\
                                          'Grade_win_num','Glade_1st','Glade_1st_win','G1_1st','G1_1st_win','page'])
    else:#ジョッキー
        persona = pd.DataFrame(np.full((1,20),'-1'), dtype='object',
                               columns = ['Jockey_ID','Jockey_Name','Birthday','EW','Trainer_ID',
                                          'Trainer_Name','hometown','blood_type','height','weight',
                                          'deview','G1_win_num','Grade_win_num' ,'first','first_win',
                                          'Grade_1st','Glade_1st_win','G1_1st','G1_1st_win','page'])
        
    #persona.iat[0,0]=Extract_from_slash(link)
    persona.iat[0,-1] = 0#resultを読んだページ数
    header = soup.find('div', attrs = {'class':'db_head_name fc'})
    try:
        persona.iat[0,1] = chr(34) + re.findall('\((.*)\)', header.find('h1').text)[0] + chr(34)
    except:
        persona.iat[0,1] = chr(34) + header.find('h1').text.replace('\n','').replace('\xa0','').replace('  ','')+chr(34)
        if " &amp;" in persona.iat[0,1]:
            persona.iat[0,1] = persona.iat[0,1].replace(" &amp;","")
    
    if flag <= 1:
        prof = header.find('p',attrs = {'class':'txt_01'})
        birthday = prof.text.split('/')
        if len(birthday) > 1:
            bday = ""
            for i in range(3):
                num = list(filter(None, re.findall('(\d*)', birthday[i])))
                bday = bday + num[0]
            persona.iat[0,2] = bday
            if '海外' in birthday[2]: persona.iat[0,3] = '2'
            elif '栗東' in birthday[2]: persona.iat[0,3] = '0'
            elif '美浦' in birthday[2]: persona.iat[0,3] = '1'
    
        if flag == 0:#ジョッキーの場合
            if len(prof.find_all('a')) > 0:
                persona.iat[0,4] = Extract_from_slash(prof.find('a').get('href'))
            else: persona.iat[0,4] = '-1'
               
            if "地方" in prof.text: persona.iat[0,5] = chr(34) + "地方" + chr(34)
            elif "フリー" in prof.text: persona.iat[0,5] = chr(34) + "フリー" + chr(34)
            elif "\]" in prof.text:
                persona.iat[0,5] = "'" + re.findall('\](.*)', prof.text)[0].replace('\xa0', '') + "'"
                
    prof = soup.find_all('table',attrs={'class':'nk_tb_common race_table_01'})
    if len(prof) > 0:
        prof = prof[0].find_all('tr')
        for con in prof:
            pr = con.find('td').text
            ph = con.find('th').text
            if flag == 0:#ジョッキー
                if '出身地' in ph:#出身地と血液型
                    pr = pr.split('/')
                    if len(pr)==2:
                        persona.iat[0,6] = "'" + pr[0] + "'"
                        if 'O' in pr[1]: persona.iat[0,7] = '0'
                        elif 'AB' in pr[1]: persona.iat[0,7] = '1'
                        elif 'B' in pr[1]: persona.iat[0,7] = '2'
                        elif 'A' in pr[1]: persona.iat[0,7] = '3'
                elif '身長' in ph:
                    pr = pr.split('/')
                    if len(pr) == 2:
                        try:
                            int(pr[0].replace('cm',''))
                            persona.iat[0,8] = pr[0].replace('cm','')
                        except: pass
                        try:
                            int(pr[1].replace('kg',''))
                            persona.iat[0,9] = pr[1].replace('kg','')
                        except: pass
                elif 'デビュー' in ph:
                    if len(re.findall('(\d*)年',pr)) > 0:
                        persona.iat[0,10] = re.findall('(\d*)年',pr)[0]
                elif 'GI勝利数' in ph:
                    if len(re.findall('^(\d*)勝\(',pr)) > 0:
                        persona.iat[0,11] = re.findall('^(\d*)勝\(', pr)[0]
                elif '重賞勝利数' in ph:
                    if len(re.findall('^(\d*)勝\(',pr)) > 0:
                        persona.iat[0,12] = re.findall('^(\d*)勝\(', pr)[0]
            elif flag == 1:#調教師
                if '出身地' in ph:
                    if not "-" in pr: persona.iat[0,4] = "'" + pr + "'"
                elif 'デビュー' in ph:
                    if len(re.findall('(\d*)年',pr)) > 0:
                        persona.iat[0,5] = re.findall('(\d*)年', pr)[0]
                elif 'GI勝利数' in ph:
                    if len(re.findall('^(\d*)勝\(',pr)) > 0:
                        persona.iat[0,6] = re.findall('^(\d*)勝\(', pr)[0]
                elif '重賞勝利数' in ph:
                    if len(re.findall('^(\d*)勝\(',pr))>0:
                        persona.iat[0,7] = re.findall('^(\d*)勝\(', pr)[0]
                elif '初出走' in ph: persona.iat[0,8] = split_merge(pr)
                elif '初勝利' in ph: persona.iat[0,9] = split_merge(pr)
                elif '初重賞出走' in ph: persona.iat[0,10] = split_merge(pr)
                elif '初重賞勝利' in ph: persona.iat[0,11] = split_merge(pr)
                elif '初G1出走' in ph: persona.iat[0,12] = split_merge(pr)
                elif '初G1勝利' in ph: persona.iat[0,13] = split_merge(pr)
            elif flag == 2:#オーナー
                if '勝負服' in ph:
                    if pr.find("，") > 0:
                        cloth = pr.split('，')
                        cloth = [chr(34) + k + chr(34) for k in cloth]
                        persona.iloc[0, 2:2 + len(cloth)] = cloth
                elif 'GI勝利数' in ph:
                    if len(re.findall('^(\d*)勝\(',pr)) > 0:
                        persona.iat[0,5] = re.findall('^(\d*)勝\(',pr)[0]
                elif '重賞勝利数' in ph:
                    if len(re.findall('^(\d*)勝\(',pr)) > 0: 
                        persona.iat[0,6] = re.findall('^(\d*)勝\(',pr)[0]
                elif '初重賞出走' in ph: persona.iat[0,7] = split_merge(pr)
                elif '初重賞勝利' in ph: persona.iat[0,8] = split_merge(pr)
                elif '初G1出走' in ph:  persona.iat[0,9] = split_merge(pr)
                elif '初G1勝利' in ph: persona.iat[0,10] = split_merge(pr)
            elif flag==3:#生産者
                if '所在地' in ph:
                    if '-' in pr:
                        continue
                    index=persona.iat[0,1].find(pr)
                    if index>0:
                        persona.iat[0,1]=persona.iat[0,1][:index]+chr(34)
                    persona.iat[0,2]=chr(34)+pr+chr(34)
                elif 'GI勝利数' in ph:
                    if len(re.findall('^(\d*)勝\(',pr))>0:
                        persona.iat[0,3]=re.findall('^(\d*)勝\(',pr)[0]
                elif '重賞勝利数' in ph:
                    if len(re.findall('^(\d*)勝\(',pr))>0:
                        persona.iat[0,4]=re.findall('^(\d*)勝\(',pr)[0]
                elif '初重賞出走' in ph:
                    persona.iat[0,5]=split_merge(pr)
                elif '初重賞勝利' in ph:
                    persona.iat[0,6]=split_merge(pr)
                elif '初G1出走' in ph:
                    persona.iat[0,7]=split_merge(pr)
                elif '初G1勝利' in ph:
                    persona.iat[0,8]=split_merge(pr)
    
    if flag==0:
        prof=soup.find_all('table',attrs={'class':'nk_tb_common race_table_01 tbl_profile_first_time'})
        if len(prof)>0:
            prof=prof[0].find_all('tr')
            for con in prof:
                if len(con.find_all('td'))==0:
                    continue
                pr=con.find('td').text
                ph=con.find('th').text
                if '初出走' in ph:
                    persona.iat[0,13]=split_merge(pr)
                elif '初勝利' in ph:
                    persona.iat[0,14]=split_merge(pr)
                elif '初重賞出走' in ph:
                    persona.iat[0,15]=split_merge(pr)
                elif '初重賞勝利' in ph:
                    persona.iat[0,16]=split_merge(pr)
                elif '初G1出走' in ph:
                    persona.iat[0,17]=split_merge(pr)
                elif '初G1勝利' in ph:
                    persona.iat[0,18]=split_merge(pr)
    return persona,browser

def split_merge(text):
    con=text.split('/')
    if len(con)<=1:
        return '-1'
    ret=""
    for i in range(len(con)):
        num=list(filter(None,re.findall('(\d*)',con[i])))
        ret=ret+num[0]
    return ret

def GetHorseData(browser,link):
    browser,soup=Getsource_fromPage(browser,link)
    
    Head=soup.find('div',attrs={'class':'db_head_name fc'})
    Profile=soup.find('div',attrs={'class':'db_main_deta'})
    persona=pd.DataFrame(np.full((1,37),'=1'),dtype='object',columns=['Horse_ID','Horse_Name','active','sex','age','kind','year',\
                                                                    'month','day','Trainer_Name','Trainer_ID','belonging','Owner_Name',\
                                                                    'Owner_ID','Breeder_Name','Breeder_ID','hometown','price','prize',\
                                                                    'no1','no2','no3','others','title','f','m','ff','fm','mf','mm',\
                                                                    'fID','mID','ffID','fmID','mfID','mmID','Local'])
    
    persona.iat[0,0]=Extract_from_slash(link)
    title=Head.find('div',attrs={'class':'horse_title'})
    if len(list(filter(None,re.findall('[\u30A0-\u30FF]+',title.find('h1').text))))>0:
        persona.iat[0,1]="'"+list(filter(None,re.findall('[\u30A0-\u30FF]+',title.find('h1').text)))[0]+"'"
    else:
        persona.iat[0,1]="'"+list(filter(None,re.findall('[0-9a-zA-Z]+',title.find('h1').text)))[0]+"'"
    if "地" in title.find('h1').text:
        persona.iat[0,36]='1'
    elif "外" in title.find('h1').text:
        persona.iat[0,36]='2'
    else:
        persona.iat[0,36]='0'
    attr=title.find('p',attrs={'class','txt_01'}).text
    #現役馬か引退馬か
    if '現' in attr:
        persona.iat[0,2]='1'
    else:
        persona.iat[0,2]='0'

    if '牝' in attr:
        persona.iat[0,3]='0'
    elif '牡' in attr:
        persona.iat[0,3]='1'
    else:
        persona.iat[0,3]='2'
    
    if '黒鹿毛' in attr:
        persona.iat[0,5]='1'
    elif '青鹿毛' in attr:
        persona.iat[0,5]='2'
    elif '青毛' in attr:
        persona.iat[0,5]='3'
    elif '栗毛' in attr:
        persona.iat[0,5]='4'
    elif '栃栗毛' in attr:
        persona.iat[0,5]='5'
    elif '芦毛' in attr:
        persona.iat[0,5]='6'
    elif '白毛' in attr:
        persona.iat[0,5]='7'
    elif '鹿毛' in attr:
        persona.iat[0,5]='0'
    else:
        persona.iat[0,5]='-1'
        
    Profile = Profile.find('div',attrs={'class':'db_prof_area_02'})
    prof = Profile.find_all('table')[0]
    for j in prof.find('tbody').find_all('tr'):
        ps=j.find('td').text
        ty=j.find('th').text
        if '生年月日' in ty:
            elem=list(filter(None,re.findall('(\d*)',ps)))
            persona.iloc[0,6:9]=elem
            #馬齢は今と生まれ年の差で判断(抹消馬だとプロフィール欄に表示されないため)
            persona.iat[0,4]=str(dt.now().year-int(elem[0]))
        elif '調教師' in ty:
            if len(re.findall('^(.*)\(',ps))==0:
                persona.iat[0,9]="-1"
            else:
                persona.iat[0,9]="'"+re.findall('^(.*)\(',ps)[0].replace(' ','')+"'"
            try:
                persona.iat[0,10]=Extract_from_slash(j.find('td').find('a').get('href'))
                judge=re.findall('\((.*)\)$',ps)[0]
                if '栗東' in judge:
                    persona.iat[0,11]='0'
                elif '美浦' in judge:
                    persona.iat[0,11]='1'
                elif '海外' in judge:
                    persona.iat[0,11]='2'
                else:
                    persona.iat[0,11]='-1'
            except:
                persona.iloc[0,10:12]='-1'
        elif '馬主' in ty:
            persona.iat[0,12]=chr(34)+j.find('td').text+chr(34)
            try:
                persona.iat[0,13]=Extract_from_slash(j.find('td').find('a').get('href'))
            except:
                persona.iat[0,13]='-1'
        elif '生産者' in ty:
            persona.iat[0,14]=chr(34)+j.find('td').text+chr(34)
            try:
                persona.iat[0,15]=Extract_from_slash(j.find('td').find('a').get('href'))
            except:
                persona.iat[0,15]='-1'
        elif '産地' in ty:
            persona.iat[0,16]="'"+ps+"'"
        elif '価格' in ty:
            km=re.findall('^(.*)万円',ps)
            if len(km)>0:
                ko=re.findall('^(\d*)億',km[0])
                if len(ko)>0:
                    km=re.findall('億(.*)',km[0])
                    persona.iat[0,17] = str(10000*int(ko[0])+int(km[0].replace(',','')))
                else:
                    persona.iat[0,17] = km[0].replace(',','')
            else:
                persona.iat[0,17] = '-1'
        elif '賞金' in ty:
            km=list(filter(None,re.findall('(.*)\(中央',ps)))
            if len(km):
                km=list(filter(None,re.findall('(.*)万円',km[0])))
                if len(km): 
                    ko=list(filter(None,re.findall('(.*)億',km[0])))
                    if len(ko)>0:
                        km=re.findall('億(.*)',km[0])
                        persona.iat[0,18]=str(10000*int(ko[0])+int(km[0].replace(',','')))
                    else:
                        persona.iat[0,18]=km[0].replace(',','')
                else:
                    persona.iat[0,18]='-1'
            else:
                km=list(filter(None,re.findall('(.*)\(地方',ps)))
                if len(km):
                    persona.iat[0,18]='-1'
                else:
                    persona.iat[0,18]='0'
        elif '通算成績' in ty:
            kk=re.findall('\[(.*)\]',ps)
            if len(kk)>0:
                kk=kk[0].split('-')
                if len(kk) !=4:
                    persona.iloc[0,19:23]=np.tile('-1',(1,4))
                else:
                    persona.iloc[0,19:23]=kk
            else:
                persona.iloc[0,19:23]=np.tile('-1',(1,4))            
        elif '勝鞍' in ty:
            kk=re.findall('\((.*)\)',ps)
            if len(kk)==0:
                persona.iat[0,23]='0'
            else:
                if 'OP' in kk[0]:
                    persona.iat[0,23]='1'
                elif 'L' in kk[0]:
                    persona.iat[0,23]='2'
                elif 'G3' in kk[0]:
                    persona.iat[0,23]='3'
                elif 'G2' in kk[0]:
                    persona.iat[0,23]='4'
                elif 'G1' in kk[0]:
                    persona.iat[0,23]='5'
                else:
                    persona.iat[0,23]='-1'
    
    #血統テーブルの抽出
    Blood=Profile.find_all('table')[1].find_all('tr')
    for j in range(len(Blood)):
        elem=Blood[j].find_all('td')
        cnt=0
        for el in elem:
            idd=Extract_from_slash(el.find('a').get('href'))
            if j==0:
                if cnt:
                    persona.iat[0,26]=chr(34)+el.text.replace('\n','')+chr(34)
                    persona.iat[0,32]=idd
                else:
                    persona.iat[0,24]=chr(34)+el.text.replace('\n','')+chr(34)
                    persona.iat[0,30]=idd
            elif j==1:
                persona.iat[0,27]=chr(34)+el.text.replace('\n','')+chr(34)
                persona.iat[0,33]=idd
            elif j == 2:
                if cnt:
                    persona.iat[0,28] = chr(34)+el.text.replace('\n','')+chr(34)
                    persona.iat[0,34] = idd
                else:
                    persona.iat[0,25]=chr(34)+el.text.replace('\n','')+chr(34)
                    persona.iat[0,31]=idd
            elif j==3:
                persona.iat[0,29]=chr(34)+el.text.replace('\n','')+chr(34)
                persona.iat[0,35]=idd
            cnt=1

    #過去レース結果の抽出
    Table = soup.find('table',attrs={'class':'db_h_race_results nk_tb_common'})
    if Table is not None:
        Table = Table.find('tbody').find_all('tr')
        RaceNum = len(Table)
        Result = pd.DataFrame(np.full((RaceNum,37),'-1'),dtype='object',columns=['year','month','day','kai','track','date','weather','round',\
                                                                           'Race_Name','Race_rank','Race_ID','total_num','waku','horse_num',\
                                                                           'winning_Odds','favorite_rank','result','Jockey_Name','Jockey_ID',\
                                                                           'weight_to_carry','type','distance','cond','Total_time',\
                                                                           'Diff_time','Init_corner','Fin_corner','Ave_pos','pace1st',\
                                                                           'pace2nd','Diff_pace','Fin3F','Fin3F_Rank','Horse_Weight',\
                                                                           'Weight_Change','intev','Day_ID'])  
             
        for i in range(RaceNum):
            parts=Table[i].find_all('td')
            for j in range(len(parts)):
                try:
                    if j in [3,5,16,19]:
                        continue
                    elif j==24:
                        break
                    elif j==0:
                        Result.iloc[i,:3]=list(map(lambda x : str(int(x)),parts[j].text.split('/')))
                        Result.iat[i,-1]="{}{:02d}{:02d}".format(Result.iat[i,0],int(Result.iat[i,1]),int(Result.iat[i,2]))
                    elif j==1:
                        Tr=int(re.findall('\/sum\/(\d*)\/',parts[j].find('a').get('href'))[0])
                        Result.iat[i,4]=str(Tr)
                        if len(re.findall('^(\d*)',parts[j].text)[0])>0:#中央
                            Result.iat[i,3]=re.findall('^(\d*)',parts[j].text)[0]
                            Result.iat[i,5]=re.findall('[^(\d*)](\d*)$',parts[j].text)[0]
                        else:#地方
                            Result.iat[i,3]='-1'
                            Result.iat[i,5]='-1'
                    elif j==2:
                        if '晴' in parts[j].text:
                            Result.iat[i,6]='0'
                        elif '曇' in parts[j].text:
                            Result.iat[i,6]='1'
                        elif '小雨' in parts[j].text:
                            Result.iat[i,6]='2'
                        elif '雨' in parts[j].text:
                            Result.iat[i,6]='3'
                        elif '小雪' in parts[j].text:
                            Result.iat[i,6]='4'
                        elif '雪' in parts[j].text:
                            Result.iat[i,6]='5'
                        elif len(parts[j].text)==0:
                            Result.iat[i,6]='-1'
                        else:
                            Result.iat[i,6]='6'
                    elif j==4:
                        data=re.findall('\((.*)\)$',parts[j].text)
                        Result.iat[i,10]=Extract_from_slash(parts[j].find('a').get('href'))
                        Result.iat[i,7]=Result.iat[i,10][-2:]
                        if len(data)>0:
                            Result.iat[i,8]="'"+re.findall('^(.*)\(',parts[j].text)[0]+"'"
                            if 'OP' in data[0]:
                                Result.iat[i,9]='1'
                            elif 'L' in data[0]:
                                Result.iat[i,9]='2'
                            elif 'G3' in data[0]:
                                Result.iat[i,9]='3'
                            elif 'G2' in data[0]:
                                Result.iat[i,9]='4'
                            elif 'G1' in data[0]:
                                Result.iat[i,9]='5'
                            elif '1勝' in data[0]:
                                Result.iat[i,9]='10'
                            elif '2勝' in data[0]:
                                Result.iat[i,9]='11'
                            elif '3勝' in data[0]:
                                Result.iat[i,9]='12'
                            elif '500' in data[0]:
                                Result.iat[i,9]='10'
                            elif '1000' in data[0]:
                                Result.iat[i,9]='11'
                            elif '1600' in data[0]:
                                Result.iat[i,9]='12'
                            else:
                                Result.iat[i,9]='-1'
                        else:
                            Result.iat[i,8]="'"+parts[j].text+"'"
                            if '1勝' in parts[j].text:
                                Result.iat[i,9]='20'
                            elif '2勝' in parts[j].text:
                                Result.iat[i,9]='21'
                            elif '3勝' in parts[j].text:
                                Result.iat[i,9]='22'
                            elif '500' in parts[j].text:
                                Result.iat[i,9]='20'
                            elif '1000' in parts[j].text:
                                Result.iat[i,9]='21'
                            elif '1600' in parts[j].text:
                                Result.iat[i,9]='22'
                            elif '未勝利' in parts[j].text:
                                Result.iat[i,9]='23'
                            elif '新馬' in parts[j].text:
                                Result.iat[i,9]='24'
                            elif 'OP' in parts[j].text:
                                Result.iat[i,9]='1'
                            else:
                                Result.iat[i,9]='-2'
                    elif j <= 11:
                        if j == 9:
                            Result.iat[i, j + 5] = "{:.1f}".format(float(parts[j].text))
                        else:
                            Result.iat[i,j + 5] = "{:d}".format(int(parts[j].text))
                    elif j == 12:
                        Result.iat[i,17] = "'"+parts[j].text.replace('\n','')+"'"
                        Result.iat[i,18] = Extract_from_slash(parts[j].find('a').get('href'))
                    elif j == 13:
                        Result.iat[i,19] = "{:.1f}".format(float(parts[j].text))
                    elif j == 14:
                        Type = re.findall('[^(\d*)]',parts[j].text)[0]
                        if '障' in Type:
                            Result.iat[i,20]='2'
                        elif '芝' in Type:
                            Result.iat[i,20]='0'
                        elif 'ダ' in Type:
                            Result.iat[i,20]='1'
                        Result.iat[i,21]=re.findall('[^(\d*)](\d*)$',parts[j].text)[0]
                    elif j==15:
                        if '稍' in parts[j].text:
                            Result.iat[i,22]='1'
                        elif '重' in parts[j].text:
                            Result.iat[i,22]='2'
                        elif '不' in parts[j].text:
                            Result.iat[i,22]='3'
                        elif '良' in parts[j].text:
                            Result.iat[i,22]='0'
                        else:
                            Result.iat[i,22]='-1'
                    elif j==17:
                        idd=parts[j].text.find(':')
                        if idd<0:
                            Result.iat[i,23] = "{:.1f}".format(float(parts[j].text))
                        else:
                            Result.iat[i,23] = "{:.1f}".format(float(parts[j].text[:idd])*60+float(parts[j].text[idd+1:]))
                    elif j==18:#着差
                        Result.iat[i,24]="{:.1f}".format(float(parts[j].text))
                    elif j==20:#コーナー通過
                        data=parts[j].text.split('-')
                        data=[int(k) for k in data]
                        if int(Result.iat[i,16])>0:
                            Result.iat[i,25]=str(data[0])
                            Result.iat[i,26]=str(data[-1])
                            Result.iat[i,27]="{:.1f}".format(sum(data)/len(data))
                        else:
                            Result.iat[i,25]=data[0]
                            Result.iloc[i,26:28]='-1'
                    elif j==21:#ペース
                        data=parts[j].text.split('-')
                        data=[float(k) for k in data]
                        Result.iat[i,28]="{:.1f}".format(data[0])
                        Result.iat[i,29]="{:.1f}".format(data[1])
                        Result.iat[i,30]="{:.1f}".format(data[0]-data[1])
                    elif j==22:#上り
                        Result.iat[i,31]="{:.1f}".format(float(parts[j].text))
                        data=re.findall('\d',parts[j].get('class')[0])
                        if len(data) > 0:
                            Result.iat[i,32] = data[0]
                        else:
                            Result.iat[i,32] = '4'
                    elif j == 23:
                        Result.iat[i,33] = re.findall('^\d*',parts[j].text)[0]
                        Result.iat[i,34] = re.findall('\((.*)\)$',parts[j].text)[0]
                        if len(Result.iat[i,33]) == 0:
                            Result.iat[i,33:35] = "-1"
                except Exception:
                    if j==1:
                        Result.iat[i,4] = '-1'
                    elif j<11:
                        Result.iat[i, j + 5] = '-1'
                    elif j==11:
                        if '中' in parts[j].text:
                            Result.iat[i, j + 5] = '-10'
                        else:
                            dd = list(filter(None, re.findall('(\d*)', parts[j].text)))
                            if len(dd) > 0:
                                Result.iat[i,j+5] = dd[0]
                    elif j == 13:
                        Result.iat[i,19] = '-1'
                    elif j == 14:
                        Result.iat[i,20:22] = '-1'
                    elif j == 17:
                        Result.iat[i,23] = '-1'
                    elif j == 18:
                        Result.iat[i,24] = '100'
                    elif j == 20:
                        Result.iloc[i,25:28] = '-1'
                    elif j == 21:
                        Result.iloc[i,28:31] = '-1'
                    elif j == 22:
                        Result.iloc[i,31:33] = '-1'
                    elif j == 23:
                        Result.iloc[i,33:35] = '-1'
    
        #前走からの間隔を追加
        date = np.zeros((RaceNum,3),dtype=int)
        for i in range(3):
            date[:,i] = list(map(lambda x : int(x),Result.iloc[:,i]))
        for i in range(RaceNum-1):       
            Result.iat[i,35] = (date[i,0]-date[i+1,0])*365 + (date[i,1] - date[i + 1,1])*30 + date[i,2] - date[i+1,2]
            Result.iat[i,35] = int(Result.iat[i,35]/7 - 0.5)
            if Result.iat[i,35] > 127: Result.iat[i,35] = 127
    
        Result.iat[-1,35] = -1
    else:
        Result = pd.DataFrame(np.full((1,37), -1), dtype = np.int8)
    return persona, Result, browser

def SetJockey2SQL(browser,YEAR,conn,curs,reff):
    Jid, RPY, persona, Result, browser = GetPastResult(browser, reff, YEAR, 0, 1)
    #カラムの作成
    #Jockey_Yearly
    if RPY.iat[0,0] != -1:
        curs.execute('USE Jockey_Yearly;')                            
        conn.commit()
        sql = 'CREATE TABLE IF NOT EXISTS `{}` ('.format(persona.iat[0,0])
        SetSQLstatemaent(sql,RPY,5,curs,conn)
        #データの挿入
        ret = Insert2SQL(persona.iat[0,0], RPY, curs, conn, 0)
        
    #Jocley_LIST
    if persona.iat[0,0] != -1:
        curs.execute('USE Jockey_LIST;')                            
        conn.commit()
        sql = 'CREATE TABLE IF NOT EXISTS `{}` ('.format(persona.iat[0,0])
        SetSQLstatemaent(sql,persona,3,curs,conn)
        #データの挿入
        ret=Insert2SQL(persona.iat[0,0],persona,curs,conn,0)
    
    try:
        if int(Jid) < 0:#地方所属ジョッキーの場合
            return browser
    except:
        pass
    #Jocley_Result
    if Result.iat[0,0] != -1:
        curs.execute('USE Jockey_Result;')                            
        conn.commit()
        sql = 'CREATE TABLE IF NOT EXISTS `{}` ('.format(Jid)
        #カラムの作成
        SetSQLstatemaent(sql, Result, 2, curs, conn)
        #データの挿入                          
        cnt = 1
        ret = Insert2SQL(Jid,Result,curs,conn,0)
        while ret:
            #次ページへ
            cnt = cnt + 1
            LINK = 'https://db.netkeiba.com/?pid=jockey_detail&id={}&page={:d}'.format(Jid,cnt)
            _, _, _, Result, browser = GetPastResult(browser, LINK, YEAR, 0, 0)
            ret = Insert2SQL(Jid, Result, curs, conn, 0)
            if  not cnt % 5:
                SavePage2SQL(cnt, Jid, conn, curs, 'jockey')
                
        SavePage2SQL(cnt, Jid, conn, curs, 'jockey')
        
    return browser
        
def SetTrainer2SQL(browser,YEAR,conn,curs,reff):
    Jid,RPY,persona,Result,browser=GetPastResult(browser,reff,YEAR,1,1)

    #Trainer_Yearly
    if RPY.iat[0,0]!=-1:
        curs.execute('USE Trainer_Yearly;')                            
        conn.commit()
        sql='CREATE TABLE IF NOT EXISTS `{}` ('.format(Jid)
        #カラムの作成
        SetSQLstatemaent(sql,RPY,5,curs,conn)
        #データの挿入
        ret=Insert2SQL(Jid,RPY,curs,conn,0)
        
    #Trainer_LIST
    try:
        if persona.iat[0,0]!=-1:
            curs.execute('USE Trainer_LIST;')                            
            conn.commit()
            sql='CREATE TABLE IF NOT EXISTS `{}` ('.format(Jid)
            SetSQLstatemaent(sql,persona,3,curs,conn)
            ret=Insert2SQL(Jid,persona,curs,conn,1)
    except:
        pass
    
    #Trainer_Result
    if Result.iat[0,0]!=-1:
        curs.execute('USE Trainer_Result;')                            
        conn.commit()
        sql='CREATE TABLE IF NOT EXISTS `{}` ('.format(Jid)
        SetSQLstatemaent(sql,Result,4,curs,conn)
        #データの挿入                          
        cnt=1#ページ数
        pp=1#カラムを作成しているか
        ret=1#所定の年度を超えていないか
        if sum(list(map(lambda x: 1 if int(x)>0 else 0 ,Result.iloc[:,0])))>0:
            pp=0
            sql='CREATE TABLE IF NOT EXISTS `{}` ('.format(Jid)
            SetSQLstatemaent(sql,Result,4,curs,conn)
            ret=Insert2SQL(Jid,Result,curs,conn,1)
        while ret:
            cnt=cnt+1
            LINK='https://db.netkeiba.com/?pid=trainer_detail&id={}&page={:d}'.format(Jid,cnt)
            dummy1,dummy2,dummy3,Result,browser=GetPastResult(browser,LINK,YEAR,1,0)
            if pp:
                if sum(list(map(lambda x: 1 if int(x)>0 else 0 ,Result.iloc[:,0])))>0:
                    pp=0
                    sql='CREATE TABLE IF NOT EXISTS `{}` ('.format(Jid)
                    SetSQLstatemaent(sql,Result,4,curs,conn)
            ret=Insert2SQL(Jid,Result,curs,conn,1)
            if  not cnt % 5:
                SavePage2SQL(cnt,Jid,conn,curs,'Trainer')
                
        SavePage2SQL(cnt, Jid, conn, curs, 'Trainer')
        
    return browser
        
def SetHorse2SQL(browser, YEAR, conn, curs, link):
    #馬のプロフィールと過去の戦績を取得   
    persona, Result, browser = GetHorseData(browser,link)
    ID = list(filter(None,re.findall('\/(\d*)',link)))[0]

    if persona.iat[0,0] != -1:
        curs.execute('USE Horse_prof;')
        conn.commit()
        sql='CREATE TABLE IF NOT EXISTS `{}` ('.format(ID)
        SetSQLstatemaent(sql,persona,1,curs,conn)
        ret=Insert2SQL(ID,persona,curs,conn,0)

    if Result.iat[0,0]!=-1:
        curs.execute('USE Horse_result;')
        conn.commit()
        sql='CREATE TABLE IF NOT EXISTS `{}` ('.format(ID)
        SetSQLstatemaent(sql,Result,2,curs,conn)
        ret=Insert2SQL(ID,Result,curs,conn,0)
                    
    '''#try:
        #生産者データを取得
        Bid=persona.iat[0,15]#生産者ID
        link=r"https://db.netkeiba.com/breeder/result/recent/"+Bid.replace("'","")
        sql="SELECT 1 FROM breeder_List.`{}` LIMIT 1;".format(Bid)
        curs.execute(sql)#例外発生:初登場
        conn.rollback()
        
        dummy,RPY,dummy1,Result,browser=GetPastResult(browser,link,YEAR,2,1)
        
        try:
            int(Bid)
            Bid="`"+str(Bid)+"`"
        except:
            pass
        
        sql="SELECT Day_ID FROM breeder_result.{} ORDER BY Day_ID DESC LIMIT 1;".format(Bid)
        curs.execute(sql)
        day_id=curs.fetchall()
        day_id[0][0]#例外発生：resultがないケース
        if day_id<int(Result.iat[0,-1]):
            
            curs.execute("USE breeder_yearly;")
            conn.commit()
            curs.execute("DROP TABLE {};".format(Bid))
            conn.commit()
            sql='CREATE TABLE {} ('.format(Bid)
            SetSQLstatemaent(sql,RPY,5,curs,conn)
            ret=Insert2SQL(Bid,RPY,curs,conn,0)
            
            cnt=1
            curs.execute("USE breeder_result;")
            conn.commit()
            while 1:
                array=np.array(list(map(lambda x:float(x),Result.iloc[:,-1])))
                array=np.where(array>day_id)[0]
                if len(array)==0:
                    break
                elif len(array)<len(Result):
                    Result=Result.iloc[np.where(array>day_id)[0],:]
                    ret=Insert2SQL(Bid,Result,curs,conn,1)
                    break
                else:
                    Result=Result.iloc[np.where(array>day_id)[0],:]
                    ret=Insert2SQL(Bid,Result,curs,conn,1)
                
                if ret==0:
                    break
                cnt+=1
                link='https://db.netkeiba.com/?pid=breeder_detail&id={}&page={:d}'.format(persona.iat[0,15],cnt)
                dummy,dummy1,dummy2,Result,browser=GetPastResult(browser,link,YEAR,2,0)        
            
    except mysql.connector.Error:#MYSQLにないテーブルだと例外が発生する
        if Bid != '-1':
            browser=SetOwner2SQL(browser,YEAR,conn,curs,link,'Breeder')
    except Exception:#listはあるがResultがないケース
        if Bid != '-1':
            browser=SetOwner2SQL(browser,YEAR,conn,curs,link,'Breeder')''' 
    
    return browser
        
def SetOwner2SQL(browser,YEAR,conn,curs,reff,dis):
    
    if dis == 'Breeder':
        flag=3
    else:
        flag=2
        
    Jid,RPY,persona,Result,browser=GetPastResult(browser,reff,YEAR,flag,1)
    #Owner_Yearly
    if float(RPY.iat[0,0]) != -1:
        sql='USE {}_Yearly;'.format(dis)
        curs.execute(sql)                            
        conn.commit()
        sql='CREATE TABLE IF NOT EXISTS `{}` ('.format(Jid)
        SetSQLstatemaent(sql,RPY,5,curs,conn)
        #データの挿入
        ret=Insert2SQL(Jid,RPY,curs,conn,0)
        
    #Owner_List
    try:
        if float(persona.iat[0,0]) != -1:
            sql='USE {}_List;'.format(dis)
            curs.execute(sql)                            
            conn.commit()
            sql='CREATE TABLE IF NOT EXISTS `{}` ('.format(Jid)
            SetSQLstatemaent(sql,persona,3,curs,conn)
            ret=Insert2SQL(Jid,persona,curs,conn,0)
    except:
        pass
    
    #Owner_Result
    if float(Result.iat[0,0]) != -1:
        sql='USE {}_Result;'.format(dis)
        curs.execute(sql)                            
        conn.commit()
        cnt, pp = 1, 1#ページ数, カラムを作成しているか
        
        while 1:
            
            if sum(list(map(lambda x: 1 if int(x)>0 else 0 ,Result.iloc[:,0])))>0 & pp:
                pp=0
                sql='CREATE TABLE IF NOT EXISTS `{}` ('.format(Jid)
                SetSQLstatemaent(sql,Result,4,curs,conn)                       
            
            ret=Insert2SQL(Jid,Result,curs,conn,1)
            if not ret: break
        
            cnt += 1
            LINK='https://db.netkeiba.com/?pid={}_detail&id={}&page={:d}'.format(dis.lower(),Jid,cnt)
            dummy1,dummy2,dummy3,Result,browser = GetPastResult(browser,LINK,YEAR,2,0)

            if  not cnt % 5:
                SavePage2SQL(cnt,Jid,conn,curs,dis)
                
        SavePage2SQL(cnt,Jid,conn,curs,dis)
        
    return browser  

def Extract_from_slash(string):
    strlen=len(string)
    in1=string.rfind('/',0, strlen-1)
    in2=string.rfind('/',0, strlen)
    if in1==in2:
        idd=string[in1+1:strlen]
    else:
        idd=string[in1+1:strlen-1]
    return idd
    #return "'"+idd+"'"
 
#SQL容量削減のために文字列長さを取得
#flag: カラム作成=0, データ挿入=1
def DefNameLength(df,flag):
    df00=pd.DataFrame(np.full((2,12),0),dtype='object',\
                      columns=['Race','Jockey','Trainer','Horse','Owner','Breeder','f','m','ff','fm','mf','mm']\
                      ,index=['ID','Name'])
    df01=df.filter(items=["track","Race_rank"])
    if df01.empty:
        ddf=range(len(df))
    else:
        ddf=[]
        for k in range(len(df)):
            if int(df.iat[k,0])>0:
                ddf.append(k)
    if len(ddf)==0:
        return pd.DataFrame(data=[-1])
    else:
        ddf=range(len(df))
        
    for k1 in range(12):
        if k1<=5:
            for k2 in range(2):
                value=df.filter(items=[df00.columns[k1]+"_"+df00.index[k2]])
                if not value.empty:
                    value=value.filter(items=ddf,axis=0)
                    try:#int型に変換
                        value=list(map(lambda x : int(x) if type(x) is str else 0,value.iloc[:,0]))
                        df00.iat[k2,k1]=Num2Index(max(value))
                    except:#文字が含まれている場合
                        if len(value)>1:
                            value=list(map(lambda x : len(x) if type(x) is str else 0,value.iloc[:,0]))
                            value=max(value)
                        else:
                            try:
                                value=len(value.iat[0,0])
                            except Exception as e:
                                print(e)
                        if value>0:
                            df00.iat[k2,k1]='VARCHAR({:d})'.format(value)
                                    
        else:
            value=df.filter(items=[df00.columns[k1]+"ID"])
            if not value.empty:
                try:
                    df00.iat[0,k1]=Num2Index(int(value.iat[0,0]))
                except:
                    df00.iat[0,k1]='VARCHAR({:d})'.format(len(value.iat[0,0]))
                
                value=df.filter(items=[df00.columns[k1]])
                df00.iat[1,k1]='VARCHAR({:d})'.format(len(value.iat[0,0]))
    
    return df00                

def Num2Index(value):
    if value<=127:
        val='TINYINT'
    elif value<=32767:
        val='SMALLINT'
    elif value<=8388607:
        val='MEDIUMINT'
    elif value<=2147483647:
        val='INT'
    else:
        val='BIGINT'
    return val