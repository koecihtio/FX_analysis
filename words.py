import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import numpy as np
import pandas as pd
import time
from dateutil.relativedelta import relativedelta
from datetime import datetime,date, timedelta,timezone
import os
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from numpy.random import *
import seaborn as sns
from sklearn import preprocessing
import config
from twitter import Twitter, OAuth
import tweepy
import pytz
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials 

def content_corrlation():
    today=datetime.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=3)
    
    
    start= three_month_ago 
    end=today
    
    dow=pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    usdjpy =pd.DataFrame(np.log(pdr.DataReader("DEXJPUS","fred",start,end)))
    nikkei=pd.DataFrame(np.log(pdr.DataReader("NIKKEI225","fred",start,end)))
    btc=pd.DataFrame(np.log(pdr.DataReader("CBBTCUSD","fred",start,end)))
    eurusd =pd.DataFrame(np.log(pdr.DataReader("DEXUSEU","fred",start,end)))
    gbpusd =pd.DataFrame(np.log(pdr.DataReader("DEXUSUK","fred",start,end)))
    usdcad =pd.DataFrame(np.log(pdr.DataReader("DEXCAUS","fred",start,end)))
    aususd =pd.DataFrame(np.log(pdr.DataReader("DEXUSAL","fred",start,end)))
    usdchf =pd.DataFrame(np.log(pdr.DataReader("DEXSZUS","fred",start,end)))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    wti= pd.DataFrame(np.log(pdr.DataReader("DCOILWTICO","fred",start,end)))
    
    data=pd.concat([usdjpy,eurusd,gbpusd,aususd,usdcad,usdchf,dow,nikkei,us10y,gold,wti,btc])
    data=data.rename(columns={"DEXJPUS":"JPY","CBBTCUSD":"BTC",
                              "DEXUSEU":"EUR","DEXUSUK":"GBP",
                              "DEXCAUS":"CAD","DEXUSAL":"AUD",
                              "DGS10":"US10Y","VIXCLS":"VIX",
                              "GOLDAMGBD228NLBM":"GOLD","DCOILWTICO":"WTI",
                              "DEXSZUS":"CHF"})
    
    data = data.fillna(data.mean())
    data = pd.DataFrame(preprocessing.minmax_scale(data, axis=1),columns=["JPY","EUR","GBP","AUD",
                                                                          "CAD","CHF","DOW","NIKKEI","US10Y",
                                                                          "XAU","WTI","BTC"])
    data_corr = data.corr()
    
    fig = plt.figure(figsize=(10,10))
    sns.heatmap(data_corr,cmap = "GnBu",annot=True)
    
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text="""【本日の相関係数まとめ】
    
    この図は各商品同士の３ヶ月間の値動きの相関係数をまとめて表示したものです。
    
    色が濃い青に近いほど、正の相関が強くなります。
    
    逆に、色が薄い緑に近いほど、逆の相関が強くなります。"""
    
    return text

def usdjpy_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("DEXJPUS","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"DEXJPUS":"USDJPY"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change()*100)
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["USDJPY"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日のドル円ボラティリティ
    ドル円の昨日の終値ボラティリティは{}%です。。""".format(round(usdjpy.iloc[-1]["USDJPY"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption
    
    
def eurusd_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("DEXUSEU","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"DEXUSEU":"EURUSD"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["EURUSD"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日のユーロドルボラティリティ
    ユーロドルの昨日の終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["EURUSD"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption

def gbpusd_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("DEXUSUK","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"DEXUSUK":"GBPUSD"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["GBPUSD"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日のポンドドルボラティリティ
    ポンドドルの昨日の終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["GBPUSD"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption

def audusd_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("DEXUSAL","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"DEXUSAL":"AUDUSD"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["AUDUSD"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日のオージードルボラティリティ
    オージードルの昨日の終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["AUDUSD"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption
    
def cadusd_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("DEXCAUS","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"DEXCAUS":"USDCAD"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["USDCAD"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日のドルカナボラティリティ
    ドルカナの昨日の終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["USDCAD"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption

def usdchf_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=3)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("DEXSZUS","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"DEXSZUS":"USDCHF"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["USDCHF"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日のドルフラボラティリティ
    ドルフラの昨日の終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["USDCHF"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption

def dow_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"DJIA":"DOW"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["DOW"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日のダウボラティリティ
    ダウの昨日の終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["DOW"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption

def nikkei_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("NIKKEI225","fred",start,end)))
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["NIKKEI225"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日の日経平均ボラティリティ
    日経平均の昨日の終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["NIKKEI225"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption

def btc_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("CBBTCUSD","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"CBBTCUSD":"BTCUSD"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["BTCUSD"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日のBTCボラティリティ
    BTCの昨日の終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["BTCUSD"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption

def gold_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"GOLDAMGBD228NLBM":"XAUUSD"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["XAUUSD"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日のGOLDボラティリティ
    GOLDの昨日の終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["XAUUSD"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption

def wti_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("DCOILWTICO","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"DCOILWTICO":"WTI"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["WTI"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日の原油ボラティリティ
    原油の昨日の終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["WTI"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption

def vix_volatility():
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=6)
    
    start= three_month_ago 
    end=yesterday
    
    usdjpy = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    usdjpy = usdjpy.rename(columns={"VIXCLS":"VIX"})
    usdjpy = usdjpy.dropna()
    usdjpy = abs(usdjpy.pct_change())
    
    
    fig=plt.figure(figsize=(20,5))
    plt.plot(usdjpy.index,usdjpy["VIX"])
    plt.title("The Volatility Index between three month")
    plt.ylabel("%")
    plt.xlabel("Date")
    
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """昨日のVIXボラティリティ
    原油のVIXの終値ボラティリティは{}%です。""".format(round(usdjpy.iloc[-1]["VIX"],3))
                                       
                                       
    caption = """ボラティリティとは値動きの度合いを表したものです。
    
    ボラティリティが大きいと大きな利益のチャンスとなります。。
    
    しかし、損失額も大きくなってしまいます。
    
    ボラティリティが小さいとその逆です。"""
    
    return text,caption

def usdjpy_predict(k=10):
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=360)
    start= three_month_ago
    end=yesterday
    nikkei225=pd.DataFrame(np.log(pdr.DataReader("NIKKEI225","fred",start,end)))
    dow=pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    usdjpy =pd.DataFrame(pdr.DataReader("DEXJPUS","fred",start,end))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    lusdjpy=np.log(usdjpy)
    data=pd.concat([usdjpy,nikkei225,dow,us10y,gold,vix],axis=1)
    data=data.rename(columns={"DEXJPUS":"USDJPY","DGS10Y":"US10Y","GOLDAMGBD228NLBM":"XAUUSD","VIXCLS":"VIX"})
    data["Log_USDJPY"]=lusdjpy
    data=data.dropna()
    
    y_test1 = data.loc["2019-01-02":yesterday,"USDJPY"]
    x_train1 = data.loc["2016-01-01":"2018-12-31","NIKKEI225":"Log_USDJPY"]
    y_train1 = data.loc["2016-01-02":"2019-01-01","USDJPY"]
    x_test1 = data.loc["2019-01-01":yesterday,"NIKKEI225":"Log_USDJPY"]
    
    x_train2 = x_train1
    y_train2 = y_train1
    x_test2 = x_test1
    
    x_train3 = x_train1
    y_train3 = y_train1
    x_test3 = x_test1
    
    x_train4 = x_train1
    y_train4 = y_train1
    x_test4 = x_test1
    
    x_train5 = x_train1
    y_train5 = y_train1
    x_test5 = x_test1
    
    x_train6 = x_train1
    y_train6 = y_train1
    x_test6 = x_test1
    
    x_train7 = x_train1
    y_train7 = y_train1
    x_test7 = x_test1
    
   
    from sklearn.linear_model import LinearRegression
    lr1 = LinearRegression()
    lr1.fit(x_train1,y_train1)
    pred_y1 = lr1.predict(x_test1)
    y_plot1=pd.DataFrame({"pred":pred_y1})
    bb1=pd.DataFrame()
    bb1["pred"] = y_plot1["pred"]
    bb1["mean"]= bb1["pred"].ewm(span=k).mean()
    bb1["std"]= bb1["pred"].ewm(span=k).std()
    bb1['upper1'] = bb1['mean'] + (bb1['std'] * 1)
    bb1['lower1'] = bb1['mean'] - (bb1['std'] * 1)
    bb1['upper2'] = bb1['mean'] + (bb1['std'] * 2)
    bb1['lower2'] = bb1['mean'] - (bb1['std'] * 2)
    bb1['upper3'] = bb1['mean'] + (bb1['std'] * 3)
    bb1['lower3'] = bb1['mean'] - (bb1['std'] * 3)
    
    from sklearn.linear_model import SGDRegressor
    lr2 = SGDRegressor()
    lr2.fit(x_train2,y_train2)
    pred_y2 = lr2.predict(x_test2)
    y_plot2=pd.DataFrame({"pred":pred_y2})
    bb2=pd.DataFrame()
    bb2["pred"] = y_plot2["pred"]
    bb2["mean"]= bb2["pred"].ewm(span=k).mean()
    bb2["std"]= bb2["pred"].ewm(span=k).std()
    bb2['upper1'] = bb2['mean'] + (bb2['std'] * 1)
    bb2['lower1'] = bb2['mean'] - (bb2['std'] * 1)
    bb2['upper2'] = bb2['mean'] + (bb2['std'] * 2)
    bb2['lower2'] = bb2['mean'] - (bb2['std'] * 2)
    bb2['upper3'] = bb2['mean'] + (bb2['std'] * 3)
    bb2['lower3'] = bb2['mean'] - (bb2['std'] * 3)
    from sklearn import svm
    lr3 = svm.SVR()
    lr3.fit(x_train3,y_train3)
    pred_y3 = lr3.predict(x_test3)
    y_plot3=pd.DataFrame({"pred":pred_y3})
    bb4=pd.DataFrame()
    bb4["pred"] = y_plot3["pred"]
    bb4["mean"]= bb4["pred"].ewm(span=k).mean()
    bb4["std"]= bb4["pred"].ewm(span=k).std()
    bb4['upper1'] = bb4['mean'] + (bb4['std'] * 1)
    bb4['lower1'] = bb4['mean'] - (bb4['std'] * 1)
    bb4['upper2'] = bb4['mean'] + (bb4['std'] * 2)
    bb4['lower2'] = bb4['mean'] - (bb4['std'] * 2)
    bb4['upper3'] = bb4['mean'] + (bb4['std'] * 3)
    bb4['lower3'] = bb4['mean'] - (bb4['std'] * 3)
    from sklearn import ensemble, tree
    lr4 = ensemble.BaggingRegressor(tree.DecisionTreeRegressor())
    lr4.fit(x_train4,y_train4)
    pred_y4 = lr4.predict(x_test4)
    y_plot4=pd.DataFrame({"pred":pred_y4})
    bb5=pd.DataFrame()
    bb5["pred"] = y_plot4["pred"]
    bb5["mean"]= bb5["pred"].ewm(span=k).mean()
    bb5["std"]= bb5["pred"].ewm(span=k).std()
    bb5['upper1'] = bb5['mean'] + (bb5['std'] * 1)
    bb5['lower1'] = bb5['mean'] - (bb5['std'] * 1)
    bb5['upper2'] = bb5['mean'] + (bb5['std'] * 2)
    bb5['lower2'] = bb5['mean'] - (bb5['std'] * 2)
    bb5['upper3'] = bb5['mean'] + (bb5['std'] * 3)
    bb5['lower3'] = bb5['mean'] - (bb5['std'] * 3)
    from sklearn.neighbors import KNeighborsRegressor
    lr5 = KNeighborsRegressor()
    lr5.fit(x_train5,y_train5)
    pred_y5 = lr5.predict(x_test5)
    y_plot5=pd.DataFrame({"pred":pred_y5})
    bb6=pd.DataFrame()
    bb6["pred"] = y_plot5["pred"]
    bb6["mean"]= bb6["pred"].ewm(span=k).mean()
    bb6["std"]= bb6["pred"].ewm(span=k).std()
    bb6['upper1'] = bb6['mean'] + (bb6['std'] * 1)
    bb6['lower1'] = bb6['mean'] - (bb6['std'] * 1)
    bb6['upper2'] = bb6['mean'] + (bb6['std'] * 2)
    bb6['lower2'] = bb6['mean'] - (bb6['std'] * 2)
    bb6['upper3'] = bb6['mean'] + (bb6['std'] * 3)
    bb6['lower3'] = bb6['mean'] - (bb6['std'] * 3)
    from sklearn.ensemble import GradientBoostingRegressor
    lr6 = GradientBoostingRegressor()
    lr6.fit(x_train6,y_train6)
    pred_y6 = lr6.predict(x_test6)
    y_plot6=pd.DataFrame({"pred":pred_y6})
    bb9=pd.DataFrame()
    bb9["pred"] = y_plot6["pred"]
    bb9["mean"]= bb9["pred"].ewm(span=k).mean()
    bb9["std"]= bb9["pred"].ewm(span=k).std()
    bb9['upper1'] = bb9['mean'] + (bb9['std'] * 1)
    bb9['lower1'] = bb9['mean'] - (bb9['std'] * 1)
    bb9['upper2'] = bb9['mean'] + (bb9['std'] * 2)
    bb9['lower2'] = bb9['mean'] - (bb9['std'] * 2)
    bb9['upper3'] = bb9['mean'] + (bb9['std'] * 3)
    bb9['lower3'] = bb9['mean'] - (bb9['std'] * 3)
    from sklearn.ensemble import AdaBoostRegressor
    lr7 =  AdaBoostRegressor()
    lr7.fit(x_train7,y_train7)
    pred_y7 = lr7.predict(x_test7)
    y_plot7=pd.DataFrame({"pred":pred_y7})
    bbb=pd.DataFrame()
    bbb["pred"] = y_plot7["pred"]
    bbb["mean"]= bbb["pred"].ewm(span=k).mean()
    bbb["std"]= bbb["pred"].ewm(span=k).std()
    bbb['upper1'] = bbb['mean'] + (bbb['std'] * 1)
    bbb['lower1'] = bbb['mean'] - (bbb['std'] * 1)
    bbb['upper2'] = bbb['mean'] + (bbb['std'] * 2)
    bbb['lower2'] = bbb['mean'] - (bbb['std'] * 2)
    bbb['upper3'] = bbb['mean'] + (bbb['std'] * 3)
    bbb['lower3'] = bbb['mean'] - (bbb['std'] * 3)
    
    
    bb=pd.DataFrame()
    bb["pred"]=bb1["pred"]+bb2["pred"]+bb4["pred"]+bb5["pred"]+bb6["pred"]+bb9["pred"]+bbb["pred"]
    bb["mean"]=bb1["mean"]+bb2["mean"]+bb4["mean"]+bb5["mean"]+bb6["mean"]+bb9["mean"]+bbb["mean"]
    for j in range(1,4):
        bb["upper{}".format(j)] = bb1["upper{}".format(j)]+bb2["upper{}".format(j)]+bb4["upper{}".format(j)]+bb5["upper{}".format(j)]+bb6["upper{}".format(j)]+bb9["upper{}".format(j)]+bbb["upper{}".format(j)]
        bb["lower{}".format(j)] = bb1["lower{}".format(j)]+bb2["lower{}".format(j)]+bb4["lower{}".format(j)]+bb5["lower{}".format(j)]+bb6["lower{}".format(j)]+bb9["lower{}".format(j)]+bbb["lower{}".format(j)]
    bb = bb/7
    fig=plt.figure(figsize=(20,5))
    
    plt.plot(y_test1.index,bb["pred"],label="pred",color="m")
    plt.plot(y_test1.index,bb["mean"],label="EMA",color="g")
    plt.plot(y_test1.index,bb["upper1"],label="upper1",color="c")
    plt.plot(y_test1.index,bb["lower1"],label="lower1",color="c")
    plt.plot(y_test1.index,bb["upper2"],label="upper2",color="y")
    plt.plot(y_test1.index,bb["lower2"],label="lower2",color="y")
    plt.plot(y_test1.index,bb["upper3"],label="upper3",color="r")
    plt.plot(y_test1.index,bb["lower3"],label="lower3",color="r")
    
    y_test1.plot(label="true",color="b")
    
    plt.title("mean of prediction")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid()
    plt.legend()
    
    plt.show()
    fig.savefig("img.png")
    
    text="""今日のドル円AI予測
    
    今日ドル円の予測値は{}です。
    
    また、約68.3％の確率で{}から{}内、
    
    約95.4％の確率で{}から{}内、
    
    約99.7％の確率で{}から{}内に
    
    値動きは収束します。""".format(round(bb.at[len(bb)-1,"pred"],3),round(bb.at[len(bb)-1,"lower1"],3),
                         round(bb.at[len(bb)-1,"upper1"],3),round(bb.at[len(bb)-1,"lower2"],3),
                         round(bb.at[len(bb)-1,"upper2"],3),round(bb.at[len(bb)-1,"lower3"],3),
                         round(bb.at[len(bb)-1,"upper3"]),3)
    
    caption="""このAI予測は約5年分のダウ、日経平均、米国10年債、金、VIX、ドル円前日終値から、
    
    7つの手法で予測します。
    
    予測値はその平均です。"""
    
    return text,caption

def nikkei_predict(k=10):
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=56)
    start= three_month_ago
    end=yesterday
    nikkei225=pd.DataFrame(pdr.DataReader("NIKKEI225","fred",start,end))
    dow=pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    usdjpy =pd.DataFrame(pdr.DataReader("DEXJPUS","fred",start,end))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    usdjpy_acf=pd.DataFrame(sm.tsa.stattools.acf(usdjpy))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    usdjpy=np.log(usdjpy)
    lnikkei225 = np.log(nikkei225)
    data=pd.concat([nikkei225,usdjpy,dow,us10y,gold,vix],axis=1)
    data=data.rename(columns={"DEXJPUS":"USDJPY","DGS10Y":"US10Y","GOLDAMGBD228NLBM":"XAUUSD","VIXCLS":"VIX"})
    data["Log_NIKKEI225"]=lnikkei225
    data=data.dropna()
    
    y_test1 = data.loc["2019-01-02":yesterday,"NIKKEI225"]
    x_train1 = data.loc["2016-01-01":"2018-12-31","USDJPY":"Log_NIKKEI225"]
    y_train1 = data.loc["2016-01-02":"2019-01-01","NIKKEI225"]
    x_test1 = data.loc["2019-01-01":yesterday,"USDJPY":"Log_NIKKEI225"]
    
    x_train2 = x_train1
    y_train2 = y_train1
    x_test2 = x_test1
    
    x_train3 = x_train1
    y_train3 = y_train1
    x_test3 = x_test1
    
    x_train4 = x_train1
    y_train4 = y_train1
    x_test4 = x_test1
    
    x_train5 = x_train1
    y_train5 = y_train1
    x_test5 = x_test1
    
    x_train6 = x_train1
    y_train6 = y_train1
    x_test6 = x_test1
    
    x_train7 = x_train1
    y_train7 = y_train1
    x_test7 = x_test1
    
   
    from sklearn.linear_model import LinearRegression
    lr1 = LinearRegression()
    lr1.fit(x_train1,y_train1)
    pred_y1 = lr1.predict(x_test1)
    y_plot1=pd.DataFrame({"pred":pred_y1})
    bb1=pd.DataFrame()
    bb1["pred"] = y_plot1["pred"]
    bb1["mean"]= bb1["pred"].ewm(span=k).mean()
    bb1["std"]= bb1["pred"].ewm(span=k).std()
    bb1['upper1'] = bb1['mean'] + (bb1['std'] * 1)
    bb1['lower1'] = bb1['mean'] - (bb1['std'] * 1)
    bb1['upper2'] = bb1['mean'] + (bb1['std'] * 2)
    bb1['lower2'] = bb1['mean'] - (bb1['std'] * 2)
    bb1['upper3'] = bb1['mean'] + (bb1['std'] * 3)
    bb1['lower3'] = bb1['mean'] - (bb1['std'] * 3)
    from sklearn.linear_model import Lars
    lr2 = Lars()
    lr2.fit(x_train2,y_train2)
    pred_y2 = lr2.predict(x_test2)
    y_plot2=pd.DataFrame({"pred":pred_y2})
    bb2=pd.DataFrame()
    bb2["pred"] = y_plot2["pred"]
    bb2["mean"]= bb2["pred"].ewm(span=k).mean()
    bb2["std"]= bb2["pred"].ewm(span=k).std()
    bb2['upper1'] = bb2['mean'] + (bb2['std'] * 1)
    bb2['lower1'] = bb2['mean'] - (bb2['std'] * 1)
    bb2['upper2'] = bb2['mean'] + (bb2['std'] * 2)
    bb2['lower2'] = bb2['mean'] - (bb2['std'] * 2)
    bb2['upper3'] = bb2['mean'] + (bb2['std'] * 3)
    bb2['lower3'] = bb2['mean'] - (bb2['std'] * 3)
    from sklearn.linear_model import HuberRegressor
    lr3 = HuberRegressor()
    lr3.fit(x_train3,y_train3)
    pred_y3 = lr3.predict(x_test3)
    y_plot3=pd.DataFrame({"pred":pred_y3})
    bb4=pd.DataFrame()
    bb4["pred"] = y_plot3["pred"]
    bb4["mean"]= bb4["pred"].ewm(span=k).mean()
    bb4["std"]= bb4["pred"].ewm(span=k).std()
    bb4['upper1'] = bb4['mean'] + (bb4['std'] * 1)
    bb4['lower1'] = bb4['mean'] - (bb4['std'] * 1)
    bb4['upper2'] = bb4['mean'] + (bb4['std'] * 2)
    bb4['lower2'] = bb4['mean'] - (bb4['std'] * 2)
    bb4['upper3'] = bb4['mean'] + (bb4['std'] * 3)
    bb4['lower3'] = bb4['mean'] - (bb4['std'] * 3)
    from sklearn import ensemble, tree
    lr4 = ensemble.BaggingRegressor(tree.DecisionTreeRegressor())
    lr4.fit(x_train4,y_train4)
    pred_y4 = lr4.predict(x_test4)
    y_plot4=pd.DataFrame({"pred":pred_y4})
    bb5=pd.DataFrame()
    bb5["pred"] = y_plot4["pred"]
    bb5["mean"]= bb5["pred"].ewm(span=k).mean()
    bb5["std"]= bb5["pred"].ewm(span=k).std()
    bb5['upper1'] = bb5['mean'] + (bb5['std'] * 1)
    bb5['lower1'] = bb5['mean'] - (bb5['std'] * 1)
    bb5['upper2'] = bb5['mean'] + (bb5['std'] * 2)
    bb5['lower2'] = bb5['mean'] - (bb5['std'] * 2)
    bb5['upper3'] = bb5['mean'] + (bb5['std'] * 3)
    bb5['lower3'] = bb5['mean'] - (bb5['std'] * 3)
    from sklearn.neighbors import KNeighborsRegressor
    lr5 = KNeighborsRegressor()
    lr5.fit(x_train5,y_train5)
    pred_y5 = lr5.predict(x_test5)
    y_plot5=pd.DataFrame({"pred":pred_y5})
    bb6=pd.DataFrame()
    bb6["pred"] = y_plot5["pred"]
    bb6["mean"]= bb6["pred"].ewm(span=k).mean()
    bb6["std"]= bb6["pred"].ewm(span=k).std()
    bb6['upper1'] = bb6['mean'] + (bb6['std'] * 1)
    bb6['lower1'] = bb6['mean'] - (bb6['std'] * 1)
    bb6['upper2'] = bb6['mean'] + (bb6['std'] * 2)
    bb6['lower2'] = bb6['mean'] - (bb6['std'] * 2)
    bb6['upper3'] = bb6['mean'] + (bb6['std'] * 3)
    bb6['lower3'] = bb6['mean'] - (bb6['std'] * 3)
    from sklearn.ensemble import GradientBoostingRegressor
    lr6 = GradientBoostingRegressor()
    lr6.fit(x_train6,y_train6)
    pred_y6 = lr6.predict(x_test6)
    y_plot6=pd.DataFrame({"pred":pred_y6})
    bb9=pd.DataFrame()
    bb9["pred"] = y_plot6["pred"]
    bb9["mean"]= bb9["pred"].ewm(span=k).mean()
    bb9["std"]= bb9["pred"].ewm(span=k).std()
    bb9['upper1'] = bb9['mean'] + (bb9['std'] * 1)
    bb9['lower1'] = bb9['mean'] - (bb9['std'] * 1)
    bb9['upper2'] = bb9['mean'] + (bb9['std'] * 2)
    bb9['lower2'] = bb9['mean'] - (bb9['std'] * 2)
    bb9['upper3'] = bb9['mean'] + (bb9['std'] * 3)
    bb9['lower3'] = bb9['mean'] - (bb9['std'] * 3)
    from sklearn.linear_model import Lasso
    lr7 =  Lasso()
    lr7.fit(x_train7,y_train7)
    pred_y7 = lr7.predict(x_test7)
    y_plot7=pd.DataFrame({"pred":pred_y7})
    bbb=pd.DataFrame()
    bbb["pred"] = y_plot7["pred"]
    bbb["mean"]= bbb["pred"].ewm(span=k).mean()
    bbb["std"]= bbb["pred"].ewm(span=k).std()
    bbb['upper1'] = bbb['mean'] + (bbb['std'] * 1)
    bbb['lower1'] = bbb['mean'] - (bbb['std'] * 1)
    bbb['upper2'] = bbb['mean'] + (bbb['std'] * 2)
    bbb['lower2'] = bbb['mean'] - (bbb['std'] * 2)
    bbb['upper3'] = bbb['mean'] + (bbb['std'] * 3)
    bbb['lower3'] = bbb['mean'] - (bbb['std'] * 3)
    bb=pd.DataFrame()
    bb["pred"]=bb1["pred"]+bb2["pred"]+bb4["pred"]+bb5["pred"]+bb6["pred"]+bb9["pred"]+bbb["pred"]
    bb["mean"]=bb1["mean"]+bb2["mean"]+bb4["mean"]+bb5["mean"]+bb6["mean"]+bb9["mean"]+bbb["mean"]
    for j in range(1,4):
        bb["upper{}".format(j)] = bb1["upper{}".format(j)]+bb2["upper{}".format(j)]+bb4["upper{}".format(j)]+bb5["upper{}".format(j)]+bb6["upper{}".format(j)]+bb9["upper{}".format(j)]+bbb["upper{}".format(j)]
        bb["lower{}".format(j)] = bb1["lower{}".format(j)]+bb2["lower{}".format(j)]+bb4["lower{}".format(j)]+bb5["lower{}".format(j)]+bb6["lower{}".format(j)]+bb9["lower{}".format(j)]+bbb["lower{}".format(j)]
    bb = bb/7
    fig=plt.figure(figsize=(20,5))
    
    plt.plot(y_test1.index,bb["pred"],label="pred",color="m")
    plt.plot(y_test1.index,bb["mean"],label="EMA",color="g")
    plt.plot(y_test1.index,bb["upper1"],label="upper1",color="c")
    plt.plot(y_test1.index,bb["lower1"],label="lower1",color="c")
    plt.plot(y_test1.index,bb["upper2"],label="upper2",color="y")
    plt.plot(y_test1.index,bb["lower2"],label="lower2",color="y")
    plt.plot(y_test1.index,bb["upper3"],label="upper3",color="r")
    plt.plot(y_test1.index,bb["lower3"],label="lower3",color="r")
    
    y_test1.plot(label="true",color="b")
    
    plt.title("mean of prediction")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid()
    plt.legend()
    
    plt.show()
    fig.savefig("img.png")
    
    text="""今日の日経平均AI予測
    
    今日日経平均の予測値は{}です。
    
    また、約68.3％の確率で{}から{}内、
    
    約95.4％の確率で{}から{}内、
    
    約99.7％の確率で{}から{}内に
    
    値動きは収束します。""".format(round(bb.at[len(bb)-1,"pred"]),round(bb.at[len(bb)-1,"lower1"]),
                         round(bb.at[len(bb)-1,"upper1"]),round(bb.at[len(bb)-1,"lower2"]),
                         round(bb.at[len(bb)-1,"upper2"]),round(bb.at[len(bb)-1,"lower3"]),
                         round(bb.at[len(bb)-1,"upper3"]))
    
    caption="""このAI予測は約5年分のダウ、ドル円、米国10年債、金、VIX、日経平均前日終値から、
    
    7つの手法で予測します。
    
    予測値はその平均です。"""
    
    return text,caption


def dow_predict(k=10):
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=56)
    start= three_month_ago
    end=yesterday
    wti=pd.DataFrame(np.log(pdr.DataReader("DCOILWTICO","fred",start,end)))
    dow=pd.DataFrame(pdr.DataReader("DJIA","fred",start,end))
    usdjpy =pd.DataFrame(pdr.DataReader("DEXJPUS","fred",start,end))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    eurusd=pd.DataFrame(np.log(pdr.DataReader("DEXUSEU","fred",start,end)))
    btc=pd.DataFrame(np.log(pdr.DataReader("CBBTCUSD","fred",start,end)))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    usdjpy=np.log(usdjpy)
    ldow=np.log(dow)
    data=pd.concat([dow,wti,usdjpy,us10y,eurusd,btc,gold,vix],axis=1)
    data=data.rename(columns={"DEXJPUS":"USDJPY","DGS10Y":"US10Y","GOLDAMGBD228NLBM":"XAUUSD",
                          "VIXCLS":"VIX","CBBTCUSD":"BTCUSD","DCOILWTICO":"WTIUSD","DEXUSEU":"EURUSD"})
    data["Log_DJIA"]=ldow
    data=data.dropna()
    
    y_test1 = data.loc["2019-01-02":yesterday,"DJIA"]
    x_train1 = data.loc["2016-01-01":"2018-12-31","WTIUSD":"Log_DJIA"]
    y_train1 = data.loc["2016-01-02":"2019-01-01","DJIA"]
    x_test1 = data.loc["2019-01-01":yesterday,"WTIUSD":"Log_DJIA"]
    
    x_train2 = x_train1
    y_train2 = y_train1
    x_test2 = x_test1
    
    x_train3 = x_train1
    y_train3 = y_train1
    x_test3 = x_test1
    
    x_train4 = x_train1
    y_train4 = y_train1
    x_test4 = x_test1
    
    x_train5 = x_train1
    y_train5 = y_train1
    x_test5 = x_test1
    
    x_train6 = x_train1
    y_train6 = y_train1
    x_test6 = x_test1
    
    x_train7 = x_train1
    y_train7 = y_train1
    x_test7 = x_test1
    
   
    from sklearn.linear_model import LinearRegression
    lr1 = LinearRegression()
    lr1.fit(x_train1,y_train1)
    pred_y1 = lr1.predict(x_test1)
    y_plot1=pd.DataFrame({"pred":pred_y1})
    bb1=pd.DataFrame()
    bb1["pred"] = y_plot1["pred"]
    bb1["mean"]= bb1["pred"].ewm(span=k).mean()
    bb1["std"]= bb1["pred"].ewm(span=k).std()
    bb1['upper1'] = bb1['mean'] + (bb1['std'] * 1)
    bb1['lower1'] = bb1['mean'] - (bb1['std'] * 1)
    bb1['upper2'] = bb1['mean'] + (bb1['std'] * 2)
    bb1['lower2'] = bb1['mean'] - (bb1['std'] * 2)
    bb1['upper3'] = bb1['mean'] + (bb1['std'] * 3)
    bb1['lower3'] = bb1['mean'] - (bb1['std'] * 3)
    
    from sklearn.linear_model import Lars
    lr2 = Lars()
    lr2.fit(x_train2,y_train2)
    pred_y2 = lr2.predict(x_test2)
    y_plot2=pd.DataFrame({"pred":pred_y2})
    bb2=pd.DataFrame()
    bb2["pred"] = y_plot2["pred"]
    bb2["mean"]= bb2["pred"].ewm(span=k).mean()
    bb2["std"]= bb2["pred"].ewm(span=k).std()
    bb2['upper1'] = bb2['mean'] + (bb2['std'] * 1)
    bb2['lower1'] = bb2['mean'] - (bb2['std'] * 1)
    bb2['upper2'] = bb2['mean'] + (bb2['std'] * 2)
    bb2['lower2'] = bb2['mean'] - (bb2['std'] * 2)
    bb2['upper3'] = bb2['mean'] + (bb2['std'] * 3)
    bb2['lower3'] = bb2['mean'] - (bb2['std'] * 3)
    from sklearn.linear_model import HuberRegressor
    lr3 = HuberRegressor()
    lr3.fit(x_train3,y_train3)
    pred_y3 = lr3.predict(x_test3)
    y_plot3=pd.DataFrame({"pred":pred_y3})
    bb4=pd.DataFrame()
    bb4["pred"] = y_plot3["pred"]
    bb4["mean"]= bb4["pred"].ewm(span=k).mean()
    bb4["std"]= bb4["pred"].ewm(span=k).std()
    bb4['upper1'] = bb4['mean'] + (bb4['std'] * 1)
    bb4['lower1'] = bb4['mean'] - (bb4['std'] * 1)
    bb4['upper2'] = bb4['mean'] + (bb4['std'] * 2)
    bb4['lower2'] = bb4['mean'] - (bb4['std'] * 2)
    bb4['upper3'] = bb4['mean'] + (bb4['std'] * 3)
    bb4['lower3'] = bb4['mean'] - (bb4['std'] * 3)
    from sklearn import ensemble, tree
    lr4 = ensemble.BaggingRegressor(tree.DecisionTreeRegressor())
    lr4.fit(x_train4,y_train4)
    pred_y4 = lr4.predict(x_test4)
    y_plot4=pd.DataFrame({"pred":pred_y4})
    bb5=pd.DataFrame()
    bb5["pred"] = y_plot4["pred"]
    bb5["mean"]= bb5["pred"].ewm(span=k).mean()
    bb5["std"]= bb5["pred"].ewm(span=k).std()
    bb5['upper1'] = bb5['mean'] + (bb5['std'] * 1)
    bb5['lower1'] = bb5['mean'] - (bb5['std'] * 1)
    bb5['upper2'] = bb5['mean'] + (bb5['std'] * 2)
    bb5['lower2'] = bb5['mean'] - (bb5['std'] * 2)
    bb5['upper3'] = bb5['mean'] + (bb5['std'] * 3)
    bb5['lower3'] = bb5['mean'] - (bb5['std'] * 3)
    from sklearn.neighbors import KNeighborsRegressor
    lr5 = KNeighborsRegressor()
    lr5.fit(x_train5,y_train5)
    pred_y5 = lr5.predict(x_test5)
    y_plot5=pd.DataFrame({"pred":pred_y5})
    bb6=pd.DataFrame()
    bb6["pred"] = y_plot5["pred"]
    bb6["mean"]= bb6["pred"].ewm(span=k).mean()
    bb6["std"]= bb6["pred"].ewm(span=k).std()
    bb6['upper1'] = bb6['mean'] + (bb6['std'] * 1)
    bb6['lower1'] = bb6['mean'] - (bb6['std'] * 1)
    bb6['upper2'] = bb6['mean'] + (bb6['std'] * 2)
    bb6['lower2'] = bb6['mean'] - (bb6['std'] * 2)
    bb6['upper3'] = bb6['mean'] + (bb6['std'] * 3)
    bb6['lower3'] = bb6['mean'] - (bb6['std'] * 3)
    from sklearn.ensemble import GradientBoostingRegressor
    lr6 = GradientBoostingRegressor()
    lr6.fit(x_train6,y_train6)
    pred_y6 = lr6.predict(x_test6)
    y_plot6=pd.DataFrame({"pred":pred_y6})
    bb9=pd.DataFrame()
    bb9["pred"] = y_plot6["pred"]
    bb9["mean"]= bb9["pred"].ewm(span=k).mean()
    bb9["std"]= bb9["pred"].ewm(span=k).std()
    bb9['upper1'] = bb9['mean'] + (bb9['std'] * 1)
    bb9['lower1'] = bb9['mean'] - (bb9['std'] * 1)
    bb9['upper2'] = bb9['mean'] + (bb9['std'] * 2)
    bb9['lower2'] = bb9['mean'] - (bb9['std'] * 2)
    bb9['upper3'] = bb9['mean'] + (bb9['std'] * 3)
    bb9['lower3'] = bb9['mean'] - (bb9['std'] * 3)
    from sklearn.linear_model import TheilSenRegressor
    lr7 = TheilSenRegressor()
    lr7.fit(x_train7,y_train7)
    pred_y7 = lr7.predict(x_test7)
    y_plot7=pd.DataFrame({"pred":pred_y7})
    bbb=pd.DataFrame()
    bbb["pred"] = y_plot7["pred"]
    bbb["mean"]= bbb["pred"].ewm(span=k).mean()
    bbb["std"]= bbb["pred"].ewm(span=k).std()
    bbb['upper1'] = bbb['mean'] + (bbb['std'] * 1)
    bbb['lower1'] = bbb['mean'] - (bbb['std'] * 1)
    bbb['upper2'] = bbb['mean'] + (bbb['std'] * 2)
    bbb['lower2'] = bbb['mean'] - (bbb['std'] * 2)
    bbb['upper3'] = bbb['mean'] + (bbb['std'] * 3)
    bbb['lower3'] = bbb['mean'] - (bbb['std'] * 3)
    bb=pd.DataFrame()
    bb["pred"]=bb1["pred"]+bb2["pred"]+bb4["pred"]+bb5["pred"]+bb6["pred"]+bb9["pred"]+bbb["pred"]
    bb["mean"]=bb1["mean"]+bb2["mean"]+bb4["mean"]+bb5["mean"]+bb6["mean"]+bb9["mean"]+bbb["mean"]
    for j in range(1,4):
        bb["upper{}".format(j)] = bb1["upper{}".format(j)]+bb2["upper{}".format(j)]+bb4["upper{}".format(j)]+bb5["upper{}".format(j)]+bb6["upper{}".format(j)]+bb9["upper{}".format(j)]+bbb["upper{}".format(j)]
        bb["lower{}".format(j)] = bb1["lower{}".format(j)]+bb2["lower{}".format(j)]+bb4["lower{}".format(j)]+bb5["lower{}".format(j)]+bb6["lower{}".format(j)]+bb9["lower{}".format(j)]+bbb["lower{}".format(j)]
    bb = bb/7
    fig=plt.figure(figsize=(20,5))
    
    plt.plot(y_test1.index,bb["pred"],label="pred",color="m")
    plt.plot(y_test1.index,bb["mean"],label="EMA",color="g")
    plt.plot(y_test1.index,bb["upper1"],label="upper1",color="c")
    plt.plot(y_test1.index,bb["lower1"],label="lower1",color="c")
    plt.plot(y_test1.index,bb["upper2"],label="upper2",color="y")
    plt.plot(y_test1.index,bb["lower2"],label="lower2",color="y")
    plt.plot(y_test1.index,bb["upper3"],label="upper3",color="r")
    plt.plot(y_test1.index,bb["lower3"],label="lower3",color="r")
    
    y_test1.plot(label="true",color="b")
    
    plt.title("mean of prediction")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid()
    plt.legend()
    
    plt.show()
    fig.savefig("img.png")
    
    text="""今日のダウAI予測
    
    今日のダウの予測値は{}です。
    
    また、約68.3％の確率で{}から{}内、
    
    約95.4％の確率で{}から{}内、
    
    約99.7％の確率で{}から{}内に
    
    値動きは収束します。""".format(round(bb.at[len(bb)-1,"pred"],2),round(bb.at[len(bb)-1,"lower1"],2),
                         round(bb.at[len(bb)-1,"upper1"],2),round(bb.at[len(bb)-1,"lower2"],2),
                         round(bb.at[len(bb)-1,"upper2"],2),round(bb.at[len(bb)-1,"lower3"],2),
                         round(bb.at[len(bb)-1,"upper3"],2))
    
    caption="""このAI予測は約5年分の日経平均、ドル円、米国10年債、金、VIX、ダウ前日終値から、
    
    7つの手法で予測します。
    
    予測値はその平均です。"""
    
    return text,caption


def eurusd_predict(k=10):
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=56)
    start= three_month_ago
    end=yesterday
    gbpusd=pd.DataFrame(np.log(pdr.DataReader("DEXUSUK","fred",start,end)))
    btc=pd.DataFrame(np.log(pdr.DataReader("CBBTCUSD","fred",start,end)))
    dow=pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    eurusd=pd.DataFrame(pdr.DataReader("DEXUSEU","fred",start,end))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    leurusd=np.log(eurusd)
    data=pd.concat([eurusd,gbpusd,dow,us10y,gold,vix,btc],axis=1)
    data=data.rename(columns={"DEXUSEU":"EURUSD","DGS10Y":"US10Y","GOLDAMGBD228NLBM":"XAUUSD","VIXCLS":"VIX",
                              "DEXUSUK":"GBPUSD","CBBTCUSD":"BTCUSD"})
    data["Log_EURUSD"]=leurusd
    data=data.dropna()
    
    y_test1 = data.loc["2019-01-02":yesterday,"EURUSD"]
    x_train1 = data.loc["2016-01-01":"2018-12-31","GBPUSD":"Log_EURUSD"]
    y_train1 = data.loc["2016-01-02":"2019-01-01","EURUSD"]
    x_test1 = data.loc["2019-01-01":yesterday,"GBPUSD":"Log_EURUSD"]
    
    x_train2 = x_train1
    y_train2 = y_train1
    x_test2 = x_test1
    
    x_train3 = x_train1
    y_train3 = y_train1
    x_test3 = x_test1
    
    x_train4 = x_train1
    y_train4 = y_train1
    x_test4 = x_test1
    
    x_train5 = x_train1
    y_train5 = y_train1
    x_test5 = x_test1
    
    x_train6 = x_train1
    y_train6 = y_train1
    x_test6 = x_test1
    
    x_train7 = x_train1
    y_train7 = y_train1
    x_test7 = x_test1
    
   
    from sklearn.linear_model import LinearRegression
    lr1 = LinearRegression()
    lr1.fit(x_train1,y_train1)
    pred_y1 = lr1.predict(x_test1)
    y_plot1=pd.DataFrame({"pred":pred_y1})
    bb1=pd.DataFrame()
    bb1["pred"] = y_plot1["pred"]
    bb1["mean"]= bb1["pred"].ewm(span=k).mean()
    bb1["std"]= bb1["pred"].ewm(span=k).std()
    bb1['upper1'] = bb1['mean'] + (bb1['std'] * 1)
    bb1['lower1'] = bb1['mean'] - (bb1['std'] * 1)
    bb1['upper2'] = bb1['mean'] + (bb1['std'] * 2)
    bb1['lower2'] = bb1['mean'] - (bb1['std'] * 2)
    bb1['upper3'] = bb1['mean'] + (bb1['std'] * 3)
    bb1['lower3'] = bb1['mean'] - (bb1['std'] * 3)
    
    from sklearn.linear_model import HuberRegressor
    lr2 = HuberRegressor()
    lr2.fit(x_train2,y_train2)
    pred_y2 = lr2.predict(x_test2)
    y_plot2=pd.DataFrame({"pred":pred_y2})
    bb2=pd.DataFrame()
    bb2["pred"] = y_plot2["pred"]
    bb2["mean"]= bb2["pred"].ewm(span=k).mean()
    bb2["std"]= bb2["pred"].ewm(span=k).std()
    bb2['upper1'] = bb2['mean'] + (bb2['std'] * 1)
    bb2['lower1'] = bb2['mean'] - (bb2['std'] * 1)
    bb2['upper2'] = bb2['mean'] + (bb2['std'] * 2)
    bb2['lower2'] = bb2['mean'] - (bb2['std'] * 2)
    bb2['upper3'] = bb2['mean'] + (bb2['std'] * 3)
    bb2['lower3'] = bb2['mean'] - (bb2['std'] * 3)
    from sklearn.linear_model import Lars
    lr3 = Lars()
    lr3.fit(x_train3,y_train3)
    pred_y3 = lr3.predict(x_test3)
    y_plot3=pd.DataFrame({"pred":pred_y3})
    bb4=pd.DataFrame()
    bb4["pred"] = y_plot3["pred"]
    bb4["mean"]= bb4["pred"].ewm(span=k).mean()
    bb4["std"]= bb4["pred"].ewm(span=k).std()
    bb4['upper1'] = bb4['mean'] + (bb4['std'] * 1)
    bb4['lower1'] = bb4['mean'] - (bb4['std'] * 1)
    bb4['upper2'] = bb4['mean'] + (bb4['std'] * 2)
    bb4['lower2'] = bb4['mean'] - (bb4['std'] * 2)
    bb4['upper3'] = bb4['mean'] + (bb4['std'] * 3)
    bb4['lower3'] = bb4['mean'] - (bb4['std'] * 3)
    from sklearn import ensemble, tree
    lr4 = ensemble.BaggingRegressor(tree.DecisionTreeRegressor())
    lr4.fit(x_train4,y_train4)
    pred_y4 = lr4.predict(x_test4)
    y_plot4=pd.DataFrame({"pred":pred_y4})
    bb5=pd.DataFrame()
    bb5["pred"] = y_plot4["pred"]
    bb5["mean"]= bb5["pred"].ewm(span=k).mean()
    bb5["std"]= bb5["pred"].ewm(span=k).std()
    bb5['upper1'] = bb5['mean'] + (bb5['std'] * 1)
    bb5['lower1'] = bb5['mean'] - (bb5['std'] * 1)
    bb5['upper2'] = bb5['mean'] + (bb5['std'] * 2)
    bb5['lower2'] = bb5['mean'] - (bb5['std'] * 2)
    bb5['upper3'] = bb5['mean'] + (bb5['std'] * 3)
    bb5['lower3'] = bb5['mean'] - (bb5['std'] * 3)
    from sklearn.neighbors import KNeighborsRegressor
    lr5 = KNeighborsRegressor()
    lr5.fit(x_train5,y_train5)
    pred_y5 = lr5.predict(x_test5)
    y_plot5=pd.DataFrame({"pred":pred_y5})
    bb6=pd.DataFrame()
    bb6["pred"] = y_plot5["pred"]
    bb6["mean"]= bb6["pred"].ewm(span=k).mean()
    bb6["std"]= bb6["pred"].ewm(span=k).std()
    bb6['upper1'] = bb6['mean'] + (bb6['std'] * 1)
    bb6['lower1'] = bb6['mean'] - (bb6['std'] * 1)
    bb6['upper2'] = bb6['mean'] + (bb6['std'] * 2)
    bb6['lower2'] = bb6['mean'] - (bb6['std'] * 2)
    bb6['upper3'] = bb6['mean'] + (bb6['std'] * 3)
    bb6['lower3'] = bb6['mean'] - (bb6['std'] * 3)
    from sklearn.ensemble import GradientBoostingRegressor
    lr6 = GradientBoostingRegressor()
    lr6.fit(x_train6,y_train6)
    pred_y6 = lr6.predict(x_test6)
    y_plot6=pd.DataFrame({"pred":pred_y6})
    bb9=pd.DataFrame()
    bb9["pred"] = y_plot6["pred"]
    bb9["mean"]= bb9["pred"].ewm(span=k).mean()
    bb9["std"]= bb9["pred"].ewm(span=k).std()
    bb9['upper1'] = bb9['mean'] + (bb9['std'] * 1)
    bb9['lower1'] = bb9['mean'] - (bb9['std'] * 1)
    bb9['upper2'] = bb9['mean'] + (bb9['std'] * 2)
    bb9['lower2'] = bb9['mean'] - (bb9['std'] * 2)
    bb9['upper3'] = bb9['mean'] + (bb9['std'] * 3)
    bb9['lower3'] = bb9['mean'] - (bb9['std'] * 3)
    from sklearn.linear_model import ARDRegression
    lr7 =  ARDRegression()
    lr7.fit(x_train7,y_train7)
    pred_y7 = lr7.predict(x_test7)
    y_plot7=pd.DataFrame({"pred":pred_y7})
    bbb=pd.DataFrame()
    bbb["pred"] = y_plot7["pred"]
    bbb["mean"]= bbb["pred"].ewm(span=k).mean()
    bbb["std"]= bbb["pred"].ewm(span=k).std()
    bbb['upper1'] = bbb['mean'] + (bbb['std'] * 1)
    bbb['lower1'] = bbb['mean'] - (bbb['std'] * 1)
    bbb['upper2'] = bbb['mean'] + (bbb['std'] * 2)
    bbb['lower2'] = bbb['mean'] - (bbb['std'] * 2)
    bbb['upper3'] = bbb['mean'] + (bbb['std'] * 3)
    bbb['lower3'] = bbb['mean'] - (bbb['std'] * 3)
    bb=pd.DataFrame()
    bb["pred"]=bb1["pred"]+bb2["pred"]+bb4["pred"]+bb5["pred"]+bb6["pred"]+bb9["pred"]+bbb["pred"]
    bb["mean"]=bb1["mean"]+bb2["mean"]+bb4["mean"]+bb5["mean"]+bb6["mean"]+bb9["mean"]+bbb["mean"]
    for j in range(1,4):
        bb["upper{}".format(j)] = bb1["upper{}".format(j)]+bb2["upper{}".format(j)]+bb4["upper{}".format(j)]+bb5["upper{}".format(j)]+bb6["upper{}".format(j)]+bb9["upper{}".format(j)]+bbb["upper{}".format(j)]
        bb["lower{}".format(j)] = bb1["lower{}".format(j)]+bb2["lower{}".format(j)]+bb4["lower{}".format(j)]+bb5["lower{}".format(j)]+bb6["lower{}".format(j)]+bb9["lower{}".format(j)]+bbb["lower{}".format(j)]
    bb = bb/7
    fig=plt.figure(figsize=(20,5))
    
    plt.plot(y_test1.index,bb["pred"],label="pred",color="m")
    plt.plot(y_test1.index,bb["mean"],label="EMA",color="g")
    plt.plot(y_test1.index,bb["upper1"],label="upper1",color="c")
    plt.plot(y_test1.index,bb["lower1"],label="lower1",color="c")
    plt.plot(y_test1.index,bb["upper2"],label="upper2",color="y")
    plt.plot(y_test1.index,bb["lower2"],label="lower2",color="y")
    plt.plot(y_test1.index,bb["upper3"],label="upper3",color="r")
    plt.plot(y_test1.index,bb["lower3"],label="lower3",color="r")
    
    y_test1.plot(label="true",color="b")
    
    plt.title("mean of prediction")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid()
    plt.legend()
    
    plt.show()
    fig.savefig("img.png")
    
    text="""今日のユーロドルAI予測
    
    今日ユーロドルの予測値は{}です。
    
    また、約68.3％の確率で{}から{}内、
    
    約95.4％の確率で{}から{}内、
    
    約99.7％の確率で{}から{}内に
    
    値動きは収束します。""".format(round(bb.at[len(bb)-1,"pred"],6),round(bb.at[len(bb)-1,"lower1"],6),
                         round(bb.at[len(bb)-1,"upper1"],6),round(bb.at[len(bb)-1,"lower2"],6),
                         round(bb.at[len(bb)-1,"upper2"],6),round(bb.at[len(bb)-1,"lower3"],6),
                         round(bb.at[len(bb)-1,"upper3"]),6)
    
    caption="""このAI予測は約5年分のダウ、ポンドドル、米国10年債、金、VIX、BTC、ユーロドル前日終値から、
    
    7つの手法で予測します。
    
    予測値はその平均です。"""
    
    return text,caption


def gbpusd_predict(k=10):
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=56)
    start= three_month_ago
    end=yesterday
    gbpusd=pd.DataFrame(pdr.DataReader("DEXUSUK","fred",start,end))
    btc=pd.DataFrame(np.log(pdr.DataReader("CBBTCUSD","fred",start,end)))
    dow=pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    eurusd=pd.DataFrame(np.log(pdr.DataReader("DEXUSEU","fred",start,end)))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    lgbpusd=np.log(gbpusd)
    data=pd.concat([gbpusd,eurusd,dow,us10y,gold,vix,btc],axis=1)
    data=data.rename(columns={"DEXUSEU":"EURUSD","DGS10Y":"US10Y","GOLDAMGBD228NLBM":"XAUUSD","VIXCLS":"VIX",
                              "DEXUSUK":"GBPUSD","CBBTCUSD":"BTCUSD"})
    data["Log_GBPUSD"]=lgbpusd
    data=data.dropna()
    
    y_test1 = data.loc["2019-01-02":yesterday,"GBPUSD"]
    x_train1 = data.loc["2016-01-01":"2018-12-31","EURUSD":"Log_GBPUSD"]
    y_train1 = data.loc["2016-01-02":"2019-01-01","GBPUSD"]
    x_test1 = data.loc["2019-01-01":yesterday,"EURUSD":"Log_GBPUSD"]
    
    x_train2 = x_train1
    y_train2 = y_train1
    x_test2 = x_test1
    
    x_train3 = x_train1
    y_train3 = y_train1
    x_test3 = x_test1
    
    x_train4 = x_train1
    y_train4 = y_train1
    x_test4 = x_test1
    
    x_train5 = x_train1
    y_train5 = y_train1
    x_test5 = x_test1
    
    x_train6 = x_train1
    y_train6 = y_train1
    x_test6 = x_test1
    
    x_train7 = x_train1
    y_train7 = y_train1
    x_test7 = x_test1
    
   
    from sklearn.linear_model import LinearRegression
    lr1 = LinearRegression()
    lr1.fit(x_train1,y_train1)
    pred_y1 = lr1.predict(x_test1)
    y_plot1=pd.DataFrame({"pred":pred_y1})
    bb1=pd.DataFrame()
    bb1["pred"] = y_plot1["pred"]
    bb1["mean"]= bb1["pred"].ewm(span=k).mean()
    bb1["std"]= bb1["pred"].ewm(span=k).std()
    bb1['upper1'] = bb1['mean'] + (bb1['std'] * 1)
    bb1['lower1'] = bb1['mean'] - (bb1['std'] * 1)
    bb1['upper2'] = bb1['mean'] + (bb1['std'] * 2)
    bb1['lower2'] = bb1['mean'] - (bb1['std'] * 2)
    bb1['upper3'] = bb1['mean'] + (bb1['std'] * 3)
    bb1['lower3'] = bb1['mean'] - (bb1['std'] * 3)
    
    from sklearn.linear_model import HuberRegressor
    lr2 = HuberRegressor()
    lr2.fit(x_train2,y_train2)
    pred_y2 = lr2.predict(x_test2)
    y_plot2=pd.DataFrame({"pred":pred_y2})
    bb2=pd.DataFrame()
    bb2["pred"] = y_plot2["pred"]
    bb2["mean"]= bb2["pred"].ewm(span=k).mean()
    bb2["std"]= bb2["pred"].ewm(span=k).std()
    bb2['upper1'] = bb2['mean'] + (bb2['std'] * 1)
    bb2['lower1'] = bb2['mean'] - (bb2['std'] * 1)
    bb2['upper2'] = bb2['mean'] + (bb2['std'] * 2)
    bb2['lower2'] = bb2['mean'] - (bb2['std'] * 2)
    bb2['upper3'] = bb2['mean'] + (bb2['std'] * 3)
    bb2['lower3'] = bb2['mean'] - (bb2['std'] * 3)
    from sklearn.linear_model import Lars
    lr3 = Lars()
    lr3.fit(x_train3,y_train3)
    pred_y3 = lr3.predict(x_test3)
    y_plot3=pd.DataFrame({"pred":pred_y3})
    bb4=pd.DataFrame()
    bb4["pred"] = y_plot3["pred"]
    bb4["mean"]= bb4["pred"].ewm(span=k).mean()
    bb4["std"]= bb4["pred"].ewm(span=k).std()
    bb4['upper1'] = bb4['mean'] + (bb4['std'] * 1)
    bb4['lower1'] = bb4['mean'] - (bb4['std'] * 1)
    bb4['upper2'] = bb4['mean'] + (bb4['std'] * 2)
    bb4['lower2'] = bb4['mean'] - (bb4['std'] * 2)
    bb4['upper3'] = bb4['mean'] + (bb4['std'] * 3)
    bb4['lower3'] = bb4['mean'] - (bb4['std'] * 3)
    from sklearn import ensemble, tree
    lr4 = ensemble.BaggingRegressor(tree.DecisionTreeRegressor())
    lr4.fit(x_train4,y_train4)
    pred_y4 = lr4.predict(x_test4)
    y_plot4=pd.DataFrame({"pred":pred_y4})
    bb5=pd.DataFrame()
    bb5["pred"] = y_plot4["pred"]
    bb5["mean"]= bb5["pred"].ewm(span=k).mean()
    bb5["std"]= bb5["pred"].ewm(span=k).std()
    bb5['upper1'] = bb5['mean'] + (bb5['std'] * 1)
    bb5['lower1'] = bb5['mean'] - (bb5['std'] * 1)
    bb5['upper2'] = bb5['mean'] + (bb5['std'] * 2)
    bb5['lower2'] = bb5['mean'] - (bb5['std'] * 2)
    bb5['upper3'] = bb5['mean'] + (bb5['std'] * 3)
    bb5['lower3'] = bb5['mean'] - (bb5['std'] * 3)
    from sklearn.neighbors import KNeighborsRegressor
    lr5 = KNeighborsRegressor()
    lr5.fit(x_train5,y_train5)
    pred_y5 = lr5.predict(x_test5)
    y_plot5=pd.DataFrame({"pred":pred_y5})
    bb6=pd.DataFrame()
    bb6["pred"] = y_plot5["pred"]
    bb6["mean"]= bb6["pred"].ewm(span=k).mean()
    bb6["std"]= bb6["pred"].ewm(span=k).std()
    bb6['upper1'] = bb6['mean'] + (bb6['std'] * 1)
    bb6['lower1'] = bb6['mean'] - (bb6['std'] * 1)
    bb6['upper2'] = bb6['mean'] + (bb6['std'] * 2)
    bb6['lower2'] = bb6['mean'] - (bb6['std'] * 2)
    bb6['upper3'] = bb6['mean'] + (bb6['std'] * 3)
    bb6['lower3'] = bb6['mean'] - (bb6['std'] * 3)
    from sklearn.ensemble import GradientBoostingRegressor
    lr6 = GradientBoostingRegressor()
    lr6.fit(x_train6,y_train6)
    pred_y6 = lr6.predict(x_test6)
    y_plot6=pd.DataFrame({"pred":pred_y6})
    bb9=pd.DataFrame()
    bb9["pred"] = y_plot6["pred"]
    bb9["mean"]= bb9["pred"].ewm(span=k).mean()
    bb9["std"]= bb9["pred"].ewm(span=k).std()
    bb9['upper1'] = bb9['mean'] + (bb9['std'] * 1)
    bb9['lower1'] = bb9['mean'] - (bb9['std'] * 1)
    bb9['upper2'] = bb9['mean'] + (bb9['std'] * 2)
    bb9['lower2'] = bb9['mean'] - (bb9['std'] * 2)
    bb9['upper3'] = bb9['mean'] + (bb9['std'] * 3)
    bb9['lower3'] = bb9['mean'] - (bb9['std'] * 3)
    from sklearn.linear_model import ARDRegression
    lr7 =  ARDRegression()
    lr7.fit(x_train7,y_train7)
    pred_y7 = lr7.predict(x_test7)
    y_plot7=pd.DataFrame({"pred":pred_y7})
    bbb=pd.DataFrame()
    bbb["pred"] = y_plot7["pred"]
    bbb["mean"]= bbb["pred"].ewm(span=k).mean()
    bbb["std"]= bbb["pred"].ewm(span=k).std()
    bbb['upper1'] = bbb['mean'] + (bbb['std'] * 1)
    bbb['lower1'] = bbb['mean'] - (bbb['std'] * 1)
    bbb['upper2'] = bbb['mean'] + (bbb['std'] * 2)
    bbb['lower2'] = bbb['mean'] - (bbb['std'] * 2)
    bbb['upper3'] = bbb['mean'] + (bbb['std'] * 3)
    bbb['lower3'] = bbb['mean'] - (bbb['std'] * 3)
    bb=pd.DataFrame()
    bb["pred"]=bb1["pred"]+bb2["pred"]+bb4["pred"]+bb5["pred"]+bb6["pred"]+bb9["pred"]+bbb["pred"]
    bb["mean"]=bb1["mean"]+bb2["mean"]+bb4["mean"]+bb5["mean"]+bb6["mean"]+bb9["mean"]+bbb["mean"]
    for j in range(1,4):
        bb["upper{}".format(j)] = bb1["upper{}".format(j)]+bb2["upper{}".format(j)]+bb4["upper{}".format(j)]+bb5["upper{}".format(j)]+bb6["upper{}".format(j)]+bb9["upper{}".format(j)]+bbb["upper{}".format(j)]
        bb["lower{}".format(j)] = bb1["lower{}".format(j)]+bb2["lower{}".format(j)]+bb4["lower{}".format(j)]+bb5["lower{}".format(j)]+bb6["lower{}".format(j)]+bb9["lower{}".format(j)]+bbb["lower{}".format(j)]
    bb = bb/7
    fig=plt.figure(figsize=(20,5))
    
    plt.plot(y_test1.index,bb["pred"],label="pred",color="m")
    plt.plot(y_test1.index,bb["mean"],label="EMA",color="g")
    plt.plot(y_test1.index,bb["upper1"],label="upper1",color="c")
    plt.plot(y_test1.index,bb["lower1"],label="lower1",color="c")
    plt.plot(y_test1.index,bb["upper2"],label="upper2",color="y")
    plt.plot(y_test1.index,bb["lower2"],label="lower2",color="y")
    plt.plot(y_test1.index,bb["upper3"],label="upper3",color="r")
    plt.plot(y_test1.index,bb["lower3"],label="lower3",color="r")
    
    y_test1.plot(label="true",color="b")
    
    plt.title("mean of prediction")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid()
    plt.legend()
    
    plt.show()
    fig.savefig("img.png")
    
    text="""今日のポンドドルAI予測
    
    今日ポンドドルの予測値は{}です。
    
    また、約68.3％の確率で{}から{}内、
    
    約95.4％の確率で{}から{}内、
    
    約99.7％の確率で{}から{}内に
    
    値動きは収束します。""".format(round(bb.at[len(bb)-1,"pred"],6),round(bb.at[len(bb)-1,"lower1"],6),
                         round(bb.at[len(bb)-1,"upper1"],6),round(bb.at[len(bb)-1,"lower2"],6),
                         round(bb.at[len(bb)-1,"upper2"],6),round(bb.at[len(bb)-1,"lower3"],6),
                         round(bb.at[len(bb)-1,"upper3"]),6)
    
    caption="""このAI予測は約5年分のダウ、ユーロドル、米国10年債、金、VIX、BTC、ポンドドル前日終値から、
    
    7つの手法で予測します。
    
    予測値はその平均です。"""
    
    return text,caption


def audusd_predict(k=10):
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=56)
    start= three_month_ago
    end=yesterday
    audusd=pd.DataFrame(pdr.DataReader("DEXUSAL","fred",start,end))
    btc=pd.DataFrame(np.log(pdr.DataReader("CBBTCUSD","fred",start,end)))
    dow=pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    wti=pd.DataFrame(np.log(pdr.DataReader("DCOILWTICO","fred",start,end)))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    laudusd=np.log(audusd)
    data=pd.concat([audusd,wti,dow,us10y,gold,vix,btc],axis=1)
    data=data.rename(columns={"DCOILWTICO":"WTIUSD","DGS10Y":"US10Y","GOLDAMGBD228NLBM":"XAUUSD","VIXCLS":"VIX",
                              "DEXUSAL":"AUDUSD","CBBTCUSD":"BTCUSD"})
    data["Log_AUDUSD"]=laudusd
    data=data.dropna()
    
    y_test1 = data.loc["2019-01-02":yesterday,"AUDUSD"]
    x_train1 = data.loc["2016-01-01":"2018-12-31","WTIUSD":"Log_AUDUSD"]
    y_train1 = data.loc["2016-01-02":"2019-01-01","AUDUSD"]
    x_test1 = data.loc["2019-01-01":yesterday,"WTIUSD":"Log_AUDUSD"]
    
    x_train2 = x_train1
    y_train2 = y_train1
    x_test2 = x_test1
    
    x_train3 = x_train1
    y_train3 = y_train1
    x_test3 = x_test1
    
    x_train4 = x_train1
    y_train4 = y_train1
    x_test4 = x_test1
    
    x_train5 = x_train1
    y_train5 = y_train1
    x_test5 = x_test1
    
    x_train6 = x_train1
    y_train6 = y_train1
    x_test6 = x_test1
    
    x_train7 = x_train1
    y_train7 = y_train1
    x_test7 = x_test1
    
   
    from sklearn.linear_model import LinearRegression
    lr1 = LinearRegression()
    lr1.fit(x_train1,y_train1)
    pred_y1 = lr1.predict(x_test1)
    y_plot1=pd.DataFrame({"pred":pred_y1})
    bb1=pd.DataFrame()
    bb1["pred"] = y_plot1["pred"]
    bb1["mean"]= bb1["pred"].ewm(span=k).mean()
    bb1["std"]= bb1["pred"].ewm(span=k).std()
    bb1['upper1'] = bb1['mean'] + (bb1['std'] * 1)
    bb1['lower1'] = bb1['mean'] - (bb1['std'] * 1)
    bb1['upper2'] = bb1['mean'] + (bb1['std'] * 2)
    bb1['lower2'] = bb1['mean'] - (bb1['std'] * 2)
    bb1['upper3'] = bb1['mean'] + (bb1['std'] * 3)
    bb1['lower3'] = bb1['mean'] - (bb1['std'] * 3)
    
    from sklearn.linear_model import HuberRegressor
    lr2 = HuberRegressor()
    lr2.fit(x_train2,y_train2)
    pred_y2 = lr2.predict(x_test2)
    y_plot2=pd.DataFrame({"pred":pred_y2})
    bb2=pd.DataFrame()
    bb2["pred"] = y_plot2["pred"]
    bb2["mean"]= bb2["pred"].ewm(span=k).mean()
    bb2["std"]= bb2["pred"].ewm(span=k).std()
    bb2['upper1'] = bb2['mean'] + (bb2['std'] * 1)
    bb2['lower1'] = bb2['mean'] - (bb2['std'] * 1)
    bb2['upper2'] = bb2['mean'] + (bb2['std'] * 2)
    bb2['lower2'] = bb2['mean'] - (bb2['std'] * 2)
    bb2['upper3'] = bb2['mean'] + (bb2['std'] * 3)
    bb2['lower3'] = bb2['mean'] - (bb2['std'] * 3)
    from sklearn.linear_model import Lars
    lr3 = Lars()
    lr3.fit(x_train3,y_train3)
    pred_y3 = lr3.predict(x_test3)
    y_plot3=pd.DataFrame({"pred":pred_y3})
    bb4=pd.DataFrame()
    bb4["pred"] = y_plot3["pred"]
    bb4["mean"]= bb4["pred"].ewm(span=k).mean()
    bb4["std"]= bb4["pred"].ewm(span=k).std()
    bb4['upper1'] = bb4['mean'] + (bb4['std'] * 1)
    bb4['lower1'] = bb4['mean'] - (bb4['std'] * 1)
    bb4['upper2'] = bb4['mean'] + (bb4['std'] * 2)
    bb4['lower2'] = bb4['mean'] - (bb4['std'] * 2)
    bb4['upper3'] = bb4['mean'] + (bb4['std'] * 3)
    bb4['lower3'] = bb4['mean'] - (bb4['std'] * 3)
    from sklearn import ensemble, tree
    lr4 = ensemble.BaggingRegressor(tree.DecisionTreeRegressor())
    lr4.fit(x_train4,y_train4)
    pred_y4 = lr4.predict(x_test4)
    y_plot4=pd.DataFrame({"pred":pred_y4})
    bb5=pd.DataFrame()
    bb5["pred"] = y_plot4["pred"]
    bb5["mean"]= bb5["pred"].ewm(span=k).mean()
    bb5["std"]= bb5["pred"].ewm(span=k).std()
    bb5['upper1'] = bb5['mean'] + (bb5['std'] * 1)
    bb5['lower1'] = bb5['mean'] - (bb5['std'] * 1)
    bb5['upper2'] = bb5['mean'] + (bb5['std'] * 2)
    bb5['lower2'] = bb5['mean'] - (bb5['std'] * 2)
    bb5['upper3'] = bb5['mean'] + (bb5['std'] * 3)
    bb5['lower3'] = bb5['mean'] - (bb5['std'] * 3)
    from sklearn.linear_model import TheilSenRegressor
    lr5 = TheilSenRegressor()
    lr5.fit(x_train5,y_train5)
    pred_y5 = lr5.predict(x_test5)
    y_plot5=pd.DataFrame({"pred":pred_y5})
    bb6=pd.DataFrame()
    bb6["pred"] = y_plot5["pred"]
    bb6["mean"]= bb6["pred"].ewm(span=k).mean()
    bb6["std"]= bb6["pred"].ewm(span=k).std()
    bb6['upper1'] = bb6['mean'] + (bb6['std'] * 1)
    bb6['lower1'] = bb6['mean'] - (bb6['std'] * 1)
    bb6['upper2'] = bb6['mean'] + (bb6['std'] * 2)
    bb6['lower2'] = bb6['mean'] - (bb6['std'] * 2)
    bb6['upper3'] = bb6['mean'] + (bb6['std'] * 3)
    bb6['lower3'] = bb6['mean'] - (bb6['std'] * 3)
    from sklearn.ensemble import GradientBoostingRegressor
    lr6 = GradientBoostingRegressor()
    lr6.fit(x_train6,y_train6)
    pred_y6 = lr6.predict(x_test6)
    y_plot6=pd.DataFrame({"pred":pred_y6})
    bb9=pd.DataFrame()
    bb9["pred"] = y_plot6["pred"]
    bb9["mean"]= bb9["pred"].ewm(span=k).mean()
    bb9["std"]= bb9["pred"].ewm(span=k).std()
    bb9['upper1'] = bb9['mean'] + (bb9['std'] * 1)
    bb9['lower1'] = bb9['mean'] - (bb9['std'] * 1)
    bb9['upper2'] = bb9['mean'] + (bb9['std'] * 2)
    bb9['lower2'] = bb9['mean'] - (bb9['std'] * 2)
    bb9['upper3'] = bb9['mean'] + (bb9['std'] * 3)
    bb9['lower3'] = bb9['mean'] - (bb9['std'] * 3)
    from sklearn.linear_model import ARDRegression
    lr7 =  ARDRegression()
    lr7.fit(x_train7,y_train7)
    pred_y7 = lr7.predict(x_test7)
    y_plot7=pd.DataFrame({"pred":pred_y7})
    bbb=pd.DataFrame()
    bbb["pred"] = y_plot7["pred"]
    bbb["mean"]= bbb["pred"].ewm(span=k).mean()
    bbb["std"]= bbb["pred"].ewm(span=k).std()
    bbb['upper1'] = bbb['mean'] + (bbb['std'] * 1)
    bbb['lower1'] = bbb['mean'] - (bbb['std'] * 1)
    bbb['upper2'] = bbb['mean'] + (bbb['std'] * 2)
    bbb['lower2'] = bbb['mean'] - (bbb['std'] * 2)
    bbb['upper3'] = bbb['mean'] + (bbb['std'] * 3)
    bbb['lower3'] = bbb['mean'] - (bbb['std'] * 3)
    bb=pd.DataFrame()
    bb["pred"]=bb1["pred"]+bb2["pred"]+bb4["pred"]+bb5["pred"]+bb6["pred"]+bb9["pred"]+bbb["pred"]
    bb["mean"]=bb1["mean"]+bb2["mean"]+bb4["mean"]+bb5["mean"]+bb6["mean"]+bb9["mean"]+bbb["mean"]
    for j in range(1,4):
        bb["upper{}".format(j)] = bb1["upper{}".format(j)]+bb2["upper{}".format(j)]+bb4["upper{}".format(j)]+bb5["upper{}".format(j)]+bb6["upper{}".format(j)]+bb9["upper{}".format(j)]+bbb["upper{}".format(j)]
        bb["lower{}".format(j)] = bb1["lower{}".format(j)]+bb2["lower{}".format(j)]+bb4["lower{}".format(j)]+bb5["lower{}".format(j)]+bb6["lower{}".format(j)]+bb9["lower{}".format(j)]+bbb["lower{}".format(j)]
    bb = bb/7
    fig=plt.figure(figsize=(20,5))
    
    plt.plot(y_test1.index,bb["pred"],label="pred",color="m")
    plt.plot(y_test1.index,bb["mean"],label="EMA",color="g")
    plt.plot(y_test1.index,bb["upper1"],label="upper1",color="c")
    plt.plot(y_test1.index,bb["lower1"],label="lower1",color="c")
    plt.plot(y_test1.index,bb["upper2"],label="upper2",color="y")
    plt.plot(y_test1.index,bb["lower2"],label="lower2",color="y")
    plt.plot(y_test1.index,bb["upper3"],label="upper3",color="r")
    plt.plot(y_test1.index,bb["lower3"],label="lower3",color="r")
    
    y_test1.plot(label="true",color="b")
    
    plt.title("mean of prediction")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid()
    plt.legend()
    
    plt.show()
    fig.savefig("img.png")
    
    text="""今日のオージードルAI予測
    
    今日オージードルの予測値は{}です。
    
    また、約68.3％の確率で{}から{}内、
    
    約95.4％の確率で{}から{}内、
    
    約99.7％の確率で{}から{}内に
    
    値動きは収束します。""".format(round(bb.at[len(bb)-1,"pred"],6),round(bb.at[len(bb)-1,"lower1"],6),
                         round(bb.at[len(bb)-1,"upper1"],6),round(bb.at[len(bb)-1,"lower2"],6),
                         round(bb.at[len(bb)-1,"upper2"],6),round(bb.at[len(bb)-1,"lower3"],6),
                         round(bb.at[len(bb)-1,"upper3"]),6)
    
    caption="""このAI予測は約5年分のダウ、原油、米国10年債、金、VIX、BTC、オージードル前日終値から、
    
    7つの手法で予測します。
    
    予測値はその平均です。"""
    
    return text,caption


def usdcad_predict(k=10):
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=56)
    start= three_month_ago
    end=yesterday
    usdcad=pd.DataFrame(pdr.DataReader("DEXCAUS","fred",start,end))
    btc=pd.DataFrame(np.log(pdr.DataReader("CBBTCUSD","fred",start,end)))
    dow=pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    wti=pd.DataFrame(np.log(pdr.DataReader("DCOILWTICO","fred",start,end)))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    lusdcad=np.log(usdcad)
    data=pd.concat([usdcad,wti,dow,us10y,gold,vix,btc],axis=1)
    data=data.rename(columns={"DCOILWTICO":"WTIUSD","DGS10Y":"US10Y","GOLDAMGBD228NLBM":"XAUUSD","VIXCLS":"VIX",
                              "DEXCAUS":"USDCAD","CBBTCUSD":"BTCUSD"})
    data["Log_USDCAD"]=lusdcad
    data=data.dropna()
    
    y_test1 = data.loc["2019-01-02":yesterday,"USDCAD"]
    x_train1 = data.loc["2016-01-01":"2018-12-31","WTIUSD":"Log_USDCAD"]
    y_train1 = data.loc["2016-01-02":"2019-01-01","USDCAD"]
    x_test1 = data.loc["2019-01-01":yesterday,"WTIUSD":"Log_USDCAD"]
    
    x_train2 = x_train1
    y_train2 = y_train1
    x_test2 = x_test1
    
    x_train3 = x_train1
    y_train3 = y_train1
    x_test3 = x_test1
    
    x_train4 = x_train1
    y_train4 = y_train1
    x_test4 = x_test1
    
    x_train5 = x_train1
    y_train5 = y_train1
    x_test5 = x_test1
    
    x_train6 = x_train1
    y_train6 = y_train1
    x_test6 = x_test1
    
    x_train7 = x_train1
    y_train7 = y_train1
    x_test7 = x_test1
    
   
    from sklearn.linear_model import LinearRegression
    lr1 = LinearRegression()
    lr1.fit(x_train1,y_train1)
    pred_y1 = lr1.predict(x_test1)
    y_plot1=pd.DataFrame({"pred":pred_y1})
    bb1=pd.DataFrame()
    bb1["pred"] = y_plot1["pred"]
    bb1["mean"]= bb1["pred"].ewm(span=k).mean()
    bb1["std"]= bb1["pred"].ewm(span=k).std()
    bb1['upper1'] = bb1['mean'] + (bb1['std'] * 1)
    bb1['lower1'] = bb1['mean'] - (bb1['std'] * 1)
    bb1['upper2'] = bb1['mean'] + (bb1['std'] * 2)
    bb1['lower2'] = bb1['mean'] - (bb1['std'] * 2)
    bb1['upper3'] = bb1['mean'] + (bb1['std'] * 3)
    bb1['lower3'] = bb1['mean'] - (bb1['std'] * 3)
    
    from sklearn.linear_model import Lars
    lr2 = Lars()
    lr2.fit(x_train2,y_train2)
    pred_y2 = lr2.predict(x_test2)
    y_plot2=pd.DataFrame({"pred":pred_y2})
    bb2=pd.DataFrame()
    bb2["pred"] = y_plot2["pred"]
    bb2["mean"]= bb2["pred"].ewm(span=k).mean()
    bb2["std"]= bb2["pred"].ewm(span=k).std()
    bb2['upper1'] = bb2['mean'] + (bb2['std'] * 1)
    bb2['lower1'] = bb2['mean'] - (bb2['std'] * 1)
    bb2['upper2'] = bb2['mean'] + (bb2['std'] * 2)
    bb2['lower2'] = bb2['mean'] - (bb2['std'] * 2)
    bb2['upper3'] = bb2['mean'] + (bb2['std'] * 3)
    bb2['lower3'] = bb2['mean'] - (bb2['std'] * 3)
    from sklearn.linear_model import Lars
    lr3 = Lars()
    lr3.fit(x_train3,y_train3)
    pred_y3 = lr3.predict(x_test3)
    y_plot3=pd.DataFrame({"pred":pred_y3})
    bb4=pd.DataFrame()
    bb4["pred"] = y_plot3["pred"]
    bb4["mean"]= bb4["pred"].ewm(span=k).mean()
    bb4["std"]= bb4["pred"].ewm(span=k).std()
    bb4['upper1'] = bb4['mean'] + (bb4['std'] * 1)
    bb4['lower1'] = bb4['mean'] - (bb4['std'] * 1)
    bb4['upper2'] = bb4['mean'] + (bb4['std'] * 2)
    bb4['lower2'] = bb4['mean'] - (bb4['std'] * 2)
    bb4['upper3'] = bb4['mean'] + (bb4['std'] * 3)
    bb4['lower3'] = bb4['mean'] - (bb4['std'] * 3)
    from sklearn import ensemble, tree
    lr4 = ensemble.BaggingRegressor(tree.DecisionTreeRegressor())
    lr4.fit(x_train4,y_train4)
    pred_y4 = lr4.predict(x_test4)
    y_plot4=pd.DataFrame({"pred":pred_y4})
    bb5=pd.DataFrame()
    bb5["pred"] = y_plot4["pred"]
    bb5["mean"]= bb5["pred"].ewm(span=k).mean()
    bb5["std"]= bb5["pred"].ewm(span=k).std()
    bb5['upper1'] = bb5['mean'] + (bb5['std'] * 1)
    bb5['lower1'] = bb5['mean'] - (bb5['std'] * 1)
    bb5['upper2'] = bb5['mean'] + (bb5['std'] * 2)
    bb5['lower2'] = bb5['mean'] - (bb5['std'] * 2)
    bb5['upper3'] = bb5['mean'] + (bb5['std'] * 3)
    bb5['lower3'] = bb5['mean'] - (bb5['std'] * 3)
    from sklearn.linear_model import RANSACRegressor
    lr5 = RANSACRegressor()
    lr5.fit(x_train5,y_train5)
    pred_y5 = lr5.predict(x_test5)
    y_plot5=pd.DataFrame({"pred":pred_y5})
    bb6=pd.DataFrame()
    bb6["pred"] = y_plot5["pred"]
    bb6["mean"]= bb6["pred"].ewm(span=k).mean()
    bb6["std"]= bb6["pred"].ewm(span=k).std()
    bb6['upper1'] = bb6['mean'] + (bb6['std'] * 1)
    bb6['lower1'] = bb6['mean'] - (bb6['std'] * 1)
    bb6['upper2'] = bb6['mean'] + (bb6['std'] * 2)
    bb6['lower2'] = bb6['mean'] - (bb6['std'] * 2)
    bb6['upper3'] = bb6['mean'] + (bb6['std'] * 3)
    bb6['lower3'] = bb6['mean'] - (bb6['std'] * 3)
    from sklearn.ensemble import GradientBoostingRegressor
    lr6 = GradientBoostingRegressor()
    lr6.fit(x_train6,y_train6)
    pred_y6 = lr6.predict(x_test6)
    y_plot6=pd.DataFrame({"pred":pred_y6})
    bb9=pd.DataFrame()
    bb9["pred"] = y_plot6["pred"]
    bb9["mean"]= bb9["pred"].ewm(span=k).mean()
    bb9["std"]= bb9["pred"].ewm(span=k).std()
    bb9['upper1'] = bb9['mean'] + (bb9['std'] * 1)
    bb9['lower1'] = bb9['mean'] - (bb9['std'] * 1)
    bb9['upper2'] = bb9['mean'] + (bb9['std'] * 2)
    bb9['lower2'] = bb9['mean'] - (bb9['std'] * 2)
    bb9['upper3'] = bb9['mean'] + (bb9['std'] * 3)
    bb9['lower3'] = bb9['mean'] - (bb9['std'] * 3)
    from sklearn.linear_model import ARDRegression
    lr7 =  ARDRegression()
    lr7.fit(x_train7,y_train7)
    pred_y7 = lr7.predict(x_test7)
    y_plot7=pd.DataFrame({"pred":pred_y7})
    bbb=pd.DataFrame()
    bbb["pred"] = y_plot7["pred"]
    bbb["mean"]= bbb["pred"].ewm(span=k).mean()
    bbb["std"]= bbb["pred"].ewm(span=k).std()
    bbb['upper1'] = bbb['mean'] + (bbb['std'] * 1)
    bbb['lower1'] = bbb['mean'] - (bbb['std'] * 1)
    bbb['upper2'] = bbb['mean'] + (bbb['std'] * 2)
    bbb['lower2'] = bbb['mean'] - (bbb['std'] * 2)
    bbb['upper3'] = bbb['mean'] + (bbb['std'] * 3)
    bbb['lower3'] = bbb['mean'] - (bbb['std'] * 3)
    bb=pd.DataFrame()
    bb["pred"]=bb1["pred"]+bb2["pred"]+bb4["pred"]+bb5["pred"]+bb6["pred"]+bb9["pred"]+bbb["pred"]
    bb["mean"]=bb1["mean"]+bb2["mean"]+bb4["mean"]+bb5["mean"]+bb6["mean"]+bb9["mean"]+bbb["mean"]
    for j in range(1,4):
        bb["upper{}".format(j)] = bb1["upper{}".format(j)]+bb2["upper{}".format(j)]+bb4["upper{}".format(j)]+bb5["upper{}".format(j)]+bb6["upper{}".format(j)]+bb9["upper{}".format(j)]+bbb["upper{}".format(j)]
        bb["lower{}".format(j)] = bb1["lower{}".format(j)]+bb2["lower{}".format(j)]+bb4["lower{}".format(j)]+bb5["lower{}".format(j)]+bb6["lower{}".format(j)]+bb9["lower{}".format(j)]+bbb["lower{}".format(j)]
    bb = bb/7
    fig=plt.figure(figsize=(20,5))
    
    plt.plot(y_test1.index,bb["pred"],label="pred",color="m")
    plt.plot(y_test1.index,bb["mean"],label="EMA",color="g")
    plt.plot(y_test1.index,bb["upper1"],label="upper1",color="c")
    plt.plot(y_test1.index,bb["lower1"],label="lower1",color="c")
    plt.plot(y_test1.index,bb["upper2"],label="upper2",color="y")
    plt.plot(y_test1.index,bb["lower2"],label="lower2",color="y")
    plt.plot(y_test1.index,bb["upper3"],label="upper3",color="r")
    plt.plot(y_test1.index,bb["lower3"],label="lower3",color="r")
    
    y_test1.plot(label="true",color="b")
    
    plt.title("mean of prediction")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid()
    plt.legend()
    
    plt.show()
    fig.savefig("img.png")
    
    text="""今日のドルカナダAI予測
    
    今日ドルカナダの予測値は{}です。
    
    また、約68.3％の確率で{}から{}内、
    
    約95.4％の確率で{}から{}内、
    
    約99.7％の確率で{}から{}内に
    
    値動きは収束します。""".format(round(bb.at[len(bb)-1,"pred"],6),round(bb.at[len(bb)-1,"lower1"],6),
                         round(bb.at[len(bb)-1,"upper1"],6),round(bb.at[len(bb)-1,"lower2"],6),
                         round(bb.at[len(bb)-1,"upper2"],6),round(bb.at[len(bb)-1,"lower3"],6),
                         round(bb.at[len(bb)-1,"upper3"]),6)
    
    caption="""このAI予測は約5年分のダウ、原油、米国10年債、金、VIX、BTC、ドルカナダ前日終値から、
    
    7つの手法で予測します。
    
    予測値はその平均です。"""
    
    return text,caption


def usdchf_predict(k=10):
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=56)
    start= three_month_ago
    end=yesterday
    usdchf=pd.DataFrame(pdr.DataReader("DEXSZUS","fred",start,end))
    btc=pd.DataFrame(np.log(pdr.DataReader("CBBTCUSD","fred",start,end)))
    dow=pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    eurusd=pd.DataFrame(np.log(pdr.DataReader("DEXUSEU","fred",start,end)))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    lusdchf=np.log(usdchf)
    data=pd.concat([usdchf,eurusd,dow,us10y,gold,vix,btc],axis=1)
    data=data.rename(columns={"DEXUSEU":"EURUSD","DGS10Y":"US10Y","GOLDAMGBD228NLBM":"XAUUSD","VIXCLS":"VIX",
                              "DEXSZUS":"USDCHF","CBBTCUSD":"BTCUSD"})
    data["Log_USDCHF"]=lusdchf
    data=data.dropna()
    
    y_test1 = data.loc["2019-01-02":yesterday,"USDCHF"]
    x_train1 = data.loc["2016-01-01":"2018-12-31","EURUSD":"Log_USDCHF"]
    y_train1 = data.loc["2016-01-02":"2019-01-01","USDCHF"]
    x_test1 = data.loc["2019-01-01":yesterday,"EURUSD":"Log_USDCHF"]
    
    x_train2 = x_train1
    y_train2 = y_train1
    x_test2 = x_test1
    
    x_train3 = x_train1
    y_train3 = y_train1
    x_test3 = x_test1
    
    x_train4 = x_train1
    y_train4 = y_train1
    x_test4 = x_test1
    
    x_train5 = x_train1
    y_train5 = y_train1
    x_test5 = x_test1
    
    x_train6 = x_train1
    y_train6 = y_train1
    x_test6 = x_test1
    
    x_train7 = x_train1
    y_train7 = y_train1
    x_test7 = x_test1
    
   
    from sklearn.linear_model import LinearRegression
    lr1 = LinearRegression()
    lr1.fit(x_train1,y_train1)
    pred_y1 = lr1.predict(x_test1)
    y_plot1=pd.DataFrame({"pred":pred_y1})
    bb1=pd.DataFrame()
    bb1["pred"] = y_plot1["pred"]
    bb1["mean"]= bb1["pred"].ewm(span=k).mean()
    bb1["std"]= bb1["pred"].ewm(span=k).std()
    bb1['upper1'] = bb1['mean'] + (bb1['std'] * 1)
    bb1['lower1'] = bb1['mean'] - (bb1['std'] * 1)
    bb1['upper2'] = bb1['mean'] + (bb1['std'] * 2)
    bb1['lower2'] = bb1['mean'] - (bb1['std'] * 2)
    bb1['upper3'] = bb1['mean'] + (bb1['std'] * 3)
    bb1['lower3'] = bb1['mean'] - (bb1['std'] * 3)
    
    from sklearn.linear_model import Lars
    lr2 = Lars()
    lr2.fit(x_train2,y_train2)
    pred_y2 = lr2.predict(x_test2)
    y_plot2=pd.DataFrame({"pred":pred_y2})
    bb2=pd.DataFrame()
    bb2["pred"] = y_plot2["pred"]
    bb2["mean"]= bb2["pred"].ewm(span=k).mean()
    bb2["std"]= bb2["pred"].ewm(span=k).std()
    bb2['upper1'] = bb2['mean'] + (bb2['std'] * 1)
    bb2['lower1'] = bb2['mean'] - (bb2['std'] * 1)
    bb2['upper2'] = bb2['mean'] + (bb2['std'] * 2)
    bb2['lower2'] = bb2['mean'] - (bb2['std'] * 2)
    bb2['upper3'] = bb2['mean'] + (bb2['std'] * 3)
    bb2['lower3'] = bb2['mean'] - (bb2['std'] * 3)
    from sklearn.linear_model import Lars
    lr3 = Lars()
    lr3.fit(x_train3,y_train3)
    pred_y3 = lr3.predict(x_test3)
    y_plot3=pd.DataFrame({"pred":pred_y3})
    bb4=pd.DataFrame()
    bb4["pred"] = y_plot3["pred"]
    bb4["mean"]= bb4["pred"].ewm(span=k).mean()
    bb4["std"]= bb4["pred"].ewm(span=k).std()
    bb4['upper1'] = bb4['mean'] + (bb4['std'] * 1)
    bb4['lower1'] = bb4['mean'] - (bb4['std'] * 1)
    bb4['upper2'] = bb4['mean'] + (bb4['std'] * 2)
    bb4['lower2'] = bb4['mean'] - (bb4['std'] * 2)
    bb4['upper3'] = bb4['mean'] + (bb4['std'] * 3)
    bb4['lower3'] = bb4['mean'] - (bb4['std'] * 3)
    from sklearn import ensemble, tree
    lr4 = ensemble.BaggingRegressor(tree.DecisionTreeRegressor())
    lr4.fit(x_train4,y_train4)
    pred_y4 = lr4.predict(x_test4)
    y_plot4=pd.DataFrame({"pred":pred_y4})
    bb5=pd.DataFrame()
    bb5["pred"] = y_plot4["pred"]
    bb5["mean"]= bb5["pred"].ewm(span=k).mean()
    bb5["std"]= bb5["pred"].ewm(span=k).std()
    bb5['upper1'] = bb5['mean'] + (bb5['std'] * 1)
    bb5['lower1'] = bb5['mean'] - (bb5['std'] * 1)
    bb5['upper2'] = bb5['mean'] + (bb5['std'] * 2)
    bb5['lower2'] = bb5['mean'] - (bb5['std'] * 2)
    bb5['upper3'] = bb5['mean'] + (bb5['std'] * 3)
    bb5['lower3'] = bb5['mean'] - (bb5['std'] * 3)
    from sklearn.neighbors import KNeighborsRegressor
    lr5 = KNeighborsRegressor()
    lr5.fit(x_train5,y_train5)
    pred_y5 = lr5.predict(x_test5)
    y_plot5=pd.DataFrame({"pred":pred_y5})
    bb6=pd.DataFrame()
    bb6["pred"] = y_plot5["pred"]
    bb6["mean"]= bb6["pred"].ewm(span=k).mean()
    bb6["std"]= bb6["pred"].ewm(span=k).std()
    bb6['upper1'] = bb6['mean'] + (bb6['std'] * 1)
    bb6['lower1'] = bb6['mean'] - (bb6['std'] * 1)
    bb6['upper2'] = bb6['mean'] + (bb6['std'] * 2)
    bb6['lower2'] = bb6['mean'] - (bb6['std'] * 2)
    bb6['upper3'] = bb6['mean'] + (bb6['std'] * 3)
    bb6['lower3'] = bb6['mean'] - (bb6['std'] * 3)
    from sklearn.ensemble import GradientBoostingRegressor
    lr6 = GradientBoostingRegressor()
    lr6.fit(x_train6,y_train6)
    pred_y6 = lr6.predict(x_test6)
    y_plot6=pd.DataFrame({"pred":pred_y6})
    bb9=pd.DataFrame()
    bb9["pred"] = y_plot6["pred"]
    bb9["mean"]= bb9["pred"].ewm(span=k).mean()
    bb9["std"]= bb9["pred"].ewm(span=k).std()
    bb9['upper1'] = bb9['mean'] + (bb9['std'] * 1)
    bb9['lower1'] = bb9['mean'] - (bb9['std'] * 1)
    bb9['upper2'] = bb9['mean'] + (bb9['std'] * 2)
    bb9['lower2'] = bb9['mean'] - (bb9['std'] * 2)
    bb9['upper3'] = bb9['mean'] + (bb9['std'] * 3)
    bb9['lower3'] = bb9['mean'] - (bb9['std'] * 3)
    from sklearn.linear_model import ARDRegression
    lr7 =  ARDRegression()
    lr7.fit(x_train7,y_train7)
    pred_y7 = lr7.predict(x_test7)
    y_plot7=pd.DataFrame({"pred":pred_y7})
    bbb=pd.DataFrame()
    bbb["pred"] = y_plot7["pred"]
    bbb["mean"]= bbb["pred"].ewm(span=k).mean()
    bbb["std"]= bbb["pred"].ewm(span=k).std()
    bbb['upper1'] = bbb['mean'] + (bbb['std'] * 1)
    bbb['lower1'] = bbb['mean'] - (bbb['std'] * 1)
    bbb['upper2'] = bbb['mean'] + (bbb['std'] * 2)
    bbb['lower2'] = bbb['mean'] - (bbb['std'] * 2)
    bbb['upper3'] = bbb['mean'] + (bbb['std'] * 3)
    bbb['lower3'] = bbb['mean'] - (bbb['std'] * 3)
    bb=pd.DataFrame()
    bb["pred"]=bb1["pred"]+bb2["pred"]+bb4["pred"]+bb5["pred"]+bb6["pred"]+bb9["pred"]+bbb["pred"]
    bb["mean"]=bb1["mean"]+bb2["mean"]+bb4["mean"]+bb5["mean"]+bb6["mean"]+bb9["mean"]+bbb["mean"]
    for j in range(1,4):
        bb["upper{}".format(j)] = bb1["upper{}".format(j)]+bb2["upper{}".format(j)]+bb4["upper{}".format(j)]+bb5["upper{}".format(j)]+bb6["upper{}".format(j)]+bb9["upper{}".format(j)]+bbb["upper{}".format(j)]
        bb["lower{}".format(j)] = bb1["lower{}".format(j)]+bb2["lower{}".format(j)]+bb4["lower{}".format(j)]+bb5["lower{}".format(j)]+bb6["lower{}".format(j)]+bb9["lower{}".format(j)]+bbb["lower{}".format(j)]
    bb = bb/7
    fig=plt.figure(figsize=(20,5))
    
    plt.plot(y_test1.index,bb["pred"],label="pred",color="m")
    plt.plot(y_test1.index,bb["mean"],label="EMA",color="g")
    plt.plot(y_test1.index,bb["upper1"],label="upper1",color="c")
    plt.plot(y_test1.index,bb["lower1"],label="lower1",color="c")
    plt.plot(y_test1.index,bb["upper2"],label="upper2",color="y")
    plt.plot(y_test1.index,bb["lower2"],label="lower2",color="y")
    plt.plot(y_test1.index,bb["upper3"],label="upper3",color="r")
    plt.plot(y_test1.index,bb["lower3"],label="lower3",color="r")
    
    y_test1.plot(label="true",color="b")
    
    plt.title("mean of prediction")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid()
    plt.legend()
    
    plt.show()
    fig.savefig("img.png")
    
    text="""今日のドルフラAI予測
    
    今日ドルフランの予測値は{}です。
    
    また、約68.3％の確率で{}から{}内、
    
    約95.4％の確率で{}から{}内、
    
    約99.7％の確率で{}から{}内に
    
    値動きは収束します。""".format(round(bb.at[len(bb)-1,"pred"],6),round(bb.at[len(bb)-1,"lower1"],6),
                         round(bb.at[len(bb)-1,"upper1"],6),round(bb.at[len(bb)-1,"lower2"],6),
                         round(bb.at[len(bb)-1,"upper2"],6),round(bb.at[len(bb)-1,"lower3"],6),
                         round(bb.at[len(bb)-1,"upper3"]),6)
    
    caption="""このAI予測は約5年分のダウ、ユーロドル、米国10年債、金、VIX、BTC、ドルフラン前日終値から、
    
    7つの手法で予測します。
    
    予測値はその平均です。"""
    
    return text,caption


def gold_predict(k=10):
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=56)
    start= three_month_ago
    end=yesterday
    wti=pd.DataFrame(np.log(pdr.DataReader("DCOILWTICO","fred",start,end)))
    dow=pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    usdjpy =pd.DataFrame(pdr.DataReader("DEXJPUS","fred",start,end))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    eurusd=pd.DataFrame(np.log(pdr.DataReader("DEXUSEU","fred",start,end)))
    btc=pd.DataFrame(np.log(pdr.DataReader("CBBTCUSD","fred",start,end)))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end))
    usdjpy=np.log(usdjpy)
    lgold=np.log(gold)
    data=pd.concat([gold,wti,usdjpy,us10y,eurusd,btc,dow,vix],axis=1)
    data=data.rename(columns={"DEXJPUS":"USDJPY","DGS10Y":"US10Y","GOLDAMGBD228NLBM":"XAUUSD",
                          "VIXCLS":"VIX","CBBTCUSD":"BTCUSD","DCOILWTICO":"WTIUSD","DEXUSEU":"EURUSD"})
    data["Log_GOLD"]=lgold
    data=data.dropna()
    
    y_test1 = data.loc["2019-01-02":yesterday,"XAUUSD"]
    x_train1 = data.loc["2016-01-01":"2018-12-31","WTIUSD":"Log_GOLD"]
    y_train1 = data.loc["2016-01-02":"2019-01-01","XAUUSD"]
    x_test1 = data.loc["2019-01-01":yesterday,"WTIUSD":"Log_GOLD"]
    
    x_train2 = x_train1
    y_train2 = y_train1
    x_test2 = x_test1
    
    x_train3 = x_train1
    y_train3 = y_train1
    x_test3 = x_test1
    
    x_train4 = x_train1
    y_train4 = y_train1
    x_test4 = x_test1
    
    x_train5 = x_train1
    y_train5 = y_train1
    x_test5 = x_test1
    
    x_train6 = x_train1
    y_train6 = y_train1
    x_test6 = x_test1
    
    x_train7 = x_train1
    y_train7 = y_train1
    x_test7 = x_test1
    
   
    from sklearn.linear_model import LinearRegression
    lr1 = LinearRegression()
    lr1.fit(x_train1,y_train1)
    pred_y1 = lr1.predict(x_test1)
    y_plot1=pd.DataFrame({"pred":pred_y1})
    bb1=pd.DataFrame()
    bb1["pred"] = y_plot1["pred"]
    bb1["mean"]= bb1["pred"].ewm(span=k).mean()
    bb1["std"]= bb1["pred"].ewm(span=k).std()
    bb1['upper1'] = bb1['mean'] + (bb1['std'] * 1)
    bb1['lower1'] = bb1['mean'] - (bb1['std'] * 1)
    bb1['upper2'] = bb1['mean'] + (bb1['std'] * 2)
    bb1['lower2'] = bb1['mean'] - (bb1['std'] * 2)
    bb1['upper3'] = bb1['mean'] + (bb1['std'] * 3)
    bb1['lower3'] = bb1['mean'] - (bb1['std'] * 3)
    
    from sklearn.linear_model import Lars
    lr2 = Lars()
    lr2.fit(x_train2,y_train2)
    pred_y2 = lr2.predict(x_test2)
    y_plot2=pd.DataFrame({"pred":pred_y2})
    bb2=pd.DataFrame()
    bb2["pred"] = y_plot2["pred"]
    bb2["mean"]= bb2["pred"].ewm(span=k).mean()
    bb2["std"]= bb2["pred"].ewm(span=k).std()
    bb2['upper1'] = bb2['mean'] + (bb2['std'] * 1)
    bb2['lower1'] = bb2['mean'] - (bb2['std'] * 1)
    bb2['upper2'] = bb2['mean'] + (bb2['std'] * 2)
    bb2['lower2'] = bb2['mean'] - (bb2['std'] * 2)
    bb2['upper3'] = bb2['mean'] + (bb2['std'] * 3)
    bb2['lower3'] = bb2['mean'] - (bb2['std'] * 3)
    from sklearn.linear_model import HuberRegressor
    lr3 = HuberRegressor()
    lr3.fit(x_train3,y_train3)
    pred_y3 = lr3.predict(x_test3)
    y_plot3=pd.DataFrame({"pred":pred_y3})
    bb4=pd.DataFrame()
    bb4["pred"] = y_plot3["pred"]
    bb4["mean"]= bb4["pred"].ewm(span=k).mean()
    bb4["std"]= bb4["pred"].ewm(span=k).std()
    bb4['upper1'] = bb4['mean'] + (bb4['std'] * 1)
    bb4['lower1'] = bb4['mean'] - (bb4['std'] * 1)
    bb4['upper2'] = bb4['mean'] + (bb4['std'] * 2)
    bb4['lower2'] = bb4['mean'] - (bb4['std'] * 2)
    bb4['upper3'] = bb4['mean'] + (bb4['std'] * 3)
    bb4['lower3'] = bb4['mean'] - (bb4['std'] * 3)
    from sklearn import ensemble, tree
    lr4 = ensemble.BaggingRegressor(tree.DecisionTreeRegressor())
    lr4.fit(x_train4,y_train4)
    pred_y4 = lr4.predict(x_test4)
    y_plot4=pd.DataFrame({"pred":pred_y4})
    bb5=pd.DataFrame()
    bb5["pred"] = y_plot4["pred"]
    bb5["mean"]= bb5["pred"].ewm(span=k).mean()
    bb5["std"]= bb5["pred"].ewm(span=k).std()
    bb5['upper1'] = bb5['mean'] + (bb5['std'] * 1)
    bb5['lower1'] = bb5['mean'] - (bb5['std'] * 1)
    bb5['upper2'] = bb5['mean'] + (bb5['std'] * 2)
    bb5['lower2'] = bb5['mean'] - (bb5['std'] * 2)
    bb5['upper3'] = bb5['mean'] + (bb5['std'] * 3)
    bb5['lower3'] = bb5['mean'] - (bb5['std'] * 3)
    from sklearn.neighbors import KNeighborsRegressor
    lr5 = KNeighborsRegressor()
    lr5.fit(x_train5,y_train5)
    pred_y5 = lr5.predict(x_test5)
    y_plot5=pd.DataFrame({"pred":pred_y5})
    bb6=pd.DataFrame()
    bb6["pred"] = y_plot5["pred"]
    bb6["mean"]= bb6["pred"].ewm(span=k).mean()
    bb6["std"]= bb6["pred"].ewm(span=k).std()
    bb6['upper1'] = bb6['mean'] + (bb6['std'] * 1)
    bb6['lower1'] = bb6['mean'] - (bb6['std'] * 1)
    bb6['upper2'] = bb6['mean'] + (bb6['std'] * 2)
    bb6['lower2'] = bb6['mean'] - (bb6['std'] * 2)
    bb6['upper3'] = bb6['mean'] + (bb6['std'] * 3)
    bb6['lower3'] = bb6['mean'] - (bb6['std'] * 3)
    from sklearn.ensemble import GradientBoostingRegressor
    lr6 = GradientBoostingRegressor()
    lr6.fit(x_train6,y_train6)
    pred_y6 = lr6.predict(x_test6)
    y_plot6=pd.DataFrame({"pred":pred_y6})
    bb9=pd.DataFrame()
    bb9["pred"] = y_plot6["pred"]
    bb9["mean"]= bb9["pred"].ewm(span=k).mean()
    bb9["std"]= bb9["pred"].ewm(span=k).std()
    bb9['upper1'] = bb9['mean'] + (bb9['std'] * 1)
    bb9['lower1'] = bb9['mean'] - (bb9['std'] * 1)
    bb9['upper2'] = bb9['mean'] + (bb9['std'] * 2)
    bb9['lower2'] = bb9['mean'] - (bb9['std'] * 2)
    bb9['upper3'] = bb9['mean'] + (bb9['std'] * 3)
    bb9['lower3'] = bb9['mean'] - (bb9['std'] * 3)
    from sklearn.linear_model import ARDRegression
    lr7 =  ARDRegression()
    lr7.fit(x_train7,y_train7)
    pred_y7 = lr7.predict(x_test7)
    y_plot7=pd.DataFrame({"pred":pred_y7})
    bbb=pd.DataFrame()
    bbb["pred"] = y_plot7["pred"]
    bbb["mean"]= bbb["pred"].ewm(span=k).mean()
    bbb["std"]= bbb["pred"].ewm(span=k).std()
    bbb['upper1'] = bbb['mean'] + (bbb['std'] * 1)
    bbb['lower1'] = bbb['mean'] - (bbb['std'] * 1)
    bbb['upper2'] = bbb['mean'] + (bbb['std'] * 2)
    bbb['lower2'] = bbb['mean'] - (bbb['std'] * 2)
    bbb['upper3'] = bbb['mean'] + (bbb['std'] * 3)
    bbb['lower3'] = bbb['mean'] - (bbb['std'] * 3)
    bb=pd.DataFrame()
    bb["pred"]=bb1["pred"]+bb2["pred"]+bb4["pred"]+bb5["pred"]+bb6["pred"]+bb9["pred"]+bbb["pred"]
    bb["mean"]=bb1["mean"]+bb2["mean"]+bb4["mean"]+bb5["mean"]+bb6["mean"]+bb9["mean"]+bbb["mean"]
    for j in range(1,4):
        bb["upper{}".format(j)] = bb1["upper{}".format(j)]+bb2["upper{}".format(j)]+bb4["upper{}".format(j)]+bb5["upper{}".format(j)]+bb6["upper{}".format(j)]+bb9["upper{}".format(j)]+bbb["upper{}".format(j)]
        bb["lower{}".format(j)] = bb1["lower{}".format(j)]+bb2["lower{}".format(j)]+bb4["lower{}".format(j)]+bb5["lower{}".format(j)]+bb6["lower{}".format(j)]+bb9["lower{}".format(j)]+bbb["lower{}".format(j)]
    bb = bb/7
    fig=plt.figure(figsize=(20,5))
    
    plt.plot(y_test1.index,bb["pred"],label="pred",color="m")
    plt.plot(y_test1.index,bb["mean"],label="EMA",color="g")
    plt.plot(y_test1.index,bb["upper1"],label="upper1",color="c")
    plt.plot(y_test1.index,bb["lower1"],label="lower1",color="c")
    plt.plot(y_test1.index,bb["upper2"],label="upper2",color="y")
    plt.plot(y_test1.index,bb["lower2"],label="lower2",color="y")
    plt.plot(y_test1.index,bb["upper3"],label="upper3",color="r")
    plt.plot(y_test1.index,bb["lower3"],label="lower3",color="r")
    
    y_test1.plot(label="true",color="b")
    
    plt.title("mean of prediction")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid()
    plt.legend()
    
    plt.show()
    fig.savefig("img.png")
    
    text="""今日のゴールドAI予測
    
    今日のゴールドの予測値は{}です。
    
    また、約68.3％の確率で{}から{}内、
    
    約95.4％の確率で{}から{}内、
    
    約99.7％の確率で{}から{}内に
    
    値動きは収束します。""".format(round(bb.at[len(bb)-1,"pred"],2),round(bb.at[len(bb)-1,"lower1"],2),
                         round(bb.at[len(bb)-1,"upper1"],2),round(bb.at[len(bb)-1,"lower2"],2),
                         round(bb.at[len(bb)-1,"upper2"],2),round(bb.at[len(bb)-1,"lower3"],2),
                         round(bb.at[len(bb)-1,"upper3"],2))
    
    caption="""このAI予測は約5年分の日経平均、ユーロドル、仮想通貨、ダウ、ドル円、米国10年債、VIX、金前日終値から、
    
    7つの手法で予測します。
    
    予測値はその平均です。"""
    return text,caption

def btc_predict(k=10):
    today = date.today()
    yesterday = today - timedelta(days=1)
    three_month_ago = today - relativedelta(months=56)
    start= three_month_ago
    end=yesterday
    eurusd=pd.DataFrame(pdr.DataReader("DEXUSEU","fred",start,end))
    btc=pd.DataFrame(pdr.DataReader("CBBTCUSD","fred",start,end))
    dow=pd.DataFrame(np.log(pdr.DataReader("DJIA","fred",start,end)))
    wti=pd.DataFrame(np.log(pdr.DataReader("DCOILWTICO","fred",start,end)))
    us10y = pd.DataFrame(np.log(pdr.DataReader("DGS10","fred",start,end)))
    vix = pd.DataFrame(np.log(pdr.DataReader("VIXCLS","fred",start,end)))
    gold= pd.DataFrame(np.log(pdr.DataReader("GOLDAMGBD228NLBM","fred",start,end)))
    lbtc=np.log(btc)
    data=pd.concat([btc,wti,dow,us10y,eurusd,gold,vix],axis=1)
    data=data.rename(columns={"DCOILWTICO":"WTIUSD","DGS10Y":"US10Y","GOLDAMGBD228NLBM":"XAUUSD","VIXCLS":"VIX",
                              "DEXUSEU":"EURUSD","CBBTCUSD":"BTCUSD"})
    data["Log_BTCUSD"]=lbtc
    data=data.dropna()
    
    y_test1 = data.loc["2019-01-02":yesterday,"BTCUSD"]
    x_train1= data.loc["2016-01-01":"2018-12-31","WTIUSD":"Log_BTCUSD"]
    y_train1 = data.loc["2016-01-02":"2019-01-01","BTCUSD"]
    x_test1 = data.loc["2019-01-01":yesterday,"WTIUSD":"Log_BTCUSD"]
    
    x_train2 = x_train1
    y_train2 = y_train1
    x_test2 = x_test1
    
    x_train3 = x_train1
    y_train3 = y_train1
    x_test3 = x_test1
    
    x_train4 = x_train1
    y_train4 = y_train1
    x_test4 = x_test1
    
    x_train5 = x_train1
    y_train5 = y_train1
    x_test5 = x_test1
    
    x_train6 = x_train1
    y_train6 = y_train1
    x_test6 = x_test1
    
    x_train7 = x_train1
    y_train7 = y_train1
    x_test7 = x_test1
    
   
    from sklearn.linear_model import LinearRegression
    lr1 = LinearRegression()
    lr1.fit(x_train1,y_train1)
    pred_y1 = lr1.predict(x_test1)
    y_plot1=pd.DataFrame({"pred":pred_y1})
    bb1=pd.DataFrame()
    bb1["pred"] = y_plot1["pred"]
    bb1["mean"]= bb1["pred"].ewm(span=k).mean()
    bb1["std"]= bb1["pred"].ewm(span=k).std()
    bb1['upper1'] = bb1['mean'] + (bb1['std'] * 1)
    bb1['lower1'] = bb1['mean'] - (bb1['std'] * 1)
    bb1['upper2'] = bb1['mean'] + (bb1['std'] * 2)
    bb1['lower2'] = bb1['mean'] - (bb1['std'] * 2)
    bb1['upper3'] = bb1['mean'] + (bb1['std'] * 3)
    bb1['lower3'] = bb1['mean'] - (bb1['std'] * 3)
    
    from sklearn.linear_model import Lars
    lr2 = Lars()
    lr2.fit(x_train2,y_train2)
    pred_y2 = lr2.predict(x_test2)
    y_plot2=pd.DataFrame({"pred":pred_y2})
    bb2=pd.DataFrame()
    bb2["pred"] = y_plot2["pred"]
    bb2["mean"]= bb2["pred"].ewm(span=k).mean()
    bb2["std"]= bb2["pred"].ewm(span=k).std()
    bb2['upper1'] = bb2['mean'] + (bb2['std'] * 1)
    bb2['lower1'] = bb2['mean'] - (bb2['std'] * 1)
    bb2['upper2'] = bb2['mean'] + (bb2['std'] * 2)
    bb2['lower2'] = bb2['mean'] - (bb2['std'] * 2)
    bb2['upper3'] = bb2['mean'] + (bb2['std'] * 3)
    bb2['lower3'] = bb2['mean'] - (bb2['std'] * 3)
    from sklearn.linear_model import Lasso
    lr3 =  Lasso()
    lr3.fit(x_train3,y_train3)
    pred_y3 = lr3.predict(x_test3)
    y_plot3=pd.DataFrame({"pred":pred_y3})
    bb4=pd.DataFrame()
    bb4["pred"] = y_plot3["pred"]
    bb4["mean"]= bb4["pred"].ewm(span=k).mean()
    bb4["std"]= bb4["pred"].ewm(span=k).std()
    bb4['upper1'] = bb4['mean'] + (bb4['std'] * 1)
    bb4['lower1'] = bb4['mean'] - (bb4['std'] * 1)
    bb4['upper2'] = bb4['mean'] + (bb4['std'] * 2)
    bb4['lower2'] = bb4['mean'] - (bb4['std'] * 2)
    bb4['upper3'] = bb4['mean'] + (bb4['std'] * 3)
    bb4['lower3'] = bb4['mean'] - (bb4['std'] * 3)
    from sklearn import ensemble, tree
    lr4 = ensemble.BaggingRegressor(tree.DecisionTreeRegressor())
    lr4.fit(x_train4,y_train4)
    pred_y4 = lr4.predict(x_test4)
    y_plot4=pd.DataFrame({"pred":pred_y4})
    bb5=pd.DataFrame()
    bb5["pred"] = y_plot4["pred"]
    bb5["mean"]= bb5["pred"].ewm(span=k).mean()
    bb5["std"]= bb5["pred"].ewm(span=k).std()
    bb5['upper1'] = bb5['mean'] + (bb5['std'] * 1)
    bb5['lower1'] = bb5['mean'] - (bb5['std'] * 1)
    bb5['upper2'] = bb5['mean'] + (bb5['std'] * 2)
    bb5['lower2'] = bb5['mean'] - (bb5['std'] * 2)
    bb5['upper3'] = bb5['mean'] + (bb5['std'] * 3)
    bb5['lower3'] = bb5['mean'] - (bb5['std'] * 3)
    from sklearn.neighbors import KNeighborsRegressor
    lr5 = KNeighborsRegressor()
    lr5.fit(x_train5,y_train5)
    pred_y5 = lr5.predict(x_test5)
    y_plot5=pd.DataFrame({"pred":pred_y5})
    bb6=pd.DataFrame()
    bb6["pred"] = y_plot5["pred"]
    bb6["mean"]= bb6["pred"].ewm(span=k).mean()
    bb6["std"]= bb6["pred"].ewm(span=k).std()
    bb6['upper1'] = bb6['mean'] + (bb6['std'] * 1)
    bb6['lower1'] = bb6['mean'] - (bb6['std'] * 1)
    bb6['upper2'] = bb6['mean'] + (bb6['std'] * 2)
    bb6['lower2'] = bb6['mean'] - (bb6['std'] * 2)
    bb6['upper3'] = bb6['mean'] + (bb6['std'] * 3)
    bb6['lower3'] = bb6['mean'] - (bb6['std'] * 3)
    from sklearn.ensemble import GradientBoostingRegressor
    lr6 = GradientBoostingRegressor()
    lr6.fit(x_train6,y_train6)
    pred_y6 = lr6.predict(x_test6)
    y_plot6=pd.DataFrame({"pred":pred_y6})
    bb9=pd.DataFrame()
    bb9["pred"] = y_plot6["pred"]
    bb9["mean"]= bb9["pred"].ewm(span=k).mean()
    bb9["std"]= bb9["pred"].ewm(span=k).std()
    bb9['upper1'] = bb9['mean'] + (bb9['std'] * 1)
    bb9['lower1'] = bb9['mean'] - (bb9['std'] * 1)
    bb9['upper2'] = bb9['mean'] + (bb9['std'] * 2)
    bb9['lower2'] = bb9['mean'] - (bb9['std'] * 2)
    bb9['upper3'] = bb9['mean'] + (bb9['std'] * 3)
    bb9['lower3'] = bb9['mean'] - (bb9['std'] * 3)
    from sklearn.linear_model import ARDRegression
    lr7 =  ARDRegression()
    lr7.fit(x_train7,y_train7)
    pred_y7 = lr7.predict(x_test7)
    y_plot7=pd.DataFrame({"pred":pred_y7})
    bbb=pd.DataFrame()
    bbb["pred"] = y_plot7["pred"]
    bbb["mean"]= bbb["pred"].ewm(span=k).mean()
    bbb["std"]= bbb["pred"].ewm(span=k).std()
    bbb['upper1'] = bbb['mean'] + (bbb['std'] * 1)
    bbb['lower1'] = bbb['mean'] - (bbb['std'] * 1)
    bbb['upper2'] = bbb['mean'] + (bbb['std'] * 2)
    bbb['lower2'] = bbb['mean'] - (bbb['std'] * 2)
    bbb['upper3'] = bbb['mean'] + (bbb['std'] * 3)
    bbb['lower3'] = bbb['mean'] - (bbb['std'] * 3)
    bb=pd.DataFrame()
    bb["pred"]=bb1["pred"]+bb2["pred"]+bb4["pred"]+bb5["pred"]+bb6["pred"]+bb9["pred"]+bbb["pred"]
    bb["mean"]=bb1["mean"]+bb2["mean"]+bb4["mean"]+bb5["mean"]+bb6["mean"]+bb9["mean"]+bbb["mean"]
    for j in range(1,4):
        bb["upper{}".format(j)] = bb1["upper{}".format(j)]+bb2["upper{}".format(j)]+bb4["upper{}".format(j)]+bb5["upper{}".format(j)]+bb6["upper{}".format(j)]+bb9["upper{}".format(j)]+bbb["upper{}".format(j)]
        bb["lower{}".format(j)] = bb1["lower{}".format(j)]+bb2["lower{}".format(j)]+bb4["lower{}".format(j)]+bb5["lower{}".format(j)]+bb6["lower{}".format(j)]+bb9["lower{}".format(j)]+bbb["lower{}".format(j)]
    bb = bb/7
    fig=plt.figure(figsize=(20,5))
    
    plt.plot(y_test1.index,bb["pred"],label="pred",color="m")
    plt.plot(y_test1.index,bb["mean"],label="EMA",color="g")
    plt.plot(y_test1.index,bb["upper1"],label="upper1",color="c")
    plt.plot(y_test1.index,bb["lower1"],label="lower1",color="c")
    plt.plot(y_test1.index,bb["upper2"],label="upper2",color="y")
    plt.plot(y_test1.index,bb["lower2"],label="lower2",color="y")
    plt.plot(y_test1.index,bb["upper3"],label="upper3",color="r")
    plt.plot(y_test1.index,bb["lower3"],label="lower3",color="r")
    
    y_test1.plot(label="true",color="b")
    
    plt.title("mean of prediction")
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text="""今日のビットコインAI予測
    
    今日のビットコインの予測値は{}です。
    
    また、約68.3％の確率で{}から{}内、
    
    約95.4％の確率で{}から{}内、
    
    約99.7％の確率で{}から{}内に
    
    値動きは収束します。""".format(round(bb.at[len(bb)-1,"pred"],2),round(bb.at[len(bb)-1,"lower1"],2),
                         round(bb.at[len(bb)-1,"upper1"],2),round(bb.at[len(bb)-1,"lower2"],2),
                         round(bb.at[len(bb)-1,"upper2"],2),round(bb.at[len(bb)-1,"lower3"],2),
                         round(bb.at[len(bb)-1,"upper3"],2))
    
    caption="""このAI予測は約5年分の日経平均、ドル円、米国10年債、金、ユーロドル、原油、VIX、ダウ、ビットコイン前日終値から、
    
    7つの手法で予測します。
    
    予測値はその平均です。"""
    
    return text,caption

def get_tweets():
    start_date = datetime.now() - timedelta(days=3)
    end_date = datetime.now() - timedelta(days=2)
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    columns_name=["TW_ID","TW_TIME","TW_TEXT","FAV","RT","FOLLOWER"]
    username="YuKiYa_FX"
    tweet_data = []
    tmpTweets = api.user_timeline(username)
    for tweet in tmpTweets:
        if tweet.created_at < end_date and tweet.created_at > start_date:
            tweet_data.append([tweet.id,tweet.created_at,tweet.text.replace('\n',''),tweet.favorite_count,tweet.retweet_count,followers_count])
    data = pd.DataFrame(tweet_data,columns=columns_name)
    fig = plt.figure(figsize=(20,5))
    plt.plot(data["TW_TIME"],data["FAV"],color='lime',linestyle='solid',linewidth = 2.0, label='Favorite' ,marker='o')
    plt.plot(data["TW_TIME"],data["RT"],color='cyan',linestyle='dashed',linewidth = 1.0, label='RT' ,marker='x')
    plt.xlabel("datetime")
    plt.ylabel("Count")
    plt.title("Movement of Fav and RT of tweets {}".format(end_date.date()))
    plt.legend()
    plt.show()
    fig.savefig("img.png")
    
    text = """{}のツイートに対する「いいね！」の平均値は{}、最大値は{}、最小値は{}で、合計{}の「いいね！」を獲得しました。
    
    また、「RT」の平均値は{}、最大値は{}、最小値は{}で、合計{}の「RT」を獲得しました。
    
    ツイート数は{}件でした。""".format(end_date.date().strftime("%Y年%m月%d日"),
                            round(data["FAV"].mean(),1),data["FAV"].max(),data["FAV"].min(),data["FAV"].sum(),
                            round(data["RT"].mean(),1),data["RT"].max(),data["RT"].min(),data["RT"].sum(),
                            len(data))
    
    #2つのAPIを記述しないとリフレッシュトークンを3600秒毎に発行し続けなければならない
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

    #認証情報設定
    #ダウンロードしたjsonファイル名をクレデンシャル変数に設定（秘密鍵、Pythonファイルから読み込みしやすい位置に置く）
    credentials = ServiceAccountCredentials.from_json_keyfile_name('thinking-window-263207-9906732aa6f5.json', scope)

    #OAuth2の資格情報を使用してGoogle APIにログインします。
    gc = gspread.authorize(credentials)

    #共有設定したスプレッドシートキーを変数[SPREADSHEET_KEY]に格納する。
    SPREADSHEET_KEY = 'https://docs.google.com/spreadsheets/d/1cd1aGOmZ08FEyvbCTxVmA3g7JgfJgLdd5UmgqZiqTlI/edit#gid=0'

    #共有設定したスプレッドシートのシート1を開く
    worksheet = gc.open_by_key(SPREADSHEET_KEY).raw_data

    values = worksheet.get_all_values()
    last_row = len(values) + 1
    worksheet.update_cell(last_row,1,date.today() - timedelta(days=1))
    worksheet.update_cell(last_row,2,data["FAV"].sum())
    worksheet.update_cell(last_row,3,data["RT"].sum())
    worksheet.update_cell(last_row,4,round(data["FAV"].mean(),2))
    worksheet.update_cell(last_row,5,round(data["RT"].mean(),2))
    worksheet.update_cell(last_row,6,data["FAV"].max())
    worksheet.update_cell(last_row,7,data["RT"].max())
    worksheet.update_cell(last_row,8,data["FAV"].min())
    worksheet.update_cell(last_row,9,data["RT"].min())
    worksheet.update_cell(last_row,10,round(data["FOLLOWER"].mean(),2))

    return text