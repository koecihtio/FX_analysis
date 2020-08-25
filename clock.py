from apscheduler.schedulers.blocking import BlockingScheduler
import words

twische = BlockingScheduler()
@twische.scheduled_job('interval',minutes=30)
def timed_job():
    import datetime
    import matplotlib.pyplot as plt
    import pandas_datareader.data as pdr
    import numpy as np
    import pandas as pd
    import time
    from dateutil.relativedelta import relativedelta
    from datetime import datetime, date, timedelta,timezone
    import os
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    import seaborn as sns
    from sklearn import preprocessing
    import config
    from twitter import Twitter, OAuth
    import tweepy
    import pytz
    from oauth2client.service_account import ServiceAccountCredentials 
    import gspread
    import json
    today = date.today()
    day = datetime.now().weekday()
    dt = datetime.now(pytz.timezone('Asia/Tokyo')).timetz()
    sixhalf = datetime(2018,2,1,6,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    sevenat = datetime(2018,2,1,7,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    sevenhalf = datetime(2018,2,1,7,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    eightat = datetime(2018,2,1,8,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    eighthalf =datetime(2018,2,1,8,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    nineat = datetime(2018,2,1,7,9,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    ninehalf = datetime(2018,2,1,9,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    elevenhalf = datetime(2018,2,1,11,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    twelveat = datetime(2018,2,1,12,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    twelvehalf = datetime(2018,2,1,12,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    thirteenat  = datetime(2018,2,1,13,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    fifteenat = datetime(2018,2,1,15,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    fifteenhalf = datetime(2018,2,1,15,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    sixteenat = datetime(2018,2,1,16,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    sixteenhalf = datetime(2018,2,1,16,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    seventeenat = datetime(2018,2,1,17,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    seventeenhalf = datetime(2018,2,1,17,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    eighteenat = datetime(2018,2,1,18,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    eighteenhalf = datetime(2018,2,1,18,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    nineteenat = datetime(2018,2,1,19,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    nineteenhalf = datetime(2018,2,1,19,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    twentyat = datetime(2018,2,1,20,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    twentyhalf = datetime(2018,2,1,20,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    twentyoneat = datetime(2018,2,1,21,0,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    twentyonehalf = datetime(2018,2,1,21,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    twentytwoat =  datetime(2018,2,1,22,00,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    twentythreeat =  datetime(2018,2,1,23,00,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()
    twentythreehalf = datetime(2018,2,1,23,30,tzinfo=pytz.timezone('Asia/Tokyo')).timetz()


    if day >= 0 and day <= 4:
        CK=os.environ["CONSUMER_KEY"]
        CS=os.environ["CONSUMER_SECRET"]
        AT=os.environ["ACCESS_TOKEN_KEY"]
        AS=os.environ["ACCESS_TOKEN_SECRET"]
        auth = tweepy.OAuthHandler(CK, CS)
        auth.set_access_token(AT, AS)
        api = tweepy.API(auth)
        if sixhalf < dt and dt <= sevenat:
            text = words.content_corrlation()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif sevenat< dt and dt <= sevenhalf:
            text,caption = words.gold_predict(k=10)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif sevenhalf < dt and dt <= eightat:
            text,caption = words.usdjpy_predict(k=10)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif eightat < dt and dt <= eighthalf:
            text,caption = words.nikkei_predict(k=10)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif eighthalf < dt and dt <= nineat:
            text,caption = words.audusd_predict(k=10)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif nineat < dt and dt <= ninehalf:
            text,caption = words.usdcad_predict(k=10)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif elevenhalf < dt and dt <= twelveat:
            text,caption = words.btc_predict(k=10)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif twelvehalf < dt and dt <= thirteenat:
            text,caption = words.dow_predict(k=10)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif fifteenat < dt and dt <= fifteenhalf:
            text,caption = words.vix_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif fifteenhalf < dt and dt <= sixteenat:
            text,caption = words.usdchf_predict(k=10)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif sixteenat < dt and dt <= sixteenhalf:
            text,caption = words.eurusd_predict(k=10)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif sixteenhalf < dt and dt <= seventeenat:
            text,caption = words.gbpusd_predict(k=10)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif seventeenat < dt and dt <= seventeenhalf:
            text,caption = words.eurusd_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif seventeenhalf < dt and dt <= eighteenat:
            text,caption = words.gbpusd_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif eighteenat < dt and dt <= eighteenhalf:
            text,caption = words.usdchf_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif eighteenhalf < dt and dt <= nineteenat:
            text,caption = words.wti_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif nineteenat < dt and dt <= nineteenhalf:
            text,caption = words.cadusd_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif nineteenhalf < dt and dt <= twentyat:
            text,caption = words.audusd_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif twentyat < dt and dt <= twentyhalf:
            text,caption = words.nikkei_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif twentyhalf < dt and dt <= twentyoneat:
            text,caption = words.usdjpy_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif twentyoneat < dt and dt <= twentyonehalf:
            text,caption = words.dow_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        elif twentyonehalf < dt and dt <= twentytwoat:
            text,caption = words.gold_volatility()
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
        else:
            pass
     else:
        pass
    if day >= 1 and day <= 5:
        if twentythreeat < dt and dt <= twentythreehalf:
            text = words.get_tweets()
            CK=os.environ["CONSUMER_KEY"]
            CS=os.environ["CONSUMER_SECRET"]
            AT=os.environ["ACCESS_TOKEN_KEY"]
            AS=os.environ["ACCESS_TOKEN_SECRET"]
            auth = tweepy.OAuthHandler(CK, CS)
            auth.set_access_token(AT, AS)
            api = tweepy.API(auth)
            api.update_with_media(status = text, filename = 'img.png')
            os.remove("img.png")
            plt.close()
if __name__ == "__main__":
    twische.start()