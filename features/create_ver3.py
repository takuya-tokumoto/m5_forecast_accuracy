from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from nyaggle.feature_store import cached_feature
from tqdm import tqdm

INPUT_DIR = "data/input/"
timesteps = 20
startDay = 0
TrTestWin = 56

@cached_feature("pre_sale_val", INPUT_DIR)
def create_pre_sale_val(sales_train_validation):
    print("prepare pre_sale_val")
    dt = sales_train_validation
    dt.index = dt.id
    dt = dt.T
    pd_SelesTrain = dt[6 + startDay:]
    
    pd_SelesTrain = pd_SelesTrain.reset_index(drop = True)
    
    return pd_SelesTrain


@cached_feature("former_agg_event", INPUT_DIR)
def create_calendar_event(calendar):
    print("prepare event_calendar")
    
    event_1_dummy = pd.get_dummies(calendar["event_name_1"].astype("category"), prefix='', prefix_sep='')
    event_2_dummy = pd.get_dummies(calendar["event_name_2"].astype("category"), prefix='event_name_2')

    agg_event = pd.concat([event_1_dummy, event_2_dummy], axis = 1)
    agg_event = pd.concat([agg_event,  calendar["date"]], axis = 1)
    
    agg_event["Easter"] = agg_event["Easter"] + agg_event["event_name_2_Easter"]
    agg_event["Cinco De Mayo"] = agg_event["Cinco De Mayo"] + agg_event["event_name_2_Cinco De Mayo"]
    agg_event["OrthodoxEaster"] = agg_event["OrthodoxEaster"] + agg_event["event_name_2_OrthodoxEaster"]
    agg_event["Father's day"] = agg_event["Father's day"] + agg_event["event_name_2_Father's day"]

    agg_event = agg_event.rename(columns={
                              'Cinco De Mayo' : 'Cinco_De_Mayo', 
                              "Father's day"  : 'Father_day', 
                              'Chanukah End'  :'Chanukah_End',
                              'Eid al-Fitr'   : 'Eid_al_Fitr', 
                              "Pesach End"    :"Pesach_End", 
                              "Purim End"     :"Purim_End",
                              'Ramadan starts':'Ramadan_starts',
                              "Mother's day"  :"Mother_day"
                             })

    agg_event = agg_event.drop([
                    "event_name_2_Easter", 
                    "event_name_2_Cinco De Mayo",
                    "event_name_2_OrthodoxEaster", 
                    "event_name_2_Father's day", 
                    'NBAFinalsEnd',
                    'NBAFinalsStart', 
                    'Ramadan_starts'
                    ], axis = 1)
    
    nba_finals_dates = [
        "2011-05-31", "2011-06-02", "2011-06-05", "2011-06-07", "2011-06-09", "2011-06-12", 
        "2012-06-12", "2012-06-14", "2012-06-17", "2012-06-19", "2012-06-21", 
        "2013-06-06", "2013-06-09", "2013-06-11", "2013-06-13", "2013-06-16", "2013-06-18", "2013-06-20", 
        "2014-06-05", "2014-06-08", "2014-06-10", "2014-06-12", "2014-06-15", 
        "2015-06-04", "2015-06-07", "2015-06-09", "2015-06-11", "2015-06-14", "2015-06-16", 
        "2016-06-02", "2016-06-05", "2016-06-08", "2016-06-10", "2016-06-13", "2016-06-16", "2016-06-19", 
    ]

    pd_nba_finals_dates = pd.DataFrame(nba_finals_dates, columns = ["date"])
    pd_nba_finals_dates["nba_finals_flag"] = 1
    
    agg_event = pd.merge(agg_event, pd_nba_finals_dates, on = "date", how = "left")

    agg_event = agg_event.drop("date", axis = 1)
    
#　イベント日付を1日前に設定
    former_agg_event = agg_event.shift(-1)
    former_agg_event = former_agg_event.fillna(0)
    
    return former_agg_event


@cached_feature("wday_info", INPUT_DIR)
def create_wday(calendar):
    print("prepare wday info")
    
    calendar = calendar["wday"]
    wday_info = pd.get_dummies(calendar["wday"].astype("category"), prefix='wday_', drop_first = True)
    
    return wday_info


@cached_feature("month_info", INPUT_DIR)
def create_month_info(calendar):
    print("prepare month info")
    
    calendar = calendar["date"]
    
    calendar["date"]  = calendar["date"].astype('datetime64[ns]')
    calendar["month"] = calendar["date"].dt.month
    
    month_info = pd.get_dummies(calendar["month"].astype("category"), prefix='month_', drop_first = True)
    
    return month_info
    

@cached_feature("day_info", INPUT_DIR)
def create_day_info(calendar):
    print("prepare day info")
    
    calendar = calendar["date"]
    
    calendar["date"] = calendar["date"].astype('datetime64[ns]')
    calendar["day"]   = calendar["date"].dt.day
    
    day_info = pd.get_dummies(calendar["day"].astype("category"), prefix='day_', drop_first = True)
    
    return day_info


if __name__ == "__main__":
    ROOT_DIR = ""
    INPUT_DIR = ROOT_DIR + "data/input/"
    sales_train_validation = pd.read_feather(INPUT_DIR + "sales_train_validation.feather")
    calendar = pd.read_feather(INPUT_DIR + "calendar.feather")

    create_pre_sale_val(sales_train_validation)
    create_calendar_event(calendar)
    create_wday(calendar)
    create_month_info(calendar)
    create_day_info(calendar)