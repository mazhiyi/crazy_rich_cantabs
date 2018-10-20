import quandl
import pandas as pd
import numpy as np

import datetime

from utils import get_event_keys_mapping, daterange

quandl.ApiConfig.api_key = 'jTAiyyEPyJkAw1fxpj2_'




def get_demo_simple_dense_data():
    # Return format: X, y
    train_table = quandl.get_table('SHARADAR/SF1', calendardate='2010-12-31', ticker='ZZ')
    train_X = train_table[['assets', 'sps']]/100000000
    train_y = train_table['workingcapital']/100000000

    test_table = quandl.get_table('SHARADAR/SF1', calendardate='2011-12-31', ticker='ZZ')
    test_X = test_table[['assets', 'sps']]/100000000
    test_y = test_table['workingcapital']/100000000

    return(train_X, train_y, test_X, test_y)


def _get_event_table(dimension, year, quarter=None):
    quarter_dates = ['12-31','03-31', '06-30', '09-30']
    start_year = year
    if quarter == 0:
        start_year -= 1
    start_dt = datetime.datetime.strptime(str(start_year) + '-' + quarter_dates[quarter], "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(str(year) + '-' + quarter_dates[(quarter+1)%4], "%Y-%m-%d")
    table = quandl.get_table('SHARADAR/EVENTS', date=list(daterange(start_dt, end_dt)), paginate=True)
    return table

def get_event_data(dimension, year, quarter=None):
    event_table = _get_event_table(dimension, year, quarter)
    EVENT_KEYS_MAPPING = get_event_keys_mapping()
    X_dict = {}
    for _, row in event_table.iterrows():
        if row['ticker'] not in X_dict:
            X_dict[row['ticker']] = np.zeros(len(EVENT_KEYS_MAPPING))
        events = [int(x) for x in row['eventcodes'].split('|')]
        for x in events:
            X_dict[row['ticker']][EVENT_KEYS_MAPPING[x]] += 1
    return pd.DataFrame.from_dict(X_dict,orient='index')

def API(year, quarter, dimension = 'ARQ'):

    df = _get_price_table(year, quarter)

    df = df[df['dimension'] == dimension]
    df = df[['ticker', 'price', 'calendardate']]
    df.dropna(inplace = True)
    df = df.reset_index(drop=True)

    names = df.ticker.unique()
    delta = {}
         
    for i, row in df.iterrows():
        if i == 0:
            dummy_name = row['ticker']
            p2 = row['price']
        else:
            if dummy_name == row['ticker']:
                p1 = row['price']
                ratio = (p2 - p1)/p1
                delta[dummy_name] = ratio
                dummy_name = row['ticker']
            else:
                dummy_name = row['ticker']
                p2 = row['price']
    return delta


