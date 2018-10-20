import quandl
import pandas as pd
import numpy as np

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