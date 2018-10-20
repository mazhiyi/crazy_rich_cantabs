import quandl
import pandas as pd
import numpy as np

quandl.ApiConfig.api_key = 'jTAiyyEPyJkAw1fxpj2_'

def get_demo_simple_dense_data():
    # Return format: X, y
    table = quandl.get_table('SHARADAR/SF1', calendardate='2011-12-31', ticker='ZZ')
    assets = table[['assets', 'sps']]/100000000
    workingcapital = table['workingcapital']/100000000
    return(assets, workingcapital)