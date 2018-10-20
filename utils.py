import numpy as np
from datetime import timedelta, date

EVENT_LIST = [11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 26, 32, 33, 34, 
              35, 36, 40, 41, 42, 51, 52, 53, 54, 55, 56, 58, 62, 64, 
              65, 71, 81, 91, 57,31]

def get_event_keys_mapping(event_list=EVENT_LIST):
    EVENT_KEYS_MAPPING = {}
    for i, key in enumerate(event_list):
        EVENT_KEYS_MAPPING[key] = i
    return EVENT_KEYS_MAPPING

def daterange(date1, date2):
    for n in range(int((date2 - date1).days)+1):
        yield str(date1 + timedelta(n))
    
def expand_for_conv(X):
    return np.array(X).reshape(X.shape[0], X.shape[1], 1)
