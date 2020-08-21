
import pandas as pd 
import numpy as np 
import akshare as ak 
import datetime 
import matplotlib.pyplot as plt
import tensorflow as tf 

def datef_p_strf(data, date_format):     #date-dateformat时  导出str-rawtime
    dates = data['date']  
    raw_time = []

    for i in dates:
        raw_time.append(i.strftime(date_format))
    return raw_time

def RSI(t, periods=10):
    length = len(t)
    rsies = [np.nan]*length
    #用于快速计算；
    up_avg = 0
    down_avg = 0

    #首先计算第一个RSI，用前periods+1个数据，构成periods个价差序列;
    first_t = t.iloc[:periods+1]
    for i in range(1, len(first_t)):
        #价格上涨;
        if first_t[i] >= first_t[i-1]:
            up_avg += first_t[i] - first_t[i-1]
        #价格下跌;
        else:
            down_avg += first_t[i-1] - first_t[i]
    up_avg = up_avg / periods
    down_avg = down_avg / periods
    rs = up_avg / down_avg
    rsies[periods] = 100 - 100/(1+rs)

    #后面的将使用快速计算；
    for j in range(periods+1, length):
        up = 0
        down = 0
        if t[j] >= t[j-1]:
            up = t[j] - t[j-1]
            down = 0
        else:
            up = 0
            down = t[j-1] - t[j]
        #类似移动平均的计算公式;
        up_avg = (up_avg*(periods - 1) + up)/periods
        down_avg = (down_avg*(periods - 1) + down)/periods
        rs = up_avg/down_avg
        rsies[j] = 100 - 100/(1+rs)
    return rsies

def KDJ(data):
    price = data.copy()
    price['rolling_high'] = price['high'].rolling(window = 9, min_periods = 1).max()
    price['rolling_low'] = price['low'].rolling(window = 9, min_periods = 1).min()
    price['fastk'] = (price['close'] - price['rolling_low']) / (price['rolling_high'] - price['rolling_low']) * 100
    price['fastd'] = price['fastk'].ewm(com = 2, adjust = False).mean()
    price['K'] = price['fastd']
    price['D'] = price['K'].ewm(com = 2, adjust = False).mean()
    price['J'] = 3 * price['K'] - 2 * price['D']
    return price['K'], price['D'], price['J']

def window_data(data, target, history_size):
    #data 为np.array
    #target_size = 1
    features = []
    labels = []

    start_index = history_size
    end_index = len(data) - 1

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        features.append(data[indices])
        labels.append(target[i])
    
    return np.array(features), np.array(labels)

def loss_plot(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))
    plt.plot(epochs, loss, 'b', label = 'loss')
    plt.plot(epochs, val_loss, 'r', label = 'val_loss')
    plt.legend()
    plt.show()

def calculate(y_test, result, df_avg):
    y_test_gap = np.zeros(3)    
    df_avg_gap = np.zeros(3)
    for i in range(len(y_test)-3):
        y_test_gap[i] = y_test[i+3]-y_test[i]
        df_avg_gap[i] = result[i+3]-result[i]
    if y_test_gap[0] * df_avg_gap[0] >0 and\
         y_test_gap[1] * df_avg_gap[1] >0 and\
              y_test_gap[2] * df_avg_gap[2] >0:
        order = 1
    else:
        order = 0
    result_gap = result[6]-result[2]
    point = result_gap *order
    return point

def operating(point, y_test, start_cash):
    cash = start_cash
    stock = 0
    for i in range(1, len(point)-1):
        if point[i] > 0.0000001:
            if cash >= point[i] * 10000 *y_test[i-1]:
                cash -= point[i] * 10000 *y_test[i-1]
                stock += point[i] * 10000
            else:
                stock += cash / y_test[i-1]
                cash = 0
        elif point[i] < -0.0000001:
            if stock >= -10000 * point[i]:
                cash += point[i] * 10000 *y_test[i-1]
                stock -= point[i] * 10000
            else:
                cash += stock * y_test[i-1]
                stock = 0
    if stock > 0:
        cash +=stock *y_test[-1]
    return cash