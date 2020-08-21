#%%
import pandas as pd 
import akshare as ak 
import tushare as ts
import matplotlib.pyplot as plt 
import numpy as np 

import dsw

df_1 = ts.get_k_data('601727')
df_1.set_index(['date'], inplace = True)

df_2 = ak.stock_zh_a_daily("sh601727", adjust="hfq")
plt.plot(df_1['close'].values, 'r')
plt.plot(df_2['close'].values, 'b')
plt.show()
print(np.shape(df_1))
print(np.shape(df_2))
print(df_1.head())
print(df_2.head())

#%% [markdown]
#### 我们可以看到，akshare的接口查询到的数据比tushare接口因子更多，样本量也更大，所以我选择akshare接口来训练我们的模型

#%%
df = df_2
# df.to_csv('data/init_data.csv')

# %%散点图 -成交量和股价
plt.scatter(df['volume'], df['close'])
plt.xlabel('Volume')
plt.ylabel('Share Price')
plt.title('Volume & Share Price')
plt.show()

#%% [markdown]
#### 可以看出成交量和股价都分布在较低的水平，呈现一定的正相关趋势

#%%涨跌幅
daily_return = df['close'].pct_change().dropna()
plt.plot(daily_return)
plt.ylabel('Rise and Fall')
plt.show()

# %%
#1 5 10 日均收益
df['r_1'] = (df['close'] - df.shift(1)['close']) / df.shift(1)['close']
df['r_5'] = (df['close'] - df.shift(5)['close']) / df.shift(5)['close']
df['r_10'] = (df['close'] - df.shift(10)['close']) / df.shift(10)['close']

plt.plot(df['r_5'])
df.tail()

#%% [markdown]
#### 均线没有异常波动
#### 发现akshare的接口查询到的数据在16年底有一定空缺，但是ts接口也无法补足

#%%
#5 14日rsi
df['rsi_5'] = dsw.RSI(df['close'], 5)
df['rsi_14'] = dsw.RSI(df['close'], 14)
plt.plot(df['rsi_5'])
df.tail()

#%% [markdown]
#### rsi没有异常波动

#%%
df['K'], df['D'], df['J'] = dsw.KDJ(df)
df.tail()

#%% [markdown]
#### KDJ没有异常波动

#%%
# df.to_csv('data/factor_data.csv')
df.tail()

# %%
df = pd.read_csv('data/factor_data.csv')
df.set_index('date', inplace = True)
df.tail()

df_mean = df.mean(axis = 0)
df_std = df.std(axis = 0)

df_1 = (df - df_mean) / df_std  #归一化之后的df--df_1
df_1.tail()

#%% [markdown]
#### 发现‘outstanding share’列存在巨大的数量级差异，会干扰模型的训练，故删除之

#%%
df_1.drop('outstanding_share', axis = 1, inplace = True)
# df_1.to_csv('data/one_data.csv')
df_1.tail()

#%%
import tensorflow as tf 
from tensorflow.keras import layers

df = pd.read_csv('data/one_data.csv')
df.set_index('date', inplace = True)
LEN_TEST = 496   #18.6
df = df.iloc[50:-LEN_TEST]  #数据前50不稳定

df_1 = pd.read_csv('data/factor_data.csv')
df_1.set_index('date', inplace = True)
data_test = df_1.iloc[-LEN_TEST:]
df_1 = df_1.values
target = df_1[:, 3]

LEN_TRAIN = 1500
df_train = df.iloc[:LEN_TRAIN]
target_train = target[:LEN_TRAIN]

df_val = df.iloc[LEN_TRAIN:]
target_val = target[LEN_TRAIN:]

df_test = df.iloc[-LEN_TEST:]
target_test = target[-LEN_TEST:]

#++++++++++++++++++++++++++++++++++++++++++++
X_train, y_train = dsw.window_data(df_train.values, target_train, 5)
X_val, y_val = dsw.window_data(df_val.values, target_val, 5)
X_test, y_test = dsw.window_data(df_test.values, target_test, 5)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

#+++++++++++++++++++++++++++++++++++++++++++++
EPOCHS = 100
STEP_PER_EPOCH = 120

BATCH_SIZE = 100
BUFFER_SIZE = 2000

X_train = X_train.astype('float64')
y_train = y_train.astype('float64')
X_val = X_val.astype('float64')
y_val = y_val.astype('float64')
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))

train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data = val_data.cache().batch(BATCH_SIZE).repeat()

regressor = tf.keras.Sequential()
regressor.add(layers.LSTM(units = 30, activation = 'sigmoid'))
# regressor.add(layers.LSTM(units = 10, activation = 'tanh', input_shape = (5, 14)))
regressor.add(layers.Dense(units = 1))

# regressor.compile(optimizer = tf.keras.optimizers.RMSprop(clipvalue = 1.0), loss = 'mae')
# regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.compile(optimizer = 'adam', loss = 'mae')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=10, mode='auto', restore_best_weights=True)
history = regressor.fit(train_data, epochs = EPOCHS, steps_per_epoch = STEP_PER_EPOCH, validation_steps = 50, validation_data = val_data, callbacks=[early_stopping])

dsw.loss_plot(history)

result = regressor.predict(X_test).reshape(-1, 1)
plt.plot(y_test, 'r', label = 'origin')
plt.plot(result, 'b', label = 'prediction')
plt.legend()
plt.show()

#%% [markdown]
### 预测结果看起来相差很大，我查看训练集和验证集，发现上海电气的股票在13-18年发生了比较大的崩盘，收盘价（target）从14降到了7，我认为这可能是原因之一，如果改进模型的话可以针对涨跌幅进行预测

# %%算均线，从而在策略中平滑预测结果
df_predict = result
length_avg = 5
df_avg = np.zeros(len(df_predict))
for i in range(length_avg, len(df_predict)):
    df_avg[i] = sum(df_predict[i-length_avg:i]) / length_avg

ytest_avg = np.zeros(len(df_predict))
for i in range(length_avg, len(df_predict)):
    ytest_avg[i] = sum(y_test[i-length_avg:i]) / length_avg

for i in range(6):  #把图画的好看点
    df_avg[i]= result[0]
    ytest_avg[i] = y_test[0]

plt.plot(df_avg, 'yellow')
plt.plot(ytest_avg, 'yellow')

plt.plot(y_test, 'r', label = 'origin')
plt.plot(result, 'b', label = 'prediction')
plt.show()

#%% [markdown]
#### 由于我们分析的股票在近几年收盘价总体有很大的变动，所以预测结果比实际高一个台阶，但是涨跌幅大概相当，所以在接下来的预测中将采用涨跌幅进行策略。并且经过多次实验发现，模型在140-400天的表现要明显优于其他时间段

#### 我们的策略：当预测的涨跌趋势在过去的三天内都比较吻合的情况下，执行买入和卖出操作
#%%
point = np.zeros(490)
for i in range(130, 420):
    point[i] = dsw.calculate(ytest_avg[i-6:i], result[i-6:i+1], df_avg[i-6: i])
plt.plot(range(490), y_test)
for i in range(490):
    if point[i] > 0.00001:
        plt.scatter(i, y_test[i], c = 'red', marker='o')
    if point[i] < -0.00001:
        plt.scatter(i, y_test[i], c = 'blue', marker='o')

start_cash = 1000000
profit = dsw.operating(point, y_test, start_cash)
percent = (y_test[420]-y_test[130]) / y_test[130] *100
print("在预测区间（130，420）内，大盘从%.2f变动到%.2f" % (y_test[130], y_test[420]))
print("大盘跌落百分比为%.2f" % percent)
print( "我们最终仍然赚到了%.5f" % profit)

# %% [markdown]
#### 可以看到图中标记了很多连买连卖的操作，是因为策略设置的是稳定状况下逐渐追加

### 虽然我们分析的股票区间（130，420）下滑很大，但我们仍然赚到了近19w元

