# https://dacon.io/competitions/official/235576/overview/description 대회 주소

import numpy as np
import pandas as pd # 인덱스와 칼럼을 분리하는데 사용하는 함수
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = './_data/따릉이/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0) # [715 rows x 1 columns]
print(submission_csv) # NaN으로 나오는 데이터는 없는 데이터를 뜻한다. 결측치 : 일반적인 데이터 집합에서 벗어난다는 뜻을 가진 이상치(outlier)의 하위 개념

print(train_csv.shape) # (1459, 10)
print(test_csv.shape) # (715, 9)
print(submission_csv.shape) # (715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
    #   dtype='object')

print(train_csv.info())

############ 결측치 처리 1. 삭제 ##############
# train_csv.isnull().sum()
print(train_csv.isna().sum())

train_csv = train_csv.dropna() 
print(train_csv.isna().sum())
print(train_csv) # [1328 rows x 10 columns]

print(test_csv.info())

test_csv = test_csv.fillna(test_csv.mean()) # 결측치 채우기 
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
print(x) # [1328 rows x 9 columns]

y = train_csv['count']
print(y.shape) # (1328, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=4343)

#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=9))
model.add(Dense(250))
model.add(Dense(125))
model.add(Dense(72.5))
model.add(Dense(50))
model.add(Dense(1))



#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000,
          verbose=0,
          validation_split=0.5)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)
# r2 = r2_score(y_test,  results)
# print('로스 :', loss)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) # (715, 1)


######### submission.csv 만들기 // count 클럼에 값만 넣어 주면 된다. ####
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)
r2 = r2_score(y_test,  results)
print('로스 :', loss)


submission_csv.to_csv(path + 'submission_0716_1914csv')



# 로스 : 2506.734375 train .87 random 434
# 로스 : 2447.269775390625 train .9 random 434
# 로스 : 2423.58203125 train .9 random 434
# 로스 : 2201.45849609375 train .9 random 4343434
# 로스 : 1739.7637939453125 train .9 random 4343
# 로스 : 1696.930419921875  train .9 random 4343 222 9 300 30 25 5 1 1000 
# 로스 : 1518.3536376953125 train .96 random 4343 222 9 300 30 25 5 1 1000 
# 로스 : 1488.0318603515625 train .98 random 4343 222 9 300 30 25 5 1 1000 
# 로스 : 1403.3687744140625 train .9875 random 4343 222 9 300 30 25 5 1 1000
# 로스 : 1368.3687744140625 train .9875 random 4343 222 9 300 30 25 5 1 1000
# 로스 : 1362.5675048828125 train .9822 random 4343 222 9 300 30 25 5 1 1000
# 로스 : 1361.5626220703125 train .9822 random 4343 222 9 300 30 25 5 1 1000
# 로스 : 1313.7088623046875 train .9822 random 4343 500 250 125 72.5 50 1 1000
# 로스 : 1282.2501220703125 train .9822 random 3535 500 250 125 72.5 50 1 1000
# 로스 : 1134.4781494140625 train .9822 random 5757 500 250 125 72.5 50 1 1000
# 로스 : 1016.1388549804688 train .9822 random 5757 500 250 125 72.5 50 1 1000
# 로스 : 1703.3221435546875 train .9 random 4343 222 9 300 30 25 5 1 1000 