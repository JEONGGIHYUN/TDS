import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

print(tf.__version__) #2.16.2

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1)) #인풋 아웃풋

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer ='adam') #컴파일
model.fit(x,y, epochs=300)

#4. 평가 예측
result = model.predict(np.array([4]))
print('4의 예측값 :', result)