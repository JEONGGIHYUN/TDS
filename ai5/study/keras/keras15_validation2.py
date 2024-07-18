import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array(range(1,17))
y = np.array(range(1,17))

x_train = x[0:10]
y_train = y[0:10]

x_val = x[10:13]
y_val = y[10:13]

x_test = x[13:17]
y_test = y[13:17]

print(x_val)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=5,
          verbose=1
          ,validation_data=(x_val,y_val)) # verbose의 디폴트 값은 1 이다.



# verbose0 : 침묵
# verbose1 : 디폴트
# verbose2 : 프로그래스바 삭제
# verbose4 : 에포만 나온다

#4. 평가 예측
print('==========================================')
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print('로스 :', loss)
print('11의 결과값 :', results)
