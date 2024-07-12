from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

epochs = 100
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=epochs, batch_size=3)

loss = model.evaluate(x,y)
print('epochs :', epochs)
print('로스 :', loss)
result = model.predict(np.array([3]))
print('3번째 숫자의 예측값 :', result)
