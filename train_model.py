from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

data = pd.read_csv("insurance_data_linear.csv")

print(data.columns)

X = data[['age']]
y = data['charges']   

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained!")