import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

df= pd.read_csv("car_data.csv")
df_num = df.select_dtypes(include="number")
df_num
for i in range(0,2):
    Q1 = df_num.quantile(0.25)
    Q3 = df_num.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[~((df_num < lower_bound) | (df_num > upper_bound)).any(axis=1)]

X=df[['Kilometers Driven']]
y=df['Selling Price']


scaler = MinMaxScaler()
X_num_scaled = scaler.fit_transform(X)
X_num_scaled = pd.DataFrame(X_num_scaled, columns=X.columns, index=X.index)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lm = LinearRegression()
model = lm.fit(X_train,y_train)

yhat_train = model.predict(X_train)

rsq=model.score(X_train,y_train)
rsq

st.header("the performance of an model fiited")

yhat_test = model.predict([[X_test]])

st.write("b0 is", round(model.intercept_, 3))
st.write("b1 is", round(model.coef_[0], 3))
st.write("yhat test is", yhat_test)

