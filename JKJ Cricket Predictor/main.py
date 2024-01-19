import main1
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import seaborn as sn

country = input('Enter the country name: - ')
dataset1 = pd.read_excel(f'D:\\JKJ Cricket Predictor\\Statistics_Cricket\\Stats{country}.xlsx').to_csv('D:\\JKJ '
                                                                                                      'Cricket '                                                                                             'Predictor'
                                                                                                      '\\Statistics_CSV\\Statistics')
dataset = pd.read_csv('D:\\JKJ Cricket Predictor\\Statistics_CSV\\Statistics')
print(dataset)
X = dataset.iloc[:, 3:-2].values
y = dataset.iloc[:, -1].values
Y = y.reshape(-1, 1)
# print(X)
# print(y)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(Y)
Y = imputer.transform(Y)
cmap = 'tab20'
hm = sn.heatmap(X)
plt.show()
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
print(y_pred)
print(r2_score(Y_test, y_pred))

labels = 'Wins', 'Losses', 'Tied/No Result'

plt.figure(figsize=(9, 9))
if country.lower() == 'india':
    X1 = dataset.iloc[:, 3:7].sum().values
else:
    X1 = dataset.iloc[:, 4:7].sum().values
plt.pie(X1, labels=labels, autopct='%1.1f%%', startangle=140)
st = 'Pie Chart depicting scores of ' + country
plt.title(st)
plt.show()

runrate = dataset.iloc[:, -1]
st1 = 'Run rate of ' + country
plt.plot(dataset.iloc[:, 0], runrate)
plt.xlabel('Years')
plt.ylabel('Runrate')
plt.title(st1)
plt.show()
os.remove('D:\\JKJ Cricket Predictor\\Statistics_CSV\\Statistics')
