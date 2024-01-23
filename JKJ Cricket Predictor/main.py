import main1
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

country = input('Enter the country name: - ')
dataset1 = pd.read_excel(f'D:\\JKJ Cricket Predictor\\Statistics\\Stats{country.lower()}.xlsx').to_csv(
    'D:\\JKJ '
    'Cricket '                                                                                             'Predictor'
    '\\Statistics_CSV\\Statistics')
dataset = pd.read_csv('D:\\JKJ Cricket Predictor\\Statistics_CSV\\Statistics')
print('Statistics of ', country, ': - ')
print(dataset)
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
Y = y.reshape(-1, 1)
print('\n')
years = dataset.iloc[:, 1].values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(Y)
Y = imputer.transform(Y)
"""
cmap = 'tab20'
hm = sn.heatmap(X)
plt.show()
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print('Training set data of independent variables: - ')
print(X_train)
print('\n')
print('Training set data of dependent variables: - ')
print(Y_train)
print('\n')
print('Testing set data of independent variables: - ')
print(X_test)
print('\n')
print('Testing set data of dependent variables: - ')
print(Y_test)
print('\n')
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
print('Average Strike Rate Predictions made using testing set data of independent variables from dataset: - ')
print(y_pred)
print('R2 Score of the predictions: - ', r2_score(Y_test, y_pred))

labels = 'Wins', 'Losses', 'Tied/No Result'

plt.figure(figsize=(9, 9))
X1 = dataset.iloc[:, 4:7].sum().values
plt.pie(X1, labels=labels, autopct='%1.1f%%', startangle=140)
st = 'Pie Chart depicting scores of ' + country
plt.title(st)
plt.show()

runrate = dataset.iloc[:, -1]
st1 = 'Average Strike Rate of ' + country
plt.xlabel('Years')
plt.ylabel('Runrate')
x = []
for i in range(len(years)):
    x.append(i)
plt.plot(dataset.iloc[:, 0], runrate)
plt.title(st1)
plt.xticks(x, years, rotation=45)
plt.show()
os.remove('D:\\JKJ Cricket Predictor\\Statistics_CSV\\Statistics')

c = years
a, b = np.polyfit(x, Y, 1)
plt.scatter(x, Y)
plt.plot(x, a * x + b)
plt.title(st1)
plt.xticks(x, years, rotation=45)
plt.show()

print('\n')
n = int(input('Do you want to calculate the average strike rate for the country? Enter 1 for yes and 0 for no: - '))
try:
    if n == 0:
        print('Thank you for using JKJ Cricket Predictor')
    elif n == 1:
        mp = int(input('Enter the no of matches played: - '))
        won = int(input('Enter the no of matches won: - '))
        lost = int(input('Enter the no of matches lost: - '))
        tied = int(input('Enter the no of matches tied: - '))
        avg = regressor.predict([[mp, won, lost, tied]])
        print('Average Strike Rate: - ', avg)

except:
    if n != 0 or n != 1:
        print(
            "An error has occurred while calculating the average strike rate value or an invalid value is input "
            "by the"
            "user")
