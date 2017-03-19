# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:21:50 2017

@author: Satheesh
"""

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []


def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            print(row)
            prices.append(float(row[1]))
        print('hello')
    return
    
def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    #C is the penalty parameter in SVR
    print('Creating a linear svr model')
    svr_lin = SVR(kernel = 'linear', C=1e3)
    print('creating a polynomial svr model')
    svr_poly = SVR(kernel = 'poly', C=1e3, degree = 2)
    print('creating a rbf svr model')
    #radio basis function gamma defines how far two points should be apart from each other to be classified as 0
    svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma = 0.1)
    print('Fitting data to linear model')
    #fitting data to the models
    svr_lin.fit(dates, prices)
    print('Fitting data to polynomial model')
    svr_poly.fit(dates, prices)
    print('Fitting data to rbf model')
    svr_rbf.fit(dates, prices)
    print('model trained')
    
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='red', label='Linear')
    plt.plot(dates, svr_poly.predict(dates), color='green', label='Poly')
    plt.plot(dates, svr_rbf.predict(dates), color='blue', label='RBF')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regressions')
    plt.legend()
    print('About to display graph')
    plt.show()
    
    return svr_lin.predict(x)[0], svr_poly.predict(x)[0], svr_rbf.predict(x)[0]


get_data('aapl.csv')
predicted_price = predict_prices(dates, prices, 15)
print(predicted_price)

